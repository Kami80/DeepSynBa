"""Independent numeric, checkpoint and factorial audits for the twelve new fits."""
import argparse
import gc
from pathlib import Path

from train_small import dump_json, seed_everything
import numpy as np
import pandas as pd
import torch
from activation_initialization import CONDITIONS, SEEDS, build_factorial_model, collect_outputs
from ablation_model import POSITIVE_KEYS
from analyze_extended import assert_metrics, independent_metrics, save_figure
from analyze_step_budget import verify_history
from prepare_small_data import sha256
from run_activation_initialization import load_protocol, verify_completed_run
from run_next_steps import make_loaders, read_json, validate_file

LABELS = {"relu_original": "ReLU / original", "softplus_original": "Softplus / original",
    "relu_near_one": "ReLU / near-one", "softplus_near_one": "Softplus / near-one"}
MEASURES = ("rmse", "combination_rmse", "pearson", "spearman", "updates", "best_update", "training_seconds", "peak_cuda_memory_mib")


def validate_cells(table):
    expected = {(condition["id"],seed) for condition in CONDITIONS for seed in SEEDS}
    keys = list(zip(table.condition, table.seed))
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("Every factorial cell needs all three distinct paired seeds")
    if not np.isfinite(table[["rmse","combination_rmse"]].to_numpy()).all():
        raise ValueError("Non-finite factorial metrics")
    if table.parameters.nunique() != 1:
        raise ValueError("The factorial cells have different parameter counts")


def summarize(table):
    validate_cells(table)
    result = []
    for condition in CONDITIONS:
        group = table.loc[table.condition == condition["id"]]
        row = {"condition":condition["id"],"activation":condition["activation"],
            "initialization":condition["initialization"],"parameters":int(group.parameters.iloc[0]),
            "cap_fits":int((group.stop_reason=="update_limit").sum())}
        for measure in MEASURES:
            row[measure+"_mean"] = float(group[measure].mean())
            row[measure+"_sd"] = float(group[measure].std(ddof=1))
        result.append(row)
    return pd.DataFrame(result)


def factorial_contrasts(table):
    validate_cells(table)
    rows=[]
    for metric in ("rmse","combination_rmse"):
        wide=table.pivot(index="seed",columns="condition",values=metric)
        for seed in SEEDS:
            r0,s0,r1,s1=[float(wide.loc[seed,name]) for name in
                ("relu_original","softplus_original","relu_near_one","softplus_near_one")]
            values={"activation_original":s0-r0,"activation_near_one":s1-r1,
                "initialization_relu":r1-r0,"initialization_softplus":s1-s0,
                "interaction":(s1-r1)-(s0-r0),"average_activation":0.5*((s0-r0)+(s1-r1)),
                "average_initialization":0.5*((r1-r0)+(s1-s0))}
            rows += [{"metric":metric,"seed":seed,"contrast":name,"change":value} for name,value in values.items()]
    paired=pd.DataFrame(rows)
    grouped=paired.groupby(["metric","contrast"],sort=False).change.agg(["mean","std"]).reset_index()
    grouped.columns=["metric","contrast","change_mean","change_sd"]
    return paired,grouped


def verify_diagnostics(folder, outputs, stage):
    recorded=read_json(folder/"diagnostics.json")[stage]
    for key in POSITIVE_KEYS:
        value=outputs[key].astype(np.float64).ravel()
        actual={"min":value.min(),"max":value.max(),"mean":value.mean(),
            "median":np.percentile(value,50),"exact_zero_fraction":np.count_nonzero(value==0)/len(value),
            "below_0_01_fraction":np.count_nonzero(value<0.01)/len(value),"examples":len(value)}
        for field,result in actual.items():
            np.testing.assert_allclose(result,recorded[key][field],rtol=1e-12,atol=1e-12)


def verify_checkpoint(record, protocol, arrays):
    seed_everything(record["seed"],8)
    folder=Path(record["folder"])
    checkpoint=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
    for key in ("seed","dataset_sha256","protocol_sha256","best_update"):
        if checkpoint[key]!=record[key]:
            raise ValueError(f"Checkpoint provenance differs: {key}")
    if checkpoint["condition"]!=record["condition"] or checkpoint["training_source_rows"]!=protocol["training_source_rows"]:
        raise ValueError("Checkpoint condition or training IDs differ")
    model,_=build_factorial_model(checkpoint["model_config"],checkpoint["condition"])
    model.to("cuda")
    loader=make_loaders(arrays,record["seed"])["val"]
    _,initial=collect_outputs(model,loader,"cuda")
    with np.load(folder/"initial_outputs.npz",allow_pickle=False) as saved:
        for key in initial:
            np.testing.assert_array_equal(initial[key],saved[key])
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    _,best=collect_outputs(model,loader,"cuda")
    with np.load(folder/"val_predictions.npz",allow_pickle=False) as saved:
        np.testing.assert_array_equal(best["dose_response"],saved["predicted"])
    with np.load(folder/"val_outputs.npz",allow_pickle=False) as saved:
        for key in POSITIVE_KEYS:
            np.testing.assert_array_equal(best[key],saved[key])
    if sum(p.numel() for p in model.parameters())!=record["parameters"]:
        raise ValueError("Reloaded parameter count differs")
    del model,checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def make_figures(summary,per_seed,diagnostics,output):
    import matplotlib.pyplot as plt
    colors={"relu":"#2166a5","softplus":"#b36b13"}
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,metric,title in zip(axes,("rmse","combination_rmse"),("All 16 validation entries","9 combination entries")):
        for activation in ("relu","softplus"):
            group=summary.loc[summary.activation==activation]
            ax.errorbar([0,1],group[metric+"_mean"],yerr=group[metric+"_sd"],
                color=colors[activation],marker="o" if activation=="relu" else "s",capsize=4,label="ReLU" if activation=="relu" else "Softplus")
            for row in group.itertuples():
                x=0 if row.initialization=="original" else 1
                vals=per_seed.loc[per_seed.condition==row.condition,metric]
                ax.scatter(np.full(3,x),vals,s=18,color=colors[activation],alpha=.45)
        ax.set_xticks([0,1],["Original","Near-one"])
        ax.set_xlim(-.18,1.18)
        ax.set_xlabel("Output initialization")
        ax.set_ylabel("Validation RMSE (original units)")
        ax.set_title(title)
        ax.grid(axis="y",alpha=.6)
    axes[0].legend(frameon=False)
    fig.suptitle("Activation x initialization: 2,000 training matrices")
    fig.text(.5,.01,"Means and sample SD over seeds 42/43/44; dots are individual seeds. No confidence intervals.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    save_figure(fig,output,"factorial_validation")
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.5),sharey=True)
    for ax,key in zip(axes,POSITIVE_KEYS):
        for stage,offset,color in (("initial",-.17,"#89959f"),("best",.17,"#2166a5")):
            values=[]
            for condition in CONDITIONS:
                observed=diagnostics.loc[(diagnostics["condition"]==condition["id"]) &
                    (diagnostics["stage"]==stage) & (diagnostics["head"]==key),"exact_zero_fraction"]
                if len(observed)!=3 or not np.isfinite(observed).all():
                    raise ValueError("Each diagnostic bar requires three finite seed values")
                values.append(float(observed.mean())*100)
            bars=ax.bar(np.arange(4)+offset,values,width=.32,color=color,label=stage.title())
            ax.bar_label(bars,labels=[f"{value:.1f}" if value>0 else "" for value in values],padding=3,fontsize=8)
        ax.set_xticks(range(4),["ReLU\noriginal","Softplus\noriginal","ReLU\nnear-one","Softplus\nnear-one"],fontsize=8)
        ax.set_title(key.replace("_mean",""))
        ax.set_ylim(0,110)
        ax.grid(axis="y",alpha=.4)
    axes[0].set_ylabel("Exactly zero validation head outputs (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("Initial and validation-selected parameter-head behavior")
    fig.text(.5,.01,"Average fractions across three seeds. Softplus + 0.0001 is positive by construction; nonzero is not biological validation.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    save_figure(fig,output,"head_zero_fractions")


def analyze(output,verify_checkpoints):
    protocol,arrays,_=load_protocol(output)
    completed=read_json(output/"completed.json")
    validate_file(output/"selection.json",completed["selection_sha256"])
    if completed["new_training_jobs"]!=12:
        raise ValueError("The factorial experiment is incomplete")
    rows=[];diag_rows=[];checks=[]
    val_rows=torch.where(arrays["split"]==2)[0]
    for record in completed["runs"]:
        verify_completed_run(record)
        folder=Path(record["folder"])
        history=verify_history(folder,record,protocol["hyperparameters"])
        with np.load(folder/"val_predictions.npz",allow_pickle=False) as saved:
            for key in ("source_row","dose1","dose2"):
                np.testing.assert_array_equal(saved[key],arrays[key][val_rows].numpy())
            np.testing.assert_array_equal(saved["observed"],arrays["response"][val_rows].numpy())
            observed,predicted=saved["observed"].copy(),saved["predicted"].copy()
        full=independent_metrics(observed,predicted)
        combo=independent_metrics(observed[:,1:,1:],predicted[:,1:,1:])
        assert_metrics(full,record["validation"]["all_16"])
        assert_metrics(combo,record["validation"]["combination_9"])
        row={"condition":record["condition"]["id"],"activation":record["condition"]["activation"],
            "initialization":record["condition"]["initialization"],"seed":record["seed"],"parameters":record["parameters"],
            **full,"combination_rmse":combo["rmse"],"updates":record["updates_completed"],
            "best_update":record["best_update"],"training_seconds":record["training_seconds"],
            "peak_cuda_memory_mib":record["peak_cuda_memory_mib"],"stop_reason":record["stop_reason"]}
        rows.append(row)
        for filename,stage in (("initial_outputs.npz","initial"),("val_outputs.npz","best")):
            with np.load(folder/filename,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["source_row"],protocol["validation_source_rows"])
                outputs={key:saved[key] for key in POSITIVE_KEYS}
                verify_diagnostics(folder,outputs,stage)
                for key in POSITIVE_KEYS:
                    diag_rows.append({"condition":row["condition"],"seed":row["seed"],"stage":stage,"head":key,
                        **read_json(folder/"diagnostics.json")[stage][key]})
        if record["condition"]["id"]=="softplus_near_one":
            control=next(item for item in protocol["controls"] if item["seed"]==record["seed"])
            with np.load(Path(control["folder"])/"val_predictions.npz",allow_pickle=False) as previous:
                np.testing.assert_array_equal(previous["predicted"],predicted)
                np.testing.assert_array_equal(previous["observed"],observed)
            columns=["update","train_rmse","val_rmse","lr","stale_evaluations"]
            np.testing.assert_allclose(history[columns],pd.read_csv(Path(control["folder"])/"history.csv")[columns],rtol=0,atol=1e-10)
            if record["control_exactly_reproduced"] is not True:
                raise ValueError("Missing successful control assertion")
        if verify_checkpoints:
            verify_checkpoint(record,protocol,arrays)
        checks.append({"condition":row["condition"],"seed":row["seed"],"prediction_metrics_verified":True,
            "history_verified":True,"head_diagnostics_verified":True,"initial_and_best_cuda_reload_exact":verify_checkpoints})
        print(f"AUDITED {row['condition']} seed={row['seed']}; CUDA reload={verify_checkpoints}",flush=True)
    table=pd.DataFrame(rows)
    summary=summarize(table)
    paired,contrasts=factorial_contrasts(table)
    # A separate matrix multiplication checks every per-seed contrast.
    order=[c["id"] for c in CONDITIONS]
    coefficients={"activation_original":[-1,1,0,0],"activation_near_one":[0,0,-1,1],
        "initialization_relu":[-1,0,1,0],"initialization_softplus":[0,-1,0,1],
        "interaction":[1,-1,-1,1],"average_activation":[-.5,.5,-.5,.5],
        "average_initialization":[-.5,-.5,.5,.5]}
    for metric in ("rmse","combination_rmse"):
        matrix=table.pivot(index="seed",columns="condition",values=metric).loc[list(SEEDS),order].to_numpy()
        for name,vector in coefficients.items():
            actual=paired.loc[(paired.metric==metric)&(paired.contrast==name)].sort_values("seed").change.to_numpy()
            np.testing.assert_allclose(actual,matrix@np.asarray(vector),rtol=0,atol=1e-12)
    rank=summary.sort_values(["rmse_mean","rmse_sd","condition"])
    if rank.iloc[0].condition!=read_json(output/"selection.json")["selected_condition"]["id"]:
        raise ValueError("Independent ranking differs")
    if table.updates.sum()!=completed["total_optimizer_updates"]:
        raise ValueError("Completed update accounting differs")
    analysis=output/"analysis";analysis.mkdir(exist_ok=True)
    table.to_csv(analysis/"validation_per_seed.csv",index=False)
    summary.to_csv(analysis/"validation_summary.csv",index=False)
    paired.to_csv(analysis/"factorial_contrasts_per_seed.csv",index=False)
    contrasts.to_csv(analysis/"factorial_contrasts_summary.csv",index=False)
    diagnostics=pd.DataFrame(diag_rows)
    diagnostics.to_csv(analysis/"head_diagnostics.csv",index=False)
    dump_json(analysis/"verification.json",{"run_audits":checks,"exact_reproducibility_controls":3,
        "independent_metrics":True,"independent_contrast_matrix_check":True,
        "analysis_source_sha256":sha256(Path(__file__)),
        "protocol_sha256":sha256(output/"protocol.json"),"data_contain_test_rows":False,
        "actual_total_updates":int(table.updates.sum())})
    dump_json(analysis/"summary.json",{"conditions":summary.to_dict(orient="records"),
        "contrasts":contrasts.to_dict(orient="records"),"selected_condition":rank.iloc[0].condition,
        "runs":12,"all_cuda_reloads_exact":verify_checkpoints,"completed_utc":completed["completed_utc"]})
    make_figures(summary,table,diagnostics,analysis)
    narrative=["# Activation and initialization — completed","",
        "Validation-only 2×2 factorial: the same 2,000 training matrices, 1,000 validation matrices, full architecture, augmentation and optimizer-update schedule; seeds 42/43/44.","",
        "| Activation / initialization | All-16 RMSE | Combination-9 RMSE | Cap fits |",
        "|---|---:|---:|---:|"]
    for row in summary.itertuples():
        narrative.append(f"| {LABELS[row.condition]} | {row.rmse_mean:.3f} ± {row.rmse_sd:.3f} | {row.combination_rmse_mean:.3f} ± {row.combination_rmse_sd:.3f} | {row.cap_fits}/3 |")
    narrative += ["","All ± values are sample SD over three paired seeds, not confidence intervals. Negative conditional/main effects favor Softplus or near-one calibration. Interaction measures how the activation effect changes with initialization.","",
        "| All-16 paired contrast | Mean change | Seed SD |","|---|---:|---:|"]
    for row in contrasts.loc[contrasts.metric=="rmse"].itertuples():
        narrative.append(f"| {row.contrast} | {row.change_mean:+.3f} | {row.change_sd:.3f} |")
    narrative += ["",f"Validation-ranked cell: **{rank.iloc[0].condition}**. All three Softplus/near-one controls exactly reproduce the preceding 2,000-matrix enhanced fits.","",
        "The calibration factor combines 0.01 weight scaling and an activation-specific bias targeting output one. It does not isolate bias from scale. Early stopping changes realized updates; all cells share the same maximum/checkpoint/stopping rule. Initial and selected h1/h2/alpha outputs are recorded, with exact zeros and below-0.01 fractions. Softplus positivity is built into the function and is not proof of biological parameter recovery.","",
        "This experiment describes algorithmic changes under a fixed adaptive validation protocol; it supplies neither a new holdout claim, a generalization/significance interval nor split/source robustness. Both previously inspected holdouts remain development evidence. Independent-source evaluation, strong external baselines, multiple splits and drug/cell generalization remain future research.","",
        f"Actual optimizer updates: {int(table.updates.sum()):,}. Summed timed training: {completed['new_training_seconds']/60:.1f} minutes. Histories, predictions, diagnostics and protocol/data hashes audited; exact initial and selected CUDA reloads: {sum(c['initial_and_best_cuda_reload_exact'] for c in checks)}/12.","",
        "Evidence: `protocol.json`, `selection.json`, `completed.json`, training files and independent CSVs/verification JSON in this experiment. Runtime is descriptive; no controlled speed claim."]
    primary=contrasts.loc[contrasts.metric=="rmse"].set_index("contrast")
    zero_head_seeds=diagnostics.loc[(diagnostics["condition"]=="relu_original") &
        (diagnostics["stage"]=="best") & (diagnostics["exact_zero_fraction"]==1),"seed"].nunique()
    interpretation=(f"The improvement cannot be attributed to Softplus alone. Its paired RMSE change is "
        f"{primary.loc['activation_original','change_mean']:+.3f} with original initialization, compared with "
        f"{primary.loc['activation_near_one','change_mean']:+.3f} after near-one calibration. Calibration alone "
        f"changes ReLU RMSE by {primary.loc['initialization_relu','change_mean']:+.3f}; changing both yields "
        f"{summary.loc[summary.condition=='softplus_near_one','rmse_mean'].iloc[0]-summary.loc[summary.condition=='relu_original','rmse_mean'].iloc[0]:+.3f}. "
        "Either intervention recovers most of the observed reduction on this split. The smaller calibrated "
        f"activation difference does not establish equivalence. {zero_head_seeds}/3 original-ReLU seeds have at least one "
        "entirely zero head on validation at their selected checkpoints; this observation supports an "
        "optimization explanation but does not establish the complete training mechanism.")
    insertion=next(i for i,line in enumerate(narrative) if line.startswith("The calibration factor"))
    narrative[insertion:insertion]=[interpretation,""]
    (analysis/"RESULTS.md").write_text("\n".join(narrative)+"\n",encoding="utf-8")
    print(f"ANALYSIS COMPLETE: {rank.iloc[0].condition}; 12 audited runs",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,default=Path("runs/activation_initialization_20261008"))
    parser.add_argument("--verify-checkpoints",action="store_true")
    args=parser.parse_args()
    analyze(args.experiment.resolve(),args.verify_checkpoints)


if __name__=="__main__":
    main()
