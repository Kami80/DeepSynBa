"""Meaningful isolation, gradient and paired-factorial checks for the new experiment."""
from copy import deepcopy
import unittest

from train_small import seed_everything
import numpy as np
import pandas as pd
import torch
import yaml
from ablation_model import POSITIVE_KEYS, build_model, variant_catalog
from activation_initialization import CONDITIONS, build_factorial_model
from analyze_activation_initialization import factorial_contrasts, summarize
from prepare_small_data import ROOT


class ActivationInitializationTests(unittest.TestCase):
    def base(self):
        config=yaml.safe_load((ROOT/"config.yaml").read_text())
        config["model_params"].update(gene_dim=2,drug_dim=3,gene_channel=8,emb_size=32)
        return config

    def model(self,condition):
        seed_everything(42,1)
        return build_factorial_model(self.base(),condition)[0]

    def test_softplus_near_one_exactly_matches_previous_package_and_rng(self):
        seed_everything(42,1)
        previous,_=build_model(self.base(),next(v for v in variant_catalog() if v["id"]=="positive_swap"))
        previous_rng=torch.get_rng_state().clone()
        current=self.model(CONDITIONS[3])
        self.assertTrue(torch.equal(previous_rng,torch.get_rng_state()))
        for key,value in previous.state_dict().items():
            self.assertTrue(torch.equal(value,current.state_dict()[key]),key)

    def test_activation_changes_no_seeded_parameters_or_rng_draws(self):
        first=self.model(CONDITIONS[0]);state=torch.get_rng_state().clone()
        second=self.model(CONDITIONS[1])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for key,value in first.state_dict().items():
            self.assertTrue(torch.equal(value,second.state_dict()[key]),key)

    def test_calibration_changes_only_three_final_layers_and_targets_one(self):
        original=self.model(CONDITIONS[0])
        for condition in CONDITIONS[2:]:
            calibrated=self.model(condition)
            changed={key for key,value in original.state_dict().items() if not torch.equal(value,calibrated.state_dict()[key])}
            allowed=set()
            for head in POSITIVE_KEYS:
                layers=calibrated.predictors[head].prediction_head
                original_layers=original.predictors[head].prediction_head
                self.assertTrue(torch.equal(layers[-2].weight,original_layers[-2].weight*.01))
                self.assertAlmostEqual(float(layers[-1](layers[-2].bias).item()),1.,places=6)
                prefix=f"predictors.{head}.prediction_head.{len(layers)-2}."
                allowed|={prefix+"weight",prefix+"bias"}
            self.assertEqual(changed,allowed)

    def test_softplus_has_gradient_where_relu_is_zero(self):
        for condition in CONDITIONS[:2]:
            activation=self.model(condition).predictors["h1_mean"].prediction_head[-1]
            value=torch.tensor([-2.],requires_grad=True)
            result=activation(value)
            result.sum().backward()
            if condition["activation"]=="relu":
                self.assertEqual(float(result),0.)
                self.assertEqual(float(value.grad),0.)
            else:
                self.assertGreater(float(result),0.)
                self.assertGreater(float(value.grad),0.)

    def table(self):
        rows=[]
        for condition,base in zip(CONDITIONS,[10.,8.,7.,6.]):
            for seed,offset in zip([42,43,44],[0.,1.,2.]):
                value=base+offset
                rows.append({"condition":condition["id"],"seed":seed,"parameters":17,"rmse":value,
                    "combination_rmse":value*2,"pearson":.4,"spearman":.3,"updates":100,
                    "best_update":80,"training_seconds":2.,"peak_cuda_memory_mib":3.,"stop_reason":"early_stopping"})
        return pd.DataFrame(rows)

    def test_conditional_effects_and_interaction_use_matching_seeds(self):
        table=self.table().sample(frac=1,random_state=5)
        paired,summary=factorial_contrasts(table)
        expected={"activation_original":-2.,"activation_near_one":-1.,"initialization_relu":-3.,
            "initialization_softplus":-2.,"interaction":1.,"average_activation":-1.5,"average_initialization":-2.5}
        for name,value in expected.items():
            row=summary.loc[(summary.metric=="rmse")&(summary.contrast==name)].iloc[0]
            self.assertEqual(row.change_mean,value)
            self.assertEqual(row.change_sd,0.)
            np.testing.assert_array_equal(paired.loc[(paired.metric=="rmse")&(paired.contrast==name),"change"],np.full(3,value))
        self.assertEqual(summarize(table).loc[0,"rmse_sd"],1.)

    def test_missing_duplicate_seeds_and_changed_architecture_are_rejected(self):
        table=self.table()
        malformed=[table.iloc[:-1],pd.concat([table.iloc[:-1],table.iloc[[0]]])]
        changed=table.copy();changed.loc[0,"parameters"]=18;malformed.append(changed)
        for value in malformed:
            with self.assertRaises(ValueError):
                factorial_contrasts(value)

    def test_invalid_factor_definition_is_rejected(self):
        invalid=deepcopy(CONDITIONS[0]);invalid["activation"]="linear"
        with self.assertRaises(ValueError):
            build_factorial_model(self.base(),invalid)


if __name__=="__main__":
    unittest.main()
