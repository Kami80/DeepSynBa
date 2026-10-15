# Attribution and reference dependency

This repository publishes Kami80's small-data reproduction workflow, controlled
model variants, experiment protocols, analyses, result tables, figures, and report.
The original DeepSynBa model and processed NCI-ALMANAC archive are credited to
the original authors and are fetched separately for local experiments.

## Original work

- Authors' implementation: [hikuru/DeepSynBa](https://github.com/hikuru/DeepSynBa).
- Pinned reference commit: `43c16961da835c8545842d564a8efdfec58bdd2f`.
- Paper: Kuru, Zhang, Rattray, Ek, Cicek, Taştan, and Milo, *DeepSynBa*,
  *Bioinformatics* (2026), [doi:10.1093/bioinformatics/btag416](https://doi.org/10.1093/bioinformatics/btag416).
- The original repository states
  [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  terms. Those terms apply to the original material and its adaptations.

## How my experiments use the reference

`setup_reference.py` downloads the source at the pinned commit and verifies the
SHA-256 and size of each required model/configuration file and `data.zip`.
It installs only the reference files needed by the research scripts. They are
ignored by Git and are not part of the current published research source tree.

The setup applies a documented batch-dimension correction to the reference model
and predictor heads so single-example evaluation keeps its batch axis. This
correction preserves the model's parameter count and dose-response equation.
The custom variants in `ablation_model.py` and `activation_initialization.py`
then define the research interventions described in the protocols.

The published results compare these local implementations under the same
small-data protocol. They do not claim to reproduce the original paper's
full-data benchmark scores or establish superiority on all of its evaluation
scenarios. Dataset provenance, source fingerprints, and limitations are included
in the result records.
