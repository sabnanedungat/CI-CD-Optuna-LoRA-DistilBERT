# CI/CD Build-Failure Prediction: Optuna–LoRA–DistilBERT and Evaluation Validity

Research code and selected evaluation outputs associated with the manuscript **“Evaluating Optuna–LoRA–DistilBERT for CI/CD Build-Failure Prediction: Historical Data, Chronological Testing, Cross-Project Evaluation, and Kafka Streaming.”**

## Study overview

This study examines two complementary experimental settings based on TravisTorrent records from 20 open-source repositories. Their results must **not** be interpreted as a controlled head-to-head comparison, because the inputs, sampling units, model initialization and evaluation procedures differ.

- **Recorded-CI-feature benchmark:** 6,000 sampled CSV records, with classical classifiers and DistilBERT, fixed LoRA, and Optuna–LoRA. The reported Optuna–LoRA test results were F1 **0.6628**, ROC-AUC **0.8303**, and average precision (AP) **0.7506**. Some input features were known only during or after execution, and job rows were split randomly; these scores do **not** establish reliable build-start prediction.
- **Distinct-build, restricted-metadata evaluation:** 5,710 unique project/build pairs, using `branch`, `is_pr`, and `num_commits`. A within-project chronological test covered 1,144 builds. Fixed LoRA achieved F1 **0.4216**; full DistilBERT achieved AP **0.3425**. The always-predict-failure F1 reference was **0.4339**. Historical capture times of the three candidate features could not be independently verified.
- **Cross-project evaluation:** Logistic regression was evaluated across five project-disjoint test folds. It predicted failure for **99.44% of builds**; this is the rate at which it issued failure predictions, **not** an accuracy score. The corresponding false-alert count was 3,907.
- **Uncertainty analysis:** A project-clustered bootstrap using 3,000 resamples (seed 42) yielded a Fixed-LoRA-minus-full-DistilBERT F1 difference of **0.0198** (95% percentile interval **[-0.0178, 0.0673]**). The interval does not establish a reliable F1 advantage.
- **Kafka prototype:** A replay of constructed metadata messages achieved mean end-to-end latency **55.361 ms** under the paced workload. The prototype did **not** consume timestamped live CI execution logs or demonstrate a production early-warning system.

## Repository layout

```text
scripts/                Distinct-build preparation, validation, held-out tests,
                        project-disjoint evaluation and cluster bootstrap
original_experiments/   Recorded-CI-feature benchmark, model training,
                        ablations, statistical tests and project transfer
kafka/                  Producer and consumer scripts for message replay
results/
  transformers/         Frozen chronological transformer predictions/metrics
  logistic_regression/  Frozen chronological logistic-regression predictions/metrics
  bootstrap/            Project-clustered bootstrap outputs
```

## Data and reproducibility

The source is the **TravisTorrent** dataset. The full source export is not included here. Scripts may require separately obtained source records, trained checkpoints, configured local file paths, Python dependencies and Kafka services. Consult each script before execution; the presence of a script alone does **not** mean the experiment can be reproduced end to end from this repository as uploaded.

Before publishing this repository, the authors should:

1. Review all scripts and outputs for secrets, private paths, personal information and file licensing restrictions.
2. Provide the specific data source/version and any permissible build or split manifests needed for reconstruction.
3. Add a verified dependency/environment file and instructions for any separately hosted checkpoints.
4. Verify that all reported outputs and filenames match the final manuscript.

**Data policy:** Any redistribution of TravisTorrent-derived records must respect the applicable source terms. No unverified claim of unrestricted redistribution is made here.

## Interpretation

Trainable-parameter savings from LoRA, classification scores on recorded CI records and Kafka processing latency are separate findings. None alone establishes reliable operational build-failure warnings. See the manuscript for complete methods, results and limitations.

## Citation

If this repository supports a published article, add the final bibliographic citation and DOI here when available.
