# E-SAME-Surv

Core training code for **E-SAME-Surv: Evidential Spatial Entropy-Guided Adaptive Mixture-of-Experts for Pan-Cancer Survival Prediction in Whole Slide Images**.

E-SAME-Surv combines a local morphology expert (AB-MIL) and a spatial topology expert (WiKG). Differentiable spatial entropy and global semantic context guide expert routing, followed by reliability-weighted evidential fusion over survival intervals.

## Repository structure

```text
modules/        Model architectures, including ADAPTIVE_MOE_MIL and MIL baselines
process/        Training pipelines; SURVIVAL_MIL handles survival tasks
split_scripts/  Patient-based split generation and pan-cancer split merging
utils/          Data loading, survival objectives, metrics, and configuration tools
train_mil.py    YAML-driven training entry point
```

This release contains the core source code. Feature extraction, datasets, pretrained weights, evaluation/visualization scripts, and the original `configs/` directory are not included. An example configuration is provided below.

## Installation

Use a Python environment with PyTorch appropriate for your CUDA installation. Install the core dependencies:

```bash
pip install torch torchvision
pip install numpy pandas scipy scikit-learn h5py pyyaml ruamel.yaml addict tqdm pytz
```

Other bundled MIL baselines may require additional model-specific dependencies, such as `torch-geometric`, `einops`, `timm`, or `mamba-ssm`.

## Prepare data

For E-SAME-Surv, each WSI must be an HDF5 (`.h5`) file containing aligned arrays:

- `features`: patch embeddings of shape `[N, D]`.
- `coords`: patch coordinates of shape `[N, 2]`, in the same patch order.

Set `Model.in_dim` to the embedding dimension `D`. Supply a clinical CSV with one row per slide:

```csv
slide_path,case_id,event_time,censorship
/data/features/slide_001.h5,patient_001,36.5,0
/data/features/slide_002.h5,patient_002,52.0,1
```

Here `censorship=0` denotes an observed event and `censorship=1` denotes right censoring. Use a consistent survival-time unit. Generate patient-separated five-fold train/validation/test splits for each cohort:

```bash
python split_scripts/split_survival_datasets_k_fold_train_val_test.py \
  --csv_path /data/BLCA.csv \
  --save_dir data/splits/BLCA_survival_splits \
  --dataset_name TCGA_BLCA --k 5 --n_bins 4 \
  --time_col event_time --censorship_col censorship --case_id_col case_id
```

For pan-cancer training, repeat this for the other cohorts, then merge corresponding folds:

```bash
python split_scripts/merge_pan_cancer_survival_splits.py \
  --dataset_root data/splits \
  --output_dir data/splits/PAN_CANCER_survival_splits
```

The merge script adds cohort labels. The cancer-balanced sampler requires these labels for every sample. The training loader estimates survival-bin boundaries from training data and reuses them for validation and test data.

## Train

Save the example below as `esame_survival.yaml`, then update paths and feature dimensions. It is a compact starting configuration based on the local training configuration.

<details>
<summary>Example YAML configuration</summary>

```yaml
General:
  MODEL_NAME: ADAPTIVE_MOE_MIL
  task: survival
  seed: 2024
  num_classes: 4
  num_epochs: 30
  device: 0
  num_workers: 4
  best_model_metric: cancer_macro_c_index
  earlystop: {use: false, patience: 5, metric: cancer_macro_c_index}
Dataset:
  DATASET_NAME: TCGA_Pan_Survival_Splits
  dataset_root_dir: ./data/splits/PAN_CANCER_survival_splits
  dataset_csv_path: null
  balanced_sampler: {use: false}
  cancer_balanced_sampler: {use: true, replacement: true, num_samples: null}
  survival:
    n_bins: 4
    patient_level: false
    time_col: event_time
    censorship_col: censorship
    case_id_col: case_id
    slide_path_col: slide_path
    cancer_col: cancer_type
    feature_mode: adaptive_moe
Logs:
  log_root_dir: ./logs
Model:
  in_dim: 768
  ab_L: 512
  ab_D: 128
  ab_act: relu
  wikg_dim_hidden: 512
  wikg_topk: 6
  wikg_chunk_size: 1024
  wikg_agg_type: bi-interaction
  wikg_pool: attn
  wikg_act: LeakyReLU
  fusion_dim: 512
  phenotype_dim: 8
  context_dim: 128
  router_hidden_dim: 128
  router_use_mean_token: true
  router_use_dse_entropy: true
  dse_neighbors: 8
  dse_sigma: 0.0
  gate_temperature: 1.0
  gate_warmup_epochs: 5
  lambda_orth: 0.01
  dropout: 0.1
  evidential_fusion: {use: true, use_as_prediction: true}
  criterion: nll_surv
  survival_nll_weight: 0.0
  evidential_survival:
    use: true
    survival_weight: 1.0
    kl_weight: 0.01
    kl_anneal_epochs: 10
    eps: 1.0e-7
    wikg_evidence_reg_weight: 0.005
    wikg_evidence_reg_max_ratio: 1.5
    wikg_evidence_reg_target: 0.0
    wikg_evidence_reg_use_discounted: true
    wikg_evidence_reg_warmup_epochs: 5
  gate_load_balance: {use: true, weight: 0.1, queue_size: 256, min_history: 32}
  optimizer:
    which: adam
    adam_config: {lr: 0.0002, weight_decay: 1.0e-5}
  scheduler:
    warmup: 2
    which: step
    step_config: {step_size: 3, gamma: 0.9}
  alpha_surv: 0.15
  grad_accumulation: 1
  survival_ranked_contrastive:
    use: true
    weight: 0.03
    within_cancer: true
    per_cancer_memory: true
    memory_size: 128
    unknown_cancer: UNKNOWN
    time_scale: auto
    time_scale_quantile: 0.75
    auto_time_scale_event_only: true
    event_weight: 2.0
    censored_weight: 0.25
    ambiguous_weight: 0.0
    censored_pair_weight: 0.0
    max_target_distance: 0.8
    hinge_margin: 0.0
    smooth_l1_beta: 0.05
    min_pairs: 1
    warmup_epochs: 1
    ramp_epochs: 3
```

</details>

```bash
python train_mil.py --yaml_path esame_survival.yaml
```

The entry point trains each fold CSV in `Dataset.dataset_root_dir`; keep only fold CSVs in that directory. Training writes configurations, checkpoints, survival bins, metrics, and routing summaries under `Logs.log_root_dir`.

Override multiple settings with **one** `--options` argument followed by `key=value` pairs. For cohort-specific training:

```bash
python train_mil.py --yaml_path esame_survival.yaml --options \
  Dataset.DATASET_NAME=TCGA_BLCA_Survival_Splits \
  Dataset.dataset_root_dir=./data/splits/BLCA_survival_splits \
  Dataset.cancer_balanced_sampler.use=false \
  General.best_model_metric=c_index \
  General.device=0
```

`Dataset.survival.patient_level: false` treats each WSI as an independent training/evaluation sample. Setting it to `true` groups a patient's slides into one feature bag. Patient-separated folds prevent slides from one patient crossing partitions; metric aggregation follows the dataset mode.

## Acknowledgements

The training framework is based on [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE), with local and spatial experts based on AB-MIL and WiKG. The original framework license notice is preserved in [LICENSE](LICENSE).
