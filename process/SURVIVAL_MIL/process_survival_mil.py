import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.general_utils import set_global_seed
from utils.model_utils import get_model_from_yaml, get_optimizer, get_param_optimizer, get_scheduler, save_best_model, save_last_model
from utils.process_utils import get_process_pipeline
from utils.survival_utils import (
    EvidentialSurvivalLoss,
    GateLoadBalanceQueue,
    SurvivalMILSlideDataset,
    SurvivalRankedContrastiveLoss,
    SurvivalRankedMemoryBank,
    get_survival_loss,
    metadata_cancer_types,
    normalize_cancer_type,
    survival_from_logits,
    survival_eval_loop,
    survival_metrics,
    survival_metrics_by_cancer,
    survival_probability_metrics_by_cancer,
    survival_train_loop,
    unpack_survival_batch,
)


def _cfg(config, *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _survival_time_grid_from_loader(loader):
    dataset = getattr(loader, "dataset", None)
    bins = getattr(dataset, "bins", None)
    if bins is None:
        return None
    bins = np.asarray(bins, dtype=float).reshape(-1)
    return bins[1:] if len(bins) > 1 else None


def _device_from_args(args):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.General.device}")
    return torch.device("cpu")


def _feature_mode_for_model(args):
    model_name = args.General.MODEL_NAME
    if model_name == "ADAPTIVE_MOE_MIL":
        return "adaptive_moe"
    if model_name == "CDP_MIL":
        return "cdp"
    if model_name == "LONG_MIL":
        return "long"
    if model_name in {"PSA_MIL", "STABLE_MIL", "DAG_MIL"}:
        return "coord"
    if model_name == "SC_MIL":
        return "sc"
    return "plain"


def _survival_dataset_kwargs(args, bins=None):
    survival_cfg = _cfg(args, "Dataset", "survival", default={}) or {}
    n_bins = int(survival_cfg.get("n_bins", _cfg(args, "General", "num_classes", default=4)))
    return {
        "n_bins": n_bins,
        "survival_bins": bins,
        "patient_level": bool(survival_cfg.get("patient_level", True)),
        "feature_mode": survival_cfg.get("feature_mode", _feature_mode_for_model(args)),
        "bayesian_dir": _cfg(args, "Dataset", "BeyesGuassian_pt_dir", default=None),
        "time_col": survival_cfg.get("time_col", None),
        "censorship_col": survival_cfg.get("censorship_col", None),
        "event_col": survival_cfg.get("event_col", None),
        "case_id_col": survival_cfg.get("case_id_col", None),
        "slide_path_col": survival_cfg.get("slide_path_col", None),
        "cancer_col": survival_cfg.get("cancer_col", None),
    }


def build_survival_datasets(args):
    dataset_csv_path = args.Dataset.dataset_csv_path
    train_dataset = SurvivalMILSlideDataset(dataset_csv_path, "train", **_survival_dataset_kwargs(args))
    val_dataset = SurvivalMILSlideDataset(dataset_csv_path, "val", **_survival_dataset_kwargs(args, bins=train_dataset.bins))
    test_dataset = SurvivalMILSlideDataset(dataset_csv_path, "test", **_survival_dataset_kwargs(args, bins=train_dataset.bins))
    args.General.num_classes = train_dataset.num_bins
    return train_dataset, val_dataset, test_dataset


def _build_train_loader(args, train_dataset, generator):
    num_workers = args.General.num_workers
    train_dataset.return_meta = True
    cancer_sampler_cfg = _cfg(args, "Dataset", "cancer_balanced_sampler", default={}) or {}
    balanced_sampler_cfg = _cfg(args, "Dataset", "balanced_sampler", default={}) or {}
    if cancer_sampler_cfg.get("use", False) and balanced_sampler_cfg.get("use", False):
        raise ValueError("Enable either cancer_balanced_sampler or balanced_sampler, not both.")
    if cancer_sampler_cfg.get("use", False):
        sampler = train_dataset.get_cancer_balanced_sampler(
            replacement=cancer_sampler_cfg.get("replacement", True),
            num_samples=cancer_sampler_cfg.get("num_samples", None),
        )
        print("Cancer-balanced sampler enabled. Training cancer counts:", dict(sorted(Counter(train_dataset.cancer_types).items())))
        return DataLoader(train_dataset, batch_size=1, num_workers=num_workers, generator=generator, sampler=sampler)
    if balanced_sampler_cfg.get("use", False):
        sampler = train_dataset.get_balanced_sampler(replacement=balanced_sampler_cfg.get("replacement", True))
        return DataLoader(train_dataset, batch_size=1, num_workers=num_workers, generator=generator, sampler=sampler)
    return DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, generator=generator)


def _init_log():
    log = {
        "epoch": [],
        "train_loss": [],
        "train_surv_loss": [],
        "train_surv_loss_weight": [],
        "train_contrast_loss": [],
        "train_contrast_weight": [],
        "train_contrast_valid_pairs": [],
        "train_contrast_event_event_pairs": [],
        "train_contrast_event_censored_pairs": [],
        "train_contrast_ambiguous_pairs": [],
        "train_contrast_censored_censored_pairs": [],
        "train_contrast_within_cancer_pairs": [],
        "train_contrast_skipped_empty_cancer_bank": [],
        "train_c_index": [],
        "train_ibs": [],
        "train_inbll": [],
        "val_loss": [],
        "val_surv_loss": [],
        "val_surv_loss_weight": [],
        "val_c_index": [],
        "val_ibs": [],
        "val_inbll": [],
        "test_loss": [],
        "test_surv_loss": [],
        "test_surv_loss_weight": [],
        "test_c_index": [],
        "test_ibs": [],
        "test_inbll": [],
    }
    for split in ("train", "val", "test"):
        for metric in (
            "aux_loss",
            "orth_loss",
            "expert_cosine",
            "alignment_norm_gap",
            "gate_entropy",
            "gate_local",
            "gate_wikg",
            "gate_balance_loss",
            "gate_balance_weight",
            "gate_load_imbalance",
            "dse_entropy",
            "t_wsi_norm",
            "t_mean_norm",
            "t_mean_token_norm",
            "t_pheno_entropy",
            "fusion_feature_norm",
            "uses_physical_coords",
            "evidential_loss",
            "edl_surv_loss",
            "edl_local_surv_loss",
            "edl_wikg_surv_loss",
            "edl_kl_loss",
            "edl_expert_kl_loss",
            "edl_surv_weight",
            "edl_local_surv_weight",
            "edl_wikg_surv_weight",
            "edl_kl_weight",
            "edl_expert_kl_weight",
            "edl_kl_anneal",
            "edl_wikg_evidence_reg_loss",
            "edl_wikg_evidence_reg_weight",
            "edl_wikg_evidence_reg_anneal",
            "edl_wikg_evidence_reg_ratio_excess",
            "edl_wikg_evidence_mass",
            "edl_local_evidence_mass",
            "local_uncertainty",
            "wikg_uncertainty",
            "discounted_local_uncertainty",
            "discounted_wikg_uncertainty",
            "fused_uncertainty",
            "local_evidence_strength",
            "wikg_evidence_strength",
            "discounted_local_evidence_strength",
            "discounted_wikg_evidence_strength",
            "fused_evidence_strength",
            "expert_disagreement",
            "fused_probability_entropy",
            "fused_belief_entropy",
            "evidential_as_prediction",
        ):
            log[f"{split}_{metric}"] = []
    return log


def _append_log(log, epoch, train_loss, train_metrics, val_loss, val_metrics, test_loss, test_metrics):
    log["epoch"].append(epoch + 1)
    log["train_loss"].append(train_loss)
    log["train_surv_loss"].append(None if train_metrics is None else train_metrics.get("surv_loss"))
    log["train_surv_loss_weight"].append(None if train_metrics is None else train_metrics.get("surv_loss_weight"))
    log["train_contrast_loss"].append(None if train_metrics is None else train_metrics.get("contrast_loss"))
    log["train_contrast_weight"].append(None if train_metrics is None else train_metrics.get("contrast_weight"))
    log["train_contrast_valid_pairs"].append(None if train_metrics is None else train_metrics.get("contrast_valid_pairs"))
    log["train_contrast_event_event_pairs"].append(None if train_metrics is None else train_metrics.get("contrast_event_event_pairs"))
    log["train_contrast_event_censored_pairs"].append(None if train_metrics is None else train_metrics.get("contrast_event_censored_pairs"))
    log["train_contrast_ambiguous_pairs"].append(None if train_metrics is None else train_metrics.get("contrast_ambiguous_pairs"))
    log["train_contrast_censored_censored_pairs"].append(
        None if train_metrics is None else train_metrics.get("contrast_censored_censored_pairs")
    )
    log["train_contrast_within_cancer_pairs"].append(
        None if train_metrics is None else train_metrics.get("contrast_within_cancer_pairs")
    )
    log["train_contrast_skipped_empty_cancer_bank"].append(
        None if train_metrics is None else train_metrics.get("contrast_skipped_empty_cancer_bank")
    )
    log["train_c_index"].append(None if train_metrics is None else train_metrics.get("c_index"))
    log["train_ibs"].append(None if train_metrics is None else train_metrics.get("ibs"))
    log["train_inbll"].append(None if train_metrics is None else train_metrics.get("inbll"))
    log["val_loss"].append(val_loss)
    log["val_surv_loss"].append(None if val_metrics is None else val_metrics.get("surv_loss"))
    log["val_surv_loss_weight"].append(None if val_metrics is None else val_metrics.get("surv_loss_weight"))
    log["val_c_index"].append(None if val_metrics is None else val_metrics.get("c_index"))
    log["val_ibs"].append(None if val_metrics is None else val_metrics.get("ibs"))
    log["val_inbll"].append(None if val_metrics is None else val_metrics.get("inbll"))
    log["test_loss"].append(test_loss)
    log["test_surv_loss"].append(None if test_metrics is None else test_metrics.get("surv_loss"))
    log["test_surv_loss_weight"].append(None if test_metrics is None else test_metrics.get("surv_loss_weight"))
    log["test_c_index"].append(None if test_metrics is None else test_metrics.get("c_index"))
    log["test_ibs"].append(None if test_metrics is None else test_metrics.get("ibs"))
    log["test_inbll"].append(None if test_metrics is None else test_metrics.get("inbll"))
    for split, metrics in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)):
        for metric in (
            "aux_loss",
            "orth_loss",
            "expert_cosine",
            "alignment_norm_gap",
            "gate_entropy",
            "gate_local",
            "gate_wikg",
            "gate_balance_loss",
            "gate_balance_weight",
            "gate_load_imbalance",
            "dse_entropy",
            "t_wsi_norm",
            "t_mean_norm",
            "t_mean_token_norm",
            "t_pheno_entropy",
            "fusion_feature_norm",
            "uses_physical_coords",
            "evidential_loss",
            "edl_surv_loss",
            "edl_local_surv_loss",
            "edl_wikg_surv_loss",
            "edl_kl_loss",
            "edl_expert_kl_loss",
            "edl_surv_weight",
            "edl_local_surv_weight",
            "edl_wikg_surv_weight",
            "edl_kl_weight",
            "edl_expert_kl_weight",
            "edl_kl_anneal",
            "edl_wikg_evidence_reg_loss",
            "edl_wikg_evidence_reg_weight",
            "edl_wikg_evidence_reg_anneal",
            "edl_wikg_evidence_reg_ratio_excess",
            "edl_wikg_evidence_mass",
            "edl_local_evidence_mass",
            "local_uncertainty",
            "wikg_uncertainty",
            "discounted_local_uncertainty",
            "discounted_wikg_uncertainty",
            "fused_uncertainty",
            "local_evidence_strength",
            "wikg_evidence_strength",
            "discounted_local_evidence_strength",
            "discounted_wikg_evidence_strength",
            "fused_evidence_strength",
            "expert_disagreement",
            "fused_probability_entropy",
            "fused_belief_entropy",
            "evidential_as_prediction",
        ):
            log[f"{split}_{metric}"].append(None if metrics is None else metrics.get(metric))
        _append_cancer_metrics(log, split, metrics)


def _append_cancer_metrics(log, split, metrics):
    current_length = len(log["epoch"])
    metric_values = {
        f"{split}_{key}": value
        for key, value in (metrics or {}).items()
        if key.startswith("cancer_")
    }
    existing_keys = [key for key in log if key.startswith(f"{split}_cancer_")]
    for key in existing_keys:
        log[key].append(metric_values.pop(key, None))
    for key, value in metric_values.items():
        log[key] = [None] * (current_length - 1) + [value]


def _save_survival_log(args, log, best_epoch, process_pipeline):
    os.makedirs(args.Logs.now_log_dir, exist_ok=True)
    log_df = pd.DataFrame(log)
    log_path = os.path.join(
        args.Logs.now_log_dir,
        f"Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv",
    )
    log_df.to_csv(log_path, index=False)
    print("Global survival log CSV saved!")

    if process_pipeline != "Train_Test" and best_epoch is not None:
        best_df = log_df[log_df["epoch"] == best_epoch]
        best_path = os.path.join(
            args.Logs.now_log_dir,
            f"Best_Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv",
        )
        best_df.to_csv(best_path, index=False)
        print("Best survival log CSV saved!")


def _survival_bin_distribution_dataframe(split_datasets):
    rows = []
    for split, dataset in split_datasets:
        bins = np.asarray(dataset.bins, dtype=float)
        num_bins = int(dataset.num_bins)
        samples = list(dataset.samples)
        split_total = len(samples)
        cancer_types = sorted({normalize_cancer_type(sample.get("cancer_type")) for sample in samples})
        for cancer_type in ["ALL"] + cancer_types:
            if cancer_type == "ALL":
                cancer_samples = samples
            else:
                cancer_samples = [
                    sample for sample in samples
                    if normalize_cancer_type(sample.get("cancer_type")) == cancer_type
                ]
            cancer_total = len(cancer_samples)
            for bin_idx in range(num_bins):
                bin_samples = [sample for sample in cancer_samples if int(sample["label"]) == bin_idx]
                n = len(bin_samples)
                events = sum(1 for sample in bin_samples if float(sample["censorship"]) < 0.5)
                censored = n - events
                rows.append(
                    {
                        "split": split,
                        "cancer_type": cancer_type,
                        "split_total": split_total,
                        "cancer_total": cancer_total,
                        "bin": bin_idx,
                        "bin_left": float(bins[bin_idx]),
                        "bin_right": float(bins[bin_idx + 1]),
                        "n": n,
                        "events": events,
                        "censored": censored,
                        "event_rate": (events / n) if n > 0 else np.nan,
                        "ratio_within_cancer": (n / cancer_total) if cancer_total > 0 else np.nan,
                        "ratio_within_split": (n / split_total) if split_total > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _save_survival_bin_distribution(args, train_dataset, val_dataset, test_dataset):
    os.makedirs(args.Logs.now_log_dir, exist_ok=True)
    distribution_df = _survival_bin_distribution_dataframe(
        (
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        )
    )
    path = os.path.join(args.Logs.now_log_dir, "Survival_Bin_Distribution_By_Cancer.csv")
    distribution_df.to_csv(path, index=False)
    print(f"Survival bin distribution by cancer saved: {path}")
    return path


def _save_cancer_metrics(test_log_dir, model_name, metrics):
    cancer_rows = []
    for key, c_index in sorted(metrics.items()):
        if not key.startswith("cancer_") or not key.endswith("_c_index") or key == "cancer_macro_c_index":
            continue
        cancer_type = key[len("cancer_") : -len("_c_index")]
        cancer_rows.append(
            {
                "cancer_type": cancer_type,
                "c_index": c_index,
                "ibs": metrics.get(f"cancer_{cancer_type}_ibs"),
                "inbll": metrics.get(f"cancer_{cancer_type}_inbll"),
                "n": metrics.get(f"cancer_{cancer_type}_n"),
                "events": metrics.get(f"cancer_{cancer_type}_events"),
            }
        )
    if not cancer_rows:
        return None
    cancer_rows.append(
        {
            "cancer_type": "MACRO",
            "c_index": metrics.get("cancer_macro_c_index"),
            "ibs": metrics.get("cancer_macro_ibs"),
            "inbll": metrics.get("cancer_macro_inbll"),
            "n": sum(row["n"] for row in cancer_rows if row["n"] is not None),
            "events": sum(row["events"] for row in cancer_rows if row["events"] is not None),
        }
    )
    path = os.path.join(test_log_dir, f"Test_Cancer_Metrics_{model_name}_survival.csv")
    pd.DataFrame(cancer_rows).to_csv(path, index=False)
    return path


def _save_gate_distribution(test_log_dir, model_name, metrics):
    suffixes = (
        "n",
        "mean",
        "std",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "high_ab_ratio",
        "high_wikg_ratio",
        "soft_mixed_ratio",
    )
    rows = {}
    marker = "_gate_local_"
    for key, value in metrics.items():
        if not key.startswith("cancer_") or marker not in key:
            continue
        cancer_type, statistic = key[len("cancer_") :].split(marker, 1)
        if statistic in suffixes:
            rows.setdefault(cancer_type, {"cancer_type": cancer_type})[statistic] = value
    if not rows:
        return None
    threshold = metrics.get("cancer_gate_high_weight_threshold")
    for row in rows.values():
        row["high_weight_threshold"] = threshold
    ordered_rows = sorted(rows.values(), key=lambda row: (row["cancer_type"] != "ALL", row["cancer_type"]))
    path = os.path.join(test_log_dir, f"Test_Gate_Distribution_{model_name}_survival.csv")
    pd.DataFrame(ordered_rows, columns=("cancer_type", "high_weight_threshold") + suffixes).to_csv(path, index=False)
    return path


def _save_sample_routing(test_log_dir, model_name, results):
    rows = []
    for sample in results.values():
        routing = sample.get("routing")
        if routing is None:
            continue
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "cancer_type": sample.get("cancer_type"),
                "risk": sample.get("risk"),
                "event_time": sample.get("event_time"),
                "censorship": sample.get("censorship"),
                "gate_local": routing.get("local"),
                "gate_wikg": routing.get("wikg"),
                "dse_entropy": routing.get("dse_entropy"),
            }
        )
    if not rows:
        return None
    path = os.path.join(test_log_dir, f"Test_Sample_Routing_{model_name}_survival.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _metric_higher_is_better(metric_name):
    metric_name = str(metric_name).replace("val_", "").replace("test_", "")
    return metric_name not in {"loss", "ibs", "inbll", "cancer_macro_ibs", "cancer_macro_inbll"}


def _metric_value(metrics, loss, metric_name):
    metric_name = str(metric_name)
    if metric_name in {"loss", "val_loss"}:
        value = -loss if loss is not None else None
        return None if value is not None and pd.isna(value) else value
    if metrics is None:
        return None
    metric_name = metric_name.replace("val_", "").replace("test_", "")
    value = metrics.get(metric_name)
    if value is not None and not _metric_higher_is_better(metric_name):
        value = -value
    return None if value is not None and pd.isna(value) else value


def _should_stop(log, patience, metric_name):
    if patience is None or patience <= 0 or len(log["epoch"]) <= patience:
        return False
    metric_name = str(metric_name)
    key = "val_loss" if metric_name in {"loss", "val_loss"} else f"val_{metric_name.replace('val_', '')}"
    if key not in log:
        return False
    values = [value for value in log[key] if value is not None and not pd.isna(value)]
    if len(values) <= patience:
        return False
    recent = values[-patience:]
    higher_is_better = _metric_higher_is_better(metric_name)
    best_before_recent = max(values[:-patience]) if higher_is_better else min(values[:-patience])
    if not higher_is_better:
        return min(recent) >= best_before_recent
    return max(recent) <= best_before_recent


def _as_bool(value, default=False):
    if value in [None, "", "null"]:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_float(value, default):
    if value in [None, "", "null"]:
        return default
    return float(value)


def _estimate_contrastive_time_scale(train_dataset, quantile=0.75, event_only=True):
    if train_dataset is None or len(train_dataset.samples) < 2:
        return 60.0
    times = np.asarray([sample["event_time"] for sample in train_dataset.samples], dtype=float)
    censorships = np.asarray([sample["censorship"] for sample in train_dataset.samples], dtype=float)
    valid = np.isfinite(times) & np.isfinite(censorships)
    times = times[valid]
    censorships = censorships[valid]
    if event_only and np.sum(censorships < 0.5) >= 2:
        times = times[censorships < 0.5]
    if len(times) < 2:
        return 60.0

    deltas = np.abs(times[:, None] - times[None, :])
    deltas = deltas[np.triu_indices_from(deltas, k=1)]
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 60.0
    quantile = min(max(float(quantile), 0.05), 0.95)
    return max(float(np.quantile(deltas, quantile)), 1.0)


def _contrastive_epoch_weight(base_weight, schedule, epoch):
    if base_weight <= 0:
        return 0.0
    warmup_epochs = int(schedule.get("warmup_epochs", 0))
    ramp_epochs = int(schedule.get("ramp_epochs", 0))
    current_epoch = epoch + 1
    if current_epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_weight)
    progress = min(1.0, max(0.0, (current_epoch - warmup_epochs) / ramp_epochs))
    return float(base_weight) * progress


def _build_contrastive(args, train_dataset=None):
    cfg = _cfg(args, "Model", "survival_ranked_contrastive", default={}) or {}
    if not _as_bool(cfg.get("use", False)):
        return None, None, 0.0, {}
    within_cancer = _as_bool(cfg.get("within_cancer", False), False)
    per_cancer_memory = _as_bool(cfg.get("per_cancer_memory", within_cancer), within_cancer)
    unknown_cancer = str(cfg.get("unknown_cancer", "UNKNOWN"))

    time_scale_cfg = cfg.get("time_scale", 60.0)
    if isinstance(time_scale_cfg, str) and time_scale_cfg.strip().lower() == "auto":
        time_scale = _estimate_contrastive_time_scale(
            train_dataset,
            quantile=_as_float(cfg.get("time_scale_quantile", 0.75), 0.75),
            event_only=_as_bool(cfg.get("auto_time_scale_event_only", True), True),
        )
    else:
        time_scale = _as_float(time_scale_cfg, 60.0)

    loss_fn = SurvivalRankedContrastiveLoss(
        time_scale=time_scale,
        event_weight=_as_float(cfg.get("event_weight", 2.0), 2.0),
        censored_weight=_as_float(cfg.get("censored_weight", 0.25), 0.25),
        ambiguous_weight=_as_float(cfg.get("ambiguous_weight", 0.0), 0.0),
        censored_pair_weight=_as_float(cfg.get("censored_pair_weight", 0.0), 0.0),
        max_target_distance=_as_float(cfg.get("max_target_distance", 0.8), 0.8),
        hinge_margin=_as_float(cfg.get("hinge_margin", 0.0), 0.0),
        smooth_l1_beta=_as_float(cfg.get("smooth_l1_beta", 0.05), 0.05),
        min_pairs=int(cfg.get("min_pairs", 1)),
        within_cancer=within_cancer,
        unknown_cancer=unknown_cancer,
    )

    # loss_fn = SurvivalRankedContrastiveLoss(
    #     time_scale=time_scale,
    #     margin=0.15,               # 旧代码的 hinge_margin 改名为 margin
    #     event_weight=2.0,          # 保持不变
    #     censored_weight=0.5,       # 保持不变
    #     time_weight_scale=120.0,   # 旧代码的 time_scale 改名为 time_weight_scale
    #     min_pairs=1                # 保持不变
    #     )

    bank = SurvivalRankedMemoryBank(
        max_size=cfg.get("memory_size", 256),
        per_cancer=per_cancer_memory,
        unknown_cancer=unknown_cancer,
    )
    schedule = {
        "warmup_epochs": int(cfg.get("warmup_epochs", 1)),
        "ramp_epochs": int(cfg.get("ramp_epochs", 3)),
        "within_cancer": within_cancer,
        "per_cancer_memory": per_cancer_memory,
    }
    return loss_fn, bank, _as_float(cfg.get("weight", 0.03), 0.03), schedule


def _build_evidential_survival(args):
    cfg = _cfg(args, "Model", "evidential_survival", default={}) or {}
    if not _as_bool(cfg.get("use", False)):
        return None
    return EvidentialSurvivalLoss(
        survival_weight=_as_float(cfg.get("survival_weight", cfg.get("weight", 0.1)), 0.1),
        kl_weight=_as_float(cfg.get("kl_weight", 1e-3), 1e-3),
        kl_anneal_epochs=int(cfg.get("kl_anneal_epochs", 10)),
        eps=_as_float(cfg.get("eps", 1e-7), 1e-7),
        local_survival_weight=_as_float(
            cfg.get("local_survival_weight", cfg.get("local_expert_weight", 0.0)),
            0.0,
        ),
        wikg_survival_weight=_as_float(
            cfg.get("wikg_survival_weight", cfg.get("wikg_expert_weight", 0.0)),
            0.0,
        ),
        expert_kl_weight=_as_float(cfg.get("expert_kl_weight", 0.0), 0.0),
        wikg_evidence_reg_weight=_as_float(
            cfg.get("wikg_evidence_reg_weight", cfg.get("wikg_reg_weight", 0.0)),
            0.0,
        ),
        wikg_evidence_reg_max_ratio=_as_float(
            cfg.get("wikg_evidence_reg_max_ratio", cfg.get("wikg_reg_max_ratio", 0.0)),
            0.0,
        ),
        wikg_evidence_reg_target=_as_float(
            cfg.get("wikg_evidence_reg_target", cfg.get("wikg_reg_target", 0.0)),
            0.0,
        ),
        wikg_evidence_reg_use_discounted=_as_bool(
            cfg.get("wikg_evidence_reg_use_discounted", cfg.get("wikg_reg_use_discounted", True)),
            True,
        ),
        wikg_evidence_reg_warmup_epochs=int(
            cfg.get("wikg_evidence_reg_warmup_epochs", cfg.get("wikg_reg_warmup_epochs", 0))
        ),
    )


def _survival_nll_weight(args):
    return _as_float(_cfg(args, "Model", "survival_nll_weight", default=1.0), 1.0)


def _dtfd_state_dict(model_list):
    classifier, attention, dim_reduction, att_cls = model_list
    return {
        "classifier": classifier.state_dict(),
        "attention": attention.state_dict(),
        "dimReduction": dim_reduction.state_dict(),
        "attCls": att_cls.state_dict(),
    }


def _dtfd_set_mode(model_list, training):
    for module in model_list:
        module.train(training)


def _dtfd_pseudo_forward(model_list, bag, label, distill, total_instance):
    classifier, attention, dim_reduction, att_cls = model_list
    bag = bag.squeeze(0)
    num_group = min(max(1, getattr(_dtfd_pseudo_forward, "num_group", 1)), max(1, bag.shape[0]))
    instance_per_group = max(1, total_instance // max(num_group, 1))

    slide_sub_preds = []
    pseudo_features = []
    for sub_feat in torch.chunk(bag, num_group, dim=0):
        mid_feat = dim_reduction(sub_feat)
        attn = attention(mid_feat).squeeze(0)
        attn_feat = torch.einsum("ns,n->ns", mid_feat, attn)
        bag_feat = torch.sum(attn_feat, dim=0, keepdim=True)
        sub_pred = classifier(bag_feat)
        slide_sub_preds.append(sub_pred)

        patch_logits = torch.einsum("nf,cf->nc", attn_feat, list(classifier.parameters())[-2])
        patch_probs = torch.softmax(patch_logits, dim=1)
        k = min(instance_per_group, patch_probs.shape[0])
        _, sort_idx = torch.sort(patch_probs[:, -1], descending=True)
        top_idx = sort_idx[:k].long()
        bottom_idx = sort_idx[-k:].long()
        if distill == "MaxMinS":
            pseudo_features.append(mid_feat.index_select(dim=0, index=torch.cat([top_idx, bottom_idx], dim=0)))
        elif distill == "MaxS":
            pseudo_features.append(mid_feat.index_select(dim=0, index=top_idx))
        else:
            pseudo_features.append(bag_feat)

    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
    pseudo_features = torch.cat(pseudo_features, dim=0)
    final_output = att_cls(pseudo_features, return_WSI_feature=True)
    final_logits = final_output["logits"]
    final_feature = final_output.get("WSI_feature", final_logits)
    sub_labels = label.repeat(slide_sub_preds.shape[0])
    return slide_sub_preds, sub_labels, final_logits, final_feature


def dtfd_survival_train_loop(
    device,
    model_list,
    loader,
    loss_fn,
    optimizer_list,
    scheduler_list,
    num_group,
    grad_clipping,
    distill,
    total_instance,
    contrastive_loss_fn=None,
    contrastive_bank=None,
    contrastive_weight=0.0,
):
    import time

    start = time.time()
    _dtfd_set_mode(model_list, True)
    _dtfd_pseudo_forward.num_group = num_group
    optimizer_a, optimizer_b = optimizer_list
    train_loss = 0.0
    surv_loss_log = 0.0
    contrast_loss_log = 0.0
    contrast_stats_log = {}
    risks, event_times, censorships, cancer_types, survival_curves = [], [], [], [], []

    for batch in loader:
        bag, label, event_time, c, metadata = unpack_survival_batch(batch)
        bag = bag.to(device).float()
        label = label.to(device).long()
        c = c.to(device).float()
        batch_cancer_types = metadata_cancer_types(metadata, label.numel()) if metadata is not None else None

        sub_logits, sub_labels, final_logits, final_feature = _dtfd_pseudo_forward(model_list, bag, label, distill, total_instance)
        sub_c = c.repeat(sub_logits.shape[0])
        sub_hazards, sub_survival = survival_from_logits(sub_logits)
        loss_a = loss_fn(hazards=sub_hazards, survival=sub_survival, y=sub_labels, c=sub_c)

        optimizer_a.zero_grad()
        loss_a.backward(retain_graph=True)
        for module in model_list[:3]:
            torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clipping)

        hazards, survival = survival_from_logits(final_logits)
        surv_loss_b = loss_fn(hazards=hazards, survival=survival, y=label, c=c)
        contrast_loss = bag.new_tensor(0.0)
        if contrastive_loss_fn is not None and contrastive_bank is not None and contrastive_weight > 0:
            contrast_loss = contrastive_loss_fn(
                final_feature.reshape(final_feature.shape[0], -1),
                event_time.to(device),
                c,
                contrastive_bank,
                cancer_types=batch_cancer_types,
            )
            for key, value in getattr(contrastive_loss_fn, "last_stats", {}).items():
                contrast_stats_log[key] = contrast_stats_log.get(key, 0) + value
        loss_b = surv_loss_b + contrastive_weight * contrast_loss
        optimizer_b.zero_grad()
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(model_list[3].parameters(), grad_clipping)

        optimizer_a.step()
        optimizer_b.step()

        train_loss += (loss_a.item() + loss_b.item())
        surv_loss_log += (loss_a.item() + surv_loss_b.item())
        contrast_loss_log += contrast_loss.item()
        risks.extend((-torch.sum(survival, dim=1)).detach().cpu().numpy().tolist())
        survival_curves.extend(survival.detach().cpu().numpy().tolist())
        event_times.extend(event_time.detach().cpu().numpy().reshape(-1).tolist())
        censorships.extend(c.detach().cpu().numpy().reshape(-1).tolist())
        if batch_cancer_types is not None:
            cancer_types.extend(batch_cancer_types)
        if contrastive_bank is not None and contrastive_weight > 0:
            contrastive_bank.enqueue(
                final_feature.reshape(final_feature.shape[0], -1),
                event_time,
                c.detach().cpu(),
                cancer_types=batch_cancer_types,
            )

    for scheduler in scheduler_list:
        if scheduler is not None:
            scheduler.step()
    train_loss /= max(len(loader), 1)
    time_grid = _survival_time_grid_from_loader(loader)
    metrics = survival_metrics(
        event_times,
        risks,
        censorships,
        survival_probabilities=survival_curves,
        time_grid=time_grid,
    )
    if len(cancer_types) == len(event_times):
        metrics.update(survival_metrics_by_cancer(event_times, risks, censorships, cancer_types))
        metrics.update(
            survival_probability_metrics_by_cancer(
                event_times,
                survival_curves,
                censorships,
                cancer_types,
                time_grid=time_grid,
            )
        )
    metrics["surv_loss"] = surv_loss_log / max(len(loader), 1)
    metrics["contrast_loss"] = contrast_loss_log / max(len(loader), 1)
    metrics["contrast_weight"] = contrastive_weight
    for key, value in contrast_stats_log.items():
        metrics[f"contrast_{key}"] = value
    return train_loss, time.time() - start, metrics


def dtfd_survival_eval_loop(device, model_list, loader, loss_fn, num_group, distill, total_instance, return_results=False):
    _dtfd_set_mode(model_list, False)
    _dtfd_pseudo_forward.num_group = num_group
    loss_log = 0.0
    risks, event_times, censorships, cancer_types, survival_curves = [], [], [], [], []
    results = {}

    with torch.no_grad():
        for batch_idx, (bag, label, event_time, c) in enumerate(loader):
            bag = bag.to(device).float()
            label = label.to(device).long()
            c = c.to(device).float()
            _, _, logits, _ = _dtfd_pseudo_forward(model_list, bag, label, distill, total_instance)
            hazards, survival = survival_from_logits(logits)
            loss = loss_fn(hazards=hazards, survival=survival, y=label, c=c, alpha=0.0)
            loss_log += loss.item()
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            risks.extend(risk.tolist())
            survival_curves.extend(survival.detach().cpu().numpy().tolist())
            event_times.extend(event_time.detach().cpu().numpy().reshape(-1).tolist())
            censorships.extend(c.detach().cpu().numpy().reshape(-1).tolist())
            sample = loader.dataset.samples[batch_idx]
            cancer_types.append(sample.get("cancer_type"))

            if return_results:
                results[sample["sample_id"]] = {
                    "sample_id": sample["sample_id"],
                    "cancer_type": sample.get("cancer_type"),
                    "risk": float(risk[0]),
                    "disc_label": int(label.item()),
                    "event_time": float(event_time.item()),
                    "censorship": float(c.item()),
                    "survival_probability": survival.detach().cpu().reshape(-1, survival.shape[-1])[0].tolist(),
                }

    loss_log /= max(len(loader), 1)
    time_grid = _survival_time_grid_from_loader(loader)
    metrics = survival_metrics(
        event_times,
        risks,
        censorships,
        survival_probabilities=survival_curves,
        time_grid=time_grid,
    )
    metrics.update(survival_metrics_by_cancer(event_times, risks, censorships, cancer_types))
    if len(cancer_types) == len(event_times):
        metrics.update(
            survival_probability_metrics_by_cancer(
                event_times,
                survival_curves,
                censorships,
                cancer_types,
                time_grid=time_grid,
            )
        )
    if return_results:
        return loss_log, metrics, results
    return loss_log, metrics


def process_DTFD_SURVIVAL_MIL(args, train_dataset, val_dataset, test_dataset, process_pipeline, device, loss_fn):
    model_list = [module.to(device) for module in get_model_from_yaml(args)]
    trainable_parameters_a = []
    for module in model_list[:3]:
        trainable_parameters_a += list(module.parameters())
    optimizer_a, base_lr = get_param_optimizer(args, trainable_parameters_a)
    scheduler_a, warmup_scheduler_a = get_scheduler(args, optimizer_a, base_lr)
    optimizer_b, base_lr = get_param_optimizer(args, model_list[3].parameters())
    scheduler_b, warmup_scheduler_b = get_scheduler(args, optimizer_b, base_lr)

    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    train_loader = _build_train_loader(args, train_dataset, generator)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.General.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.General.num_workers)

    warmup_epoch = args.Model.scheduler.warmup
    num_group = int(args.Model.num_Group)
    grad_clipping = float(args.Model.grad_clipping)
    distill = args.Model.distill
    total_instance = int(args.Model.total_instance)
    best_model_metric = str(_cfg(args, "General", "best_model_metric", default="c_index"))
    best_score, best_epoch = None, None
    epoch_log = _init_log()
    earlystop_cfg = _cfg(args, "General", "earlystop", default={}) or {}
    use_earlystop = earlystop_cfg.get("use", False)
    patience = int(earlystop_cfg.get("patience", 0)) if use_earlystop else 0
    contrastive_loss_fn, contrastive_bank, contrastive_weight, contrastive_schedule = _build_contrastive(args, train_dataset)

    print("DTFD Survival Model Ready!")
    if contrastive_loss_fn is not None:
        print(
            "Survival-Ranked Contrastive MIL enabled, "
            f"base_weight={contrastive_weight}, time_scale={contrastive_loss_fn.time_scale}, "
            f"warmup={contrastive_schedule.get('warmup_epochs', 0)}, ramp={contrastive_schedule.get('ramp_epochs', 0)}, "
            f"within_cancer={contrastive_schedule.get('within_cancer', False)}, "
            f"per_cancer_memory={contrastive_schedule.get('per_cancer_memory', False)}"
        )
    print("Start DTFD Survival Process!")
    print("Using Process Pipeline:", process_pipeline)

    for epoch in tqdm(range(args.General.num_epochs), colour="GREEN"):
        scheduler_list = [
            warmup_scheduler_a if epoch + 1 <= warmup_epoch else scheduler_a,
            warmup_scheduler_b if epoch + 1 <= warmup_epoch else scheduler_b,
        ]
        now_contrastive_weight = _contrastive_epoch_weight(contrastive_weight, contrastive_schedule, epoch)
        train_loss, cost_time, train_metrics = dtfd_survival_train_loop(
            device, model_list, train_loader, loss_fn, [optimizer_a, optimizer_b], scheduler_list,
            num_group, grad_clipping, distill, total_instance,
            contrastive_loss_fn=contrastive_loss_fn,
            contrastive_bank=contrastive_bank,
            contrastive_weight=now_contrastive_weight,
        )

        val_loss, val_metrics = None, None
        test_loss, test_metrics = None, None
        if process_pipeline == "Train_Val_Test":
            val_loss, val_metrics = dtfd_survival_eval_loop(device, model_list, val_loader, loss_fn, num_group, distill, total_instance)
            test_loss, test_metrics = dtfd_survival_eval_loop(device, model_list, test_loader, loss_fn, num_group, distill, total_instance)
        elif process_pipeline == "Train_Val":
            val_loss, val_metrics = dtfd_survival_eval_loop(device, model_list, val_loader, loss_fn, num_group, distill, total_instance)
        elif process_pipeline == "Train_Test" and epoch + 1 == args.General.num_epochs:
            test_loss, test_metrics = dtfd_survival_eval_loop(device, model_list, test_loader, loss_fn, num_group, distill, total_instance)

        print("----------------INFO----------------\n")
        print(f"EPOCH:{epoch+1}, Train_Loss:{train_loss}, Val_Loss:{val_loss}, Test_Loss:{test_loss}, Cost_Time:{cost_time}\n")
        print(f"Train_Metrics: {train_metrics}\n")
        print(f"Val_Metrics: {val_metrics}\n")
        print(f"Test_Metrics: {test_metrics}\n")
        _append_log(epoch_log, epoch, train_loss, train_metrics, val_loss, val_metrics, test_loss, test_metrics)

        score = _metric_value(val_metrics, val_loss, best_model_metric)
        state_dict = _dtfd_state_dict(model_list)
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_epoch = epoch + 1
            save_best_model(args, state_dict, best_epoch)

        if use_earlystop and _should_stop(epoch_log, patience, best_model_metric):
            print(f"Early Stop In EPOCH {epoch+1}!")
            save_last_model(args, state_dict, epoch + 1)
            _save_survival_log(args, epoch_log, best_epoch, process_pipeline)
            break

        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, state_dict, epoch + 1)
            _save_survival_log(args, epoch_log, best_epoch, process_pipeline)


def process_SURVIVAL_MIL(args):

    train_dataset, val_dataset, test_dataset = build_survival_datasets(args)
    process_pipeline = get_process_pipeline(val_dataset, test_dataset)
    args.General.process_pipeline = process_pipeline

    print("Survival bins:", train_dataset.bins)
    print("Train/Val/Test samples:", len(train_dataset), len(val_dataset), len(test_dataset))
    os.makedirs(args.Logs.now_log_dir, exist_ok=True)
    with open(os.path.join(args.Logs.now_log_dir, "survival_bins.json"), "w") as f:
        json.dump(train_dataset.bins.tolist(), f, indent=2)
    _save_survival_bin_distribution(args, train_dataset, val_dataset, test_dataset)

    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    set_global_seed(args.General.seed)

    device = _device_from_args(args)
    loss_fn = get_survival_loss(args.Model.criterion, alpha=float(_cfg(args, "Model", "alpha_surv", default=0.15)))
    if args.General.MODEL_NAME == "DTFD_MIL":
        process_DTFD_SURVIVAL_MIL(args, train_dataset, val_dataset, test_dataset, process_pipeline, device, loss_fn)
        return

    train_loader = _build_train_loader(args, train_dataset, generator)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.General.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.General.num_workers)
    print("Survival DataLoader Ready!")

    model = get_model_from_yaml(args).to(device)
    print("Survival Model Ready!")

    optimizer, base_lr = get_optimizer(args, model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    warmup_epoch = args.Model.scheduler.warmup
    grad_accumulation = int(_cfg(args, "Model", "grad_accumulation", default=1))
    survival_loss_weight = _survival_nll_weight(args)
    contrastive_loss_fn, contrastive_bank, contrastive_weight, contrastive_schedule = _build_contrastive(args, train_dataset)
    evidential_loss_fn = _build_evidential_survival(args)
    gate_balance_cfg = _cfg(args, "Model", "gate_load_balance", default={}) or {}
    gate_distribution_cfg = _cfg(args, "Model", "gate_distribution", default={}) or {}
    gate_high_weight_threshold = _as_float(gate_distribution_cfg.get("high_weight_threshold", 0.7), 0.7)
    gate_balance_weight = 0.0
    gate_balance_queue = None
    if _as_bool(gate_balance_cfg.get("use", False)):
        gate_balance_weight = _as_float(gate_balance_cfg.get("weight", 0.01), 0.01)
        gate_balance_queue = GateLoadBalanceQueue(
            max_size=int(gate_balance_cfg.get("queue_size", 128)),
            min_history=int(gate_balance_cfg.get("min_history", 16)),
            num_experts=2,
        )

    best_model_metric = str(_cfg(args, "General", "best_model_metric", default="c_index"))
    best_score = None
    best_epoch = None
    epoch_log = _init_log()
    earlystop_cfg = _cfg(args, "General", "earlystop", default={}) or {}
    use_earlystop = earlystop_cfg.get("use", False)
    patience = int(earlystop_cfg.get("patience", 0)) if use_earlystop else 0

    print("Start Survival Process!")
    print(f"Survival NLL loss weight: {survival_loss_weight}")
    if contrastive_loss_fn is not None:
        print(
            "Survival-Ranked Contrastive MIL enabled, "
            f"base_weight={contrastive_weight}, time_scale={contrastive_loss_fn.time_scale}, "
            f"warmup={contrastive_schedule.get('warmup_epochs', 0)}, ramp={contrastive_schedule.get('ramp_epochs', 0)}, "
            f"within_cancer={contrastive_schedule.get('within_cancer', False)}, "
            f"per_cancer_memory={contrastive_schedule.get('per_cancer_memory', False)}"
        )
    if gate_balance_queue is not None:
        print(
            "Cross-sample gate load balancing enabled, "
            f"weight={gate_balance_weight}, queue_size={gate_balance_queue.max_size}, "
            f"min_history={gate_balance_queue.min_history}"
        )
    if evidential_loss_fn is not None:
        print(
            "Evidential survival learning enabled, "
            f"survival_weight={evidential_loss_fn.survival_weight}, "
            f"local_survival_weight={evidential_loss_fn.local_survival_weight}, "
            f"wikg_survival_weight={evidential_loss_fn.wikg_survival_weight}, "
            f"kl_weight={evidential_loss_fn.kl_weight}, "
            f"expert_kl_weight={evidential_loss_fn.expert_kl_weight}, "
            f"kl_anneal_epochs={evidential_loss_fn.kl_anneal_epochs}, "
            f"wikg_evidence_reg_weight={evidential_loss_fn.wikg_evidence_reg_weight}, "
            f"wikg_evidence_reg_max_ratio={evidential_loss_fn.wikg_evidence_reg_max_ratio}, "
            f"wikg_evidence_reg_target={evidential_loss_fn.wikg_evidence_reg_target}, "
            f"wikg_evidence_reg_use_discounted={evidential_loss_fn.wikg_evidence_reg_use_discounted}, "
            f"wikg_evidence_reg_warmup_epochs={evidential_loss_fn.wikg_evidence_reg_warmup_epochs}"
        )
    print("Using Process Pipeline:", process_pipeline)

    for epoch in tqdm(range(args.General.num_epochs), colour="GREEN"):
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        now_scheduler = warmup_scheduler if epoch + 1 <= warmup_epoch else scheduler
        now_contrastive_weight = _contrastive_epoch_weight(contrastive_weight, contrastive_schedule, epoch)
        train_loss, cost_time, train_metrics = survival_train_loop(
            device,
            model,
            train_loader,
            loss_fn,
            optimizer,
            now_scheduler,
            grad_accumulation=grad_accumulation,
            contrastive_loss_fn=contrastive_loss_fn,
            contrastive_bank=contrastive_bank,
            contrastive_weight=now_contrastive_weight,
            gate_balance_queue=gate_balance_queue,
            gate_balance_weight=gate_balance_weight,
            evidential_loss_fn=evidential_loss_fn,
            survival_loss_weight=survival_loss_weight,
            epoch=epoch,
        )

        val_loss, val_metrics = None, None
        test_loss, test_metrics = None, None
        if process_pipeline == "Train_Val_Test":
            val_loss, val_metrics = survival_eval_loop(
                device,
                model,
                val_loader,
                loss_fn,
                gate_high_weight_threshold=gate_high_weight_threshold,
                evidential_loss_fn=evidential_loss_fn,
                survival_loss_weight=survival_loss_weight,
            )
            test_loss, test_metrics = survival_eval_loop(
                device,
                model,
                test_loader,
                loss_fn,
                gate_high_weight_threshold=gate_high_weight_threshold,
                evidential_loss_fn=evidential_loss_fn,
                survival_loss_weight=survival_loss_weight,
            )
        elif process_pipeline == "Train_Val":
            val_loss, val_metrics = survival_eval_loop(
                device,
                model,
                val_loader,
                loss_fn,
                gate_high_weight_threshold=gate_high_weight_threshold,
                evidential_loss_fn=evidential_loss_fn,
                survival_loss_weight=survival_loss_weight,
            )
        elif process_pipeline == "Train_Test" and epoch + 1 == args.General.num_epochs:
            test_loss, test_metrics = survival_eval_loop(
                device,
                model,
                test_loader,
                loss_fn,
                gate_high_weight_threshold=gate_high_weight_threshold,
                evidential_loss_fn=evidential_loss_fn,
                survival_loss_weight=survival_loss_weight,
            )

        print("----------------INFO----------------\n")
        print(
            f"EPOCH:{epoch+1}, Train_Loss:{train_loss}, Val_Loss:{val_loss}, "
            f"Test_Loss:{test_loss}, Cost_Time:{cost_time}\n"
        )
        print(f"Train_Metrics: {train_metrics}\n")
        print(f"Val_Metrics: {val_metrics}\n")
        print(f"Test_Metrics: {test_metrics}\n")

        _append_log(epoch_log, epoch, train_loss, train_metrics, val_loss, val_metrics, test_loss, test_metrics)

        score = _metric_value(val_metrics, val_loss, best_model_metric)
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_epoch = epoch + 1
            save_best_model(args, model.state_dict(), best_epoch)

        if use_earlystop and _should_stop(epoch_log, patience, best_model_metric):
            print(f"Early Stop In EPOCH {epoch+1}!")
            save_last_model(args, model.state_dict(), epoch + 1)
            _save_survival_log(args, epoch_log, best_epoch, process_pipeline)
            break

        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, model.state_dict(), epoch + 1)
            _save_survival_log(args, epoch_log, best_epoch, process_pipeline)


def test_SURVIVAL_MIL(yaml_args, test_dataset_csv, model_weight_path, test_log_dir, eval_split="test"):
    if eval_split not in ("test", "val"):
        raise ValueError(f"Unsupported evaluation split: {eval_split}")
    yaml_args.Dataset.dataset_csv_path = test_dataset_csv
    bins_path = os.path.join(os.path.dirname(model_weight_path), "survival_bins.json")
    survival_bins = None
    if os.path.exists(bins_path):
        with open(bins_path, "r") as f:
            survival_bins = json.load(f)
    elif eval_split == "val":
        raise FileNotFoundError(f"Validation export requires the original training bins: {bins_path}")
    test_dataset = SurvivalMILSlideDataset(test_dataset_csv, eval_split, **_survival_dataset_kwargs(yaml_args, bins=survival_bins))
    yaml_args.General.num_classes = test_dataset.num_bins
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=yaml_args.General.num_workers)

    device = _device_from_args(yaml_args)
    loss_fn = get_survival_loss(yaml_args.Model.criterion, alpha=float(_cfg(yaml_args, "Model", "alpha_surv", default=0.15)))
    evidential_loss_fn = _build_evidential_survival(yaml_args)
    survival_loss_weight = _survival_nll_weight(yaml_args)
    if yaml_args.General.MODEL_NAME == "DTFD_MIL":
        model_list = [module.to(device) for module in get_model_from_yaml(yaml_args)]
        state_dict = torch.load(model_weight_path, map_location=device)
        model_list[0].load_state_dict(state_dict["classifier"])
        model_list[1].load_state_dict(state_dict["attention"])
        model_list[2].load_state_dict(state_dict["dimReduction"])
        model_list[3].load_state_dict(state_dict["attCls"])
        test_loss, test_metrics, results = dtfd_survival_eval_loop(
            device,
            model_list,
            test_loader,
            loss_fn,
            int(yaml_args.Model.num_Group),
            yaml_args.Model.distill,
            int(yaml_args.Model.total_instance),
            return_results=True,
        )
    else:
        model = get_model_from_yaml(yaml_args).to(device)
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        gate_distribution_cfg = _cfg(yaml_args, "Model", "gate_distribution", default={}) or {}
        gate_high_weight_threshold = _as_float(gate_distribution_cfg.get("high_weight_threshold", 0.7), 0.7)
        test_loss, test_metrics, results = survival_eval_loop(
            device,
            model,
            test_loader,
            loss_fn,
            return_results=True,
            gate_high_weight_threshold=gate_high_weight_threshold,
            evidential_loss_fn=evidential_loss_fn,
            survival_loss_weight=survival_loss_weight,
        )

    os.makedirs(test_log_dir, exist_ok=True)
    prefix = "Test" if eval_split == "test" else "Val"
    log_to_save = {f"{eval_split}_loss": test_loss, f"{eval_split}_metrics": test_metrics}
    with open(os.path.join(test_log_dir, f"{prefix}_Log_{yaml_args.General.MODEL_NAME}_survival.txt"), "w") as f:
        f.write(str(log_to_save))
    with open(os.path.join(test_log_dir, f"{prefix}_Results_{yaml_args.General.MODEL_NAME}_survival.json"), "w") as f:
        json.dump(results, f, indent=2)
    # Auxiliary exporters retain Test-prefixed filenames; keep validation outputs isolated.
    auxiliary_dir = test_log_dir if eval_split == "test" else os.path.join(test_log_dir, "val_auxiliary")
    os.makedirs(auxiliary_dir, exist_ok=True)
    cancer_metrics_path = _save_cancer_metrics(auxiliary_dir, yaml_args.General.MODEL_NAME, test_metrics)
    gate_distribution_path = _save_gate_distribution(auxiliary_dir, yaml_args.General.MODEL_NAME, test_metrics)
    sample_routing_path = _save_sample_routing(auxiliary_dir, yaml_args.General.MODEL_NAME, results)

    print("----------------INFO----------------\n")
    print(f"Test_Loss:{test_loss}\n")
    print(f"Test_Metrics:{test_metrics}\n")
    if cancer_metrics_path is not None:
        print(f"Test cancer metrics saved at: {cancer_metrics_path}")
    if gate_distribution_path is not None:
        print(f"Test gate distribution saved at: {gate_distribution_path}")
    if sample_routing_path is not None:
        print(f"Test sample routing saved at: {sample_routing_path}")
    print(f"Test survival log saved at: {test_log_dir}")
    return test_loss, test_metrics
