import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process.SURVIVAL_MIL.process_survival_mil import _survival_bin_distribution_dataframe
from utils.survival_utils import SurvivalMILSlideDataset
from utils.yaml_utils import read_yaml


def _cfg(config, *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _is_empty(value):
    return value in [None, "", "null", {}]


def _as_bool(value, default=None):
    if value in [None, "", "null"]:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _resolve_csv_paths(args, yaml_args):
    if args.dataset_csv_path:
        return [Path(args.dataset_csv_path)]

    if args.dataset_root_dir:
        root = Path(args.dataset_root_dir)
        return sorted(root.glob("*.csv"))

    if yaml_args is not None:
        csv_path = _cfg(yaml_args, "Dataset", "dataset_csv_path", default=None)
        if not _is_empty(csv_path):
            return [Path(csv_path)]

        root_dir = _cfg(yaml_args, "Dataset", "dataset_root_dir", default=None)
        if not _is_empty(root_dir):
            return sorted(Path(root_dir).glob("*.csv"))

    raise ValueError("Provide --dataset_csv_path, --dataset_root_dir, or a YAML containing one of them.")


def _dataset_kwargs(args, yaml_args):
    survival_cfg = _cfg(yaml_args or {}, "Dataset", "survival", default={}) or {}
    n_bins = args.n_bins
    if n_bins is None:
        n_bins = int(survival_cfg.get("n_bins", _cfg(yaml_args or {}, "General", "num_classes", default=4)))

    patient_level = _as_bool(args.patient_level, default=None)
    if patient_level is None:
        patient_level = bool(survival_cfg.get("patient_level", True))

    return {
        "n_bins": int(n_bins),
        "patient_level": patient_level,
        "feature_mode": args.feature_mode or survival_cfg.get("feature_mode", "plain"),
        "bayesian_dir": _cfg(yaml_args or {}, "Dataset", "BeyesGuassian_pt_dir", default=None),
        "time_col": args.time_col or survival_cfg.get("time_col", None),
        "censorship_col": args.censorship_col or survival_cfg.get("censorship_col", None),
        "event_col": args.event_col or survival_cfg.get("event_col", None),
        "case_id_col": args.case_id_col or survival_cfg.get("case_id_col", None),
        "slide_path_col": args.slide_path_col or survival_cfg.get("slide_path_col", None),
        "cancer_col": args.cancer_col or survival_cfg.get("cancer_col", None),
    }


def _default_output_dir(args, yaml_args):
    if args.output_dir:
        return Path(args.output_dir)
    if yaml_args is not None:
        log_root = _cfg(yaml_args, "Logs", "log_root_dir", default=None)
        dataset_name = _cfg(yaml_args, "Dataset", "DATASET_NAME", default="survival_bins")
        if not _is_empty(log_root):
            return Path(log_root) / str(dataset_name) / "survival_bin_distribution"
    return REPO_ROOT / "logs" / "survival_bin_distribution"


def build_distribution_for_csv(csv_path, dataset_kwargs):
    train_dataset = SurvivalMILSlideDataset(csv_path, "train", **dataset_kwargs)
    val_dataset = SurvivalMILSlideDataset(
        csv_path,
        "val",
        **{**dataset_kwargs, "survival_bins": train_dataset.bins},
    )
    test_dataset = SurvivalMILSlideDataset(
        csv_path,
        "test",
        **{**dataset_kwargs, "survival_bins": train_dataset.bins},
    )
    distribution = _survival_bin_distribution_dataframe(
        (
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        )
    )
    return distribution, np.asarray(train_dataset.bins, dtype=float)


def save_distribution(csv_paths, dataset_kwargs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    saved_paths = []
    multiple = len(csv_paths) > 1

    for fold_idx, csv_path in enumerate(csv_paths, start=1):
        distribution, bins = build_distribution_for_csv(csv_path, dataset_kwargs)
        distribution.insert(0, "source_csv", str(csv_path))
        distribution.insert(1, "fold", fold_idx if multiple else "")

        suffix = f"_{csv_path.stem}" if multiple else ""
        distribution_path = output_dir / f"Survival_Bin_Distribution_By_Cancer{suffix}.csv"
        bins_path = output_dir / f"survival_bins{suffix}.json"
        distribution.to_csv(distribution_path, index=False)
        with open(bins_path, "w", encoding="utf-8") as f:
            json.dump(bins.tolist(), f, indent=2)
        saved_paths.extend([distribution_path, bins_path])
        all_frames.append(distribution)

    if multiple and all_frames:
        combined_path = output_dir / "Survival_Bin_Distribution_By_Cancer_ALL_FOLDS.csv"
        pd.concat(all_frames, ignore_index=True).to_csv(combined_path, index=False)
        saved_paths.append(combined_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute pan-cancer survival-bin distributions per cancer and split "
            "without running model training or testing."
        )
    )
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to a MIL survival YAML config.")
    parser.add_argument("--dataset_csv_path", type=str, default=None, help="Override or provide one split CSV path.")
    parser.add_argument("--dataset_root_dir", type=str, default=None, help="Directory containing split CSV files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for output CSV/JSON files.")
    parser.add_argument("--n_bins", type=int, default=None, help="Number of survival bins if no YAML value is used.")
    parser.add_argument("--patient_level", type=str, default=None, help="true/false. Defaults to YAML survival.patient_level.")
    parser.add_argument("--feature_mode", type=str, default=None, help="Feature mode for dataset metadata parsing.")
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--censorship_col", type=str, default=None)
    parser.add_argument("--event_col", type=str, default=None)
    parser.add_argument("--case_id_col", type=str, default=None)
    parser.add_argument("--slide_path_col", type=str, default=None)
    parser.add_argument("--cancer_col", type=str, default=None)
    args = parser.parse_args()

    yaml_args = read_yaml(args.yaml_path) if args.yaml_path else None
    csv_paths = _resolve_csv_paths(args, yaml_args)
    if not csv_paths:
        raise ValueError("No split CSV files found.")

    missing = [str(path) for path in csv_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split CSV file(s): {missing}")

    dataset_kwargs = _dataset_kwargs(args, yaml_args)
    output_dir = _default_output_dir(args, yaml_args)
    saved_paths = save_distribution(csv_paths, dataset_kwargs, output_dir)

    print("Survival bin distribution statistics saved:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
