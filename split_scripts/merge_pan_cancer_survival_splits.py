import argparse
import re
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")
REQUIRED_SUFFIXES = ("slide_path", "event_time", "censorship", "case_id")


def _fold_index(path):
    match = re.search(r"_survival_splits_(\d+)\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot determine fold index from: {path}")
    return int(match.group(1))


def _cancer_type(directory):
    suffix = "_survival_splits"
    if not directory.name.endswith(suffix):
        raise ValueError(f"Expected a '*{suffix}' directory, got: {directory}")
    return directory.name[: -len(suffix)]


def _split_frame(frame, split, cancer_type, source_path):
    required = [f"{split}_{suffix}" for suffix in REQUIRED_SUFFIXES]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {source_path}: {missing}")

    columns = [column for column in frame.columns if column.startswith(f"{split}_")]
    split_frame = frame.loc[frame[f"{split}_slide_path"].notna(), columns].copy()
    split_frame = split_frame[
        split_frame[f"{split}_slide_path"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)
    split_frame[f"{split}_cancer_type"] = cancer_type
    return split_frame


def _to_wide(split_frames):
    output = {}
    for split in SPLITS:
        split_frame = pd.concat(split_frames[split], ignore_index=True)
        for column in split_frame.columns:
            output[column] = pd.Series(split_frame[column].values)
    return pd.DataFrame(output)


def _validate_no_patient_leakage(frame, output_path):
    case_sets = {}
    for split in SPLITS:
        values = frame[f"{split}_case_id"].dropna().astype(str)
        case_sets[split] = set(values)
    overlaps = {
        f"{left}/{right}": sorted(case_sets[left] & case_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        if case_sets[left] & case_sets[right]
    }
    if overlaps:
        preview = {key: values[:3] for key, values in overlaps.items()}
        raise ValueError(f"Patient leakage detected in {output_path}: {preview}")


def merge_pan_cancer_splits(dataset_root, output_dir, output_name):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    source_dirs = sorted(
        directory
        for directory in dataset_root.glob("*_survival_splits")
        if directory.is_dir() and directory.resolve() != output_dir.resolve()
    )
    if not source_dirs:
        raise ValueError(f"No cancer survival split directories found under {dataset_root}")

    folds_by_cancer = {}
    for directory in source_dirs:
        cancer_type = _cancer_type(directory)
        fold_paths = {_fold_index(path): path for path in directory.glob("*.csv")}
        folds_by_cancer[cancer_type] = fold_paths

    fold_sets = {cancer: set(paths) for cancer, paths in folds_by_cancer.items()}
    expected_folds = next(iter(fold_sets.values()))
    inconsistent = {cancer: sorted(folds) for cancer, folds in fold_sets.items() if folds != expected_folds}
    if inconsistent:
        raise ValueError(f"Inconsistent folds across cancers: {inconsistent}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in sorted(expected_folds):
        split_frames = {split: [] for split in SPLITS}
        for cancer_type, fold_paths in sorted(folds_by_cancer.items()):
            source_path = fold_paths[fold]
            frame = pd.read_csv(source_path)
            for split in SPLITS:
                split_frames[split].append(_split_frame(frame, split, cancer_type, source_path))

        merged = _to_wide(split_frames)
        output_path = output_dir / f"{output_name}_survival_splits_{fold}.csv"
        _validate_no_patient_leakage(merged, output_path)
        merged.to_csv(output_path, index=False)

        counts = {
            split: int(merged[f"{split}_slide_path"].notna().sum())
            for split in SPLITS
        }
        print(f"Saved fold {fold}: {output_path} | {counts}")


def main():
    parser = argparse.ArgumentParser(description="Merge per-cancer survival folds into pan-cancer wide split CSVs.")
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets"))
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/PAN_CANCER_survival_splits"))
    parser.add_argument("--output_name", default="TCGA_PAN_CANCER_Survival_Splits")
    args = parser.parse_args()
    merge_pan_cancer_splits(args.dataset_root, args.output_dir, args.output_name)


if __name__ == "__main__":
    main()
