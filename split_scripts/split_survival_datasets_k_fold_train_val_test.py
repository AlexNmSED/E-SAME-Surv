import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.survival_utils import discretize_survival_times, make_survival_bins  # noqa: E402


def _event_to_censorship(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip().lower()
        if "deceased" in text or "dead" in text:
            return 0
        if "living" in text or "alive" in text:
            return 1
        if ":" in text:
            text = text.split(":", 1)[0]
        value = text
    return 1 - int(float(value))


def _safe_stratify(labels, test_size):
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    if isinstance(test_size, float):
        n_test = int(np.ceil(len(labels) * test_size))
    else:
        n_test = int(test_size)
    if n_test < len(counts):
        return None
    return labels


def _make_wide_split(train_df, val_df, test_df):
    data = {}
    for prefix, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        data[f"{prefix}_slide_path"] = pd.Series(split_df["slide_path"].values)
        data[f"{prefix}_event_time"] = pd.Series(split_df["event_time"].values)
        data[f"{prefix}_censorship"] = pd.Series(split_df["censorship"].values)
        data[f"{prefix}_case_id"] = pd.Series(split_df["case_id"].values)
        data[f"{prefix}_label"] = pd.Series(split_df["disc_label"].values)
    return pd.DataFrame(data)


def _expand_cases(case_ids, df):
    return df[df["case_id"].isin(case_ids)].reset_index(drop=True)


def prepare_survival_dataframe(args):
    df = pd.read_csv(args.csv_path)
    if args.slide_path_col in df.columns:
        slide_path = df[args.slide_path_col].astype(str)
    elif args.feature_dir and args.slide_id_col in df.columns:
        slide_path = df[args.slide_id_col].astype(str).map(
            lambda name: os.path.join(args.feature_dir, name + args.feature_ext)
        )
    else:
        raise ValueError("Provide either --slide_path_col in the CSV or --feature_dir with --slide_id_col.")

    if args.censorship_col in df.columns:
        censorship = df[args.censorship_col].astype(float)
    elif args.event_col and args.event_col in df.columns:
        censorship = df[args.event_col].map(_event_to_censorship).astype(float)
    else:
        raise ValueError("Provide --censorship_col or --event_col.")

    case_id = df[args.case_id_col].astype(str) if args.case_id_col in df.columns else slide_path
    out = pd.DataFrame(
        {
            "slide_path": slide_path,
            "event_time": df[args.time_col].astype(float),
            "censorship": censorship,
            "case_id": case_id,
        }
    )
    out = out.dropna(subset=["slide_path", "event_time", "censorship"]).reset_index(drop=True)

    case_records = out.drop_duplicates("case_id").to_dict("records")
    bins = make_survival_bins(case_records, n_bins=args.n_bins)
    case_df = out.drop_duplicates("case_id").copy()
    case_df["disc_label"] = discretize_survival_times(case_df["event_time"].values, bins)
    case_df["strata"] = case_df["disc_label"].astype(str) + "_" + case_df["censorship"].astype(int).astype(str)
    out = out.merge(case_df[["case_id", "disc_label", "strata"]], on="case_id", how="left")
    return out, case_df


def main():
    parser = argparse.ArgumentParser(description="Create MIL_BASELINE survival train/val/test split CSVs.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="survival_dataset")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument("--slide_path_col", type=str, default="slide_path")
    parser.add_argument("--slide_id_col", type=str, default="slide_id")
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--feature_ext", type=str, default=".pt")
    parser.add_argument("--time_col", type=str, default="survival_months")
    parser.add_argument("--censorship_col", type=str, default="censorship")
    parser.add_argument("--event_col", type=str, default=None)
    parser.add_argument("--case_id_col", type=str, default="case_id")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    slide_df, case_df = prepare_survival_dataframe(args)
    case_ids = case_df["case_id"].values
    strata = case_df["strata"].values

    if _safe_stratify(strata, max(1, len(case_ids) // args.k)) is not None:
        splitter = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(case_ids, strata)
    else:
        splitter = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(case_ids)

    for fold, (train_val_idx, test_idx) in enumerate(split_iter):
        train_val_cases = case_df.iloc[train_val_idx].reset_index(drop=True)
        test_cases = case_df.iloc[test_idx]["case_id"].values
        val_stratify = _safe_stratify(train_val_cases["strata"].values, args.val_ratio)
        train_cases, val_cases = train_test_split(
            train_val_cases["case_id"].values,
            test_size=args.val_ratio,
            random_state=args.seed + fold,
            stratify=val_stratify,
        )

        train_df = _expand_cases(train_cases, slide_df)
        val_df = _expand_cases(val_cases, slide_df)
        test_df = _expand_cases(test_cases, slide_df)
        split_df = _make_wide_split(train_df, val_df, test_df)
        save_path = os.path.join(args.save_dir, f"{args.dataset_name}_survival_splits_{fold}.csv")
        split_df.to_csv(save_path, index=False)
        print(f"Saved fold {fold}: {save_path}")


if __name__ == "__main__":
    main()

