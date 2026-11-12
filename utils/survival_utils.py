import os
import time
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


SURVIVAL_LOSSES = {"nll_surv", "ce_surv"}


def is_survival_config(args):
    general_task = str(args.get("General", {}).get("task", "")).lower()
    model_criterion = str(args.get("Model", {}).get("criterion", "")).lower()
    return general_task == "survival" or model_criterion in SURVIVAL_LOSSES


def _cfg(config, *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_float(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return np.nan
        if ":" in value:
            value = value.split(":", 1)[0]
    return float(value)


def _event_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "":
            return np.nan
        if "deceased" in text or "dead" in text or text in {"true", "yes", "event"}:
            return 1.0
        if "living" in text or "alive" in text or text in {"false", "no", "censored"}:
            return 0.0
        if ":" in text:
            text = text.split(":", 1)[0]
        return float(text)
    return float(value)


def _censorship_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "":
            return np.nan
        if "deceased" in text or "dead" in text or text == "event":
            return 0.0
        if "living" in text or "alive" in text or text == "censored":
            return 1.0
        if ":" in text:
            text = text.split(":", 1)[0]
        return float(text)
    return float(value)


def _find_column(df, names, group=None, allow_unprefixed=True):
    candidates = []
    for name in names:
        if name in [None, "", "null"]:
            continue
        if group is not None:
            candidates.append(f"{group}_{name}")
        if allow_unprefixed:
            candidates.append(name)
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _series_notna(series):
    return series.notna() & (series.astype(str).str.strip() != "")


def make_survival_bins(records, n_bins=4, eps=1e-6):
    times = np.array([record["event_time"] for record in records], dtype=float)
    censorships = np.array([record["censorship"] for record in records], dtype=float)
    valid = np.isfinite(times) & np.isfinite(censorships)
    times = times[valid]
    censorships = censorships[valid]
    if len(times) == 0:
        raise ValueError("No valid survival times found while building survival bins.")

    uncensored_times = times[censorships < 1]
    if len(uncensored_times) == 0:
        uncensored_times = times

    if len(np.unique(uncensored_times)) == 1:
        center = float(uncensored_times[0])
        width = max(abs(center) * 0.01, eps)
        bins = np.linspace(center - width, center + width, int(n_bins) + 1)
    else:
        _, bins = pd.qcut(
            uncensored_times,
            q=int(n_bins),
            retbins=True,
            labels=False,
            duplicates="drop",
        )
        if len(bins) < 2:
            bins = np.linspace(times.min() - eps, times.max() + eps, int(n_bins) + 1)

    bins = np.asarray(bins, dtype=float)
    bins[0] = min(bins[0], times.min() - eps)
    bins[-1] = max(bins[-1], times.max() + eps)
    return bins


def discretize_survival_times(times, bins):
    labels = pd.cut(
        pd.Series(times, dtype=float),
        bins=bins,
        labels=False,
        right=False,
        include_lowest=True,
    )
    labels = labels.astype(float).fillna(len(bins) - 2).astype(int).to_numpy()
    return np.clip(labels, 0, len(bins) - 2)


def _load_pt_or_h5(slide_path):
    if str(slide_path).lower().endswith(".h5"):
        with h5py.File(slide_path, "r") as h5_file:
            if "features" not in h5_file:
                raise KeyError(f"Missing 'features' dataset in {slide_path}")
            feat = torch.from_numpy(np.array(h5_file["features"]))
            coords = torch.from_numpy(np.array(h5_file["coords"])) if "coords" in h5_file else None
        return feat, coords

    loaded = torch.load(slide_path, map_location="cpu")
    coords = None
    if isinstance(loaded, dict):
        if "feats" in loaded:
            feat = loaded["feats"]
        elif "features" in loaded:
            feat = loaded["features"]
        else:
            raise ValueError(f"Unknown feature dict in {slide_path}, keys: {list(loaded.keys())}")
        coords = loaded.get("coords", None)
    else:
        feat = loaded
    return feat, coords


def _squeeze_feature(feat):
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)
    if len(feat.shape) == 3 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    return feat.float()


def _squeeze_coords(coords):
    if coords is None:
        return None
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    if len(coords.shape) == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)
    return coords.float()


class SurvivalMILSlideDataset(Dataset):
    """MIL survival dataset for MIL_BASELINE split CSVs.

    Expected split CSV columns are prefixed by group, for example:
    train_slide_path, train_event_time, train_censorship, train_case_id.
    If case_id columns are available, rows with the same case are merged into
    one patient-level bag by concatenating all slide features.
    """

    GROUPS = ("train", "val", "test")
    TIME_ALIASES = ("event_time", "survival_months", "survival_time", "time", "os_months", "OS")
    CENSORSHIP_ALIASES = ("censorship", "censor", "censored")
    EVENT_ALIASES = ("event", "event_observed", "status", "OS_STATUS")
    CASE_ALIASES = ("case_id", "patient_id", "patient", "subject_id")
    SLIDE_ALIASES = ("slide_path", "path", "feature_path", "pt_path", "h5_path")
    CANCER_ALIASES = ("cancer_type", "cancer", "cohort", "project_id", "project", "disease", "tumor_type")

    def __init__(
        self,
        dataset_info_csv_path,
        group,
        n_bins=4,
        survival_bins=None,
        patient_level=True,
        feature_mode="plain",
        bayesian_dir=None,
        time_col=None,
        censorship_col=None,
        event_col=None,
        case_id_col=None,
        slide_path_col=None,
        cancer_col=None,
        eps=1e-6,
    ):
        assert group in self.GROUPS, "group must be in [train, val, test]"
        self.dataset_info_csv_path = dataset_info_csv_path
        self.group = group
        self.feature_mode = feature_mode
        self.bayesian_dir = bayesian_dir
        self.patient_level = patient_level
        self.return_meta = False
        self.time_names = (time_col,) + self.TIME_ALIASES if time_col else self.TIME_ALIASES
        self.censorship_names = (censorship_col,) + self.CENSORSHIP_ALIASES if censorship_col else self.CENSORSHIP_ALIASES
        self.event_names = (event_col,) + self.EVENT_ALIASES if event_col else self.EVENT_ALIASES
        self.case_names = (case_id_col,) + self.CASE_ALIASES if case_id_col else self.CASE_ALIASES
        self.slide_names = (slide_path_col,) + self.SLIDE_ALIASES if slide_path_col else self.SLIDE_ALIASES
        self.cancer_names = (cancer_col,) + self.CANCER_ALIASES if cancer_col else self.CANCER_ALIASES
        df = pd.read_csv(dataset_info_csv_path)
        all_records = self._collect_records_for_bins(df)
        self.bins = np.asarray(survival_bins if survival_bins is not None else make_survival_bins(all_records, n_bins, eps), dtype=float)
        self.num_bins = len(self.bins) - 1

        records = self._read_group_records(df, group)
        self.samples = self._records_to_samples(records)
        if len(self.samples) > 0:
            labels = discretize_survival_times([sample["event_time"] for sample in self.samples], self.bins)
            for sample, label in zip(self.samples, labels):
                sample["label"] = int(label)
        self.labels_list = [sample["label"] for sample in self.samples]
        self.cancer_types = [sample["cancer_type"] for sample in self.samples]
        self.slide_path_list = [sample["slide_paths"][0] for sample in self.samples]
        self.slide_cls_ids = [
            np.where(np.asarray(self.labels_list) == cls_idx)[0]
            for cls_idx in range(self.num_bins)
        ]

    def _available_groups(self, df):
        groups = []
        for group in self.GROUPS:
            if _find_column(df, self.slide_names, group=group, allow_unprefixed=False) is not None:
                groups.append(group)
        if groups:
            return groups
        if "split" in df.columns:
            present = set(df["split"].dropna().astype(str).str.lower())
            return [group for group in self.GROUPS if group in present]
        return [self.group]

    def _collect_records_for_bins(self, df):
        records = []
        for group in self._available_groups(df):
            records.extend(self._read_group_records(df, group))
        return records

    def _read_group_records(self, df, group):
        row_style = "split" in df.columns and _find_column(df, self.slide_names, group=None, allow_unprefixed=True) is not None
        allow_unprefixed = row_style or _find_column(df, self.slide_names, group=group, allow_unprefixed=False) is None
        path_col = _find_column(df, self.slide_names, group=group, allow_unprefixed=allow_unprefixed)
        if path_col is None:
            return []

        rows = df
        if row_style and not path_col.startswith(f"{group}_"):
            rows = df[df["split"].astype(str).str.lower() == group]

        time_col = _find_column(rows, self.time_names, group=group, allow_unprefixed=allow_unprefixed)
        censorship_col = _find_column(rows, self.censorship_names, group=group, allow_unprefixed=allow_unprefixed)
        event_col = _find_column(rows, self.event_names, group=group, allow_unprefixed=allow_unprefixed)
        case_col = _find_column(rows, self.case_names, group=group, allow_unprefixed=allow_unprefixed)
        cancer_col = _find_column(rows, self.cancer_names, group=group, allow_unprefixed=allow_unprefixed)

        if time_col is None:
            raise ValueError(f"No survival time column found for group '{group}' in {self.dataset_info_csv_path}.")
        if censorship_col is None and event_col is None:
            raise ValueError(f"No censorship/event column found for group '{group}' in {self.dataset_info_csv_path}.")

        records = []
        for _, row in rows.iterrows():
            slide_path = row.get(path_col)
            if pd.isna(slide_path) or str(slide_path).strip() == "":
                continue
            event_time = _as_float(row.get(time_col))
            if censorship_col is not None:
                censorship = _censorship_value(row.get(censorship_col))
            else:
                event_observed = _event_value(row.get(event_col))
                censorship = 1.0 - event_observed
            if not np.isfinite(event_time) or not np.isfinite(censorship):
                continue
            case_id = row.get(case_col) if case_col is not None else str(slide_path)
            cancer_type = row.get(cancer_col) if cancer_col is not None else None
            if pd.isna(cancer_type) or str(cancer_type).strip() == "":
                cancer_type = None
            records.append(
                {
                    "case_id": str(case_id),
                    "slide_path": str(slide_path),
                    "event_time": float(event_time),
                    "censorship": float(censorship),
                    "cancer_type": None if cancer_type is None else str(cancer_type),
                }
            )
        return records

    def _records_to_samples(self, records):
        if not self.patient_level:
            return [
                {
                    "sample_id": record["slide_path"],
                    "case_id": record["case_id"],
                    "slide_paths": [record["slide_path"]],
                    "event_time": record["event_time"],
                    "censorship": record["censorship"],
                    "cancer_type": record["cancer_type"],
                }
                for record in records
            ]

        grouped = {}
        for record in records:
            grouped.setdefault(
                record["case_id"],
                {
                    "sample_id": record["case_id"],
                    "case_id": record["case_id"],
                    "slide_paths": [],
                    "event_time": record["event_time"],
                    "censorship": record["censorship"],
                    "cancer_type": record["cancer_type"],
                },
            )
            if grouped[record["case_id"]]["cancer_type"] != record["cancer_type"]:
                raise ValueError(
                    f"Patient '{record['case_id']}' has inconsistent cancer types in "
                    f"{self.dataset_info_csv_path}."
                )
            grouped[record["case_id"]]["slide_paths"].append(record["slide_path"])
        return list(grouped.values())

    def __len__(self):
        return len(self.samples)

    def is_None_Dataset(self):
        return self.__len__() == 0

    def is_with_labels(self):
        return len(self.labels_list) != 0

    def get_balanced_sampler(self, replacement=True):
        label_counts = Counter(self.labels_list)
        weights = [1.0 / label_counts[label] for label in self.labels_list]
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=replacement)

    def get_cancer_balanced_sampler(self, replacement=True, num_samples=None):
        missing = [sample["sample_id"] for sample in self.samples if sample["cancer_type"] is None]
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(
                "Cancer-balanced sampling requires a cancer/cohort column for every sample. "
                f"Missing cancer type for {len(missing)} sample(s), including: {preview}"
            )
        cancer_counts = Counter(self.cancer_types)
        weights = [1.0 / cancer_counts[cancer_type] for cancer_type in self.cancer_types]
        num_samples = len(weights) if num_samples in [None, "", "null"] else int(num_samples)
        if num_samples <= 0:
            raise ValueError("Cancer-balanced sampler num_samples must be positive.")
        if not replacement and num_samples > len(weights):
            raise ValueError("Cancer-balanced sampler cannot draw more than the dataset size without replacement.")
        return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=replacement)

    def getlabel(self, idx):
        return self.labels_list[idx]

    def _resolve_slide_path(self, slide_path):
        if self.feature_mode != "cdp" or self.bayesian_dir in [None, "", "null"]:
            return slide_path
        basename = os.path.basename(slide_path)
        if basename.endswith(".pt"):
            basename = basename.replace(".pt", "_bayesian_gaussian.pt")
        return os.path.join(self.bayesian_dir, basename)

    def _load_one_slide(self, slide_path):
        resolved_path = self._resolve_slide_path(slide_path)
        spatial_h5_modes = {"coord", "long", "dag", "sc", "adaptive_moe"}
        if self.feature_mode in spatial_h5_modes and not str(resolved_path).lower().endswith(".h5"):
            raise ValueError(
                f"Spatial survival mode '{self.feature_mode}' requires direct H5 input with embedded coordinates, "
                f"got: {resolved_path}"
            )
        feat, coords = _load_pt_or_h5(resolved_path)
        feat = _squeeze_feature(feat)

        if self.feature_mode in spatial_h5_modes:
            coords = _squeeze_coords(coords)
            if coords is None:
                raise ValueError(f"Missing 'coords' dataset in H5 input: {resolved_path}")
            if coords.shape[0] != feat.shape[0] or coords.shape[-1] < 2:
                raise ValueError(
                    f"H5 feature/coordinate shape mismatch in {resolved_path}: "
                    f"features={tuple(feat.shape)}, coords={tuple(coords.shape)}"
                )
            feat = torch.cat([feat, coords[:, :2].to(feat.device)], dim=-1)
        return feat

    @staticmethod
    def _offset_adaptive_moe_slide_coords(feature_list):
        """Keep coordinate neighborhoods slide-local in a patient-level bag."""
        offset = 0.0
        adjusted_features = []
        for feat in feature_list:
            coords = feat[:, -2:]
            if torch.isfinite(coords).all():
                coords = coords - coords.min(dim=0).values
                span = float((coords.max(dim=0).values - coords.min(dim=0).values).max().cpu())
                coords[:, 0] += offset
                offset += max(span, 1.0) * 10.0 + 1_000_000.0
                feat = feat.clone()
                feat[:, -2:] = coords
            adjusted_features.append(feat)
        return adjusted_features

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feature_list = [self._load_one_slide(path) for path in sample["slide_paths"]]
        if self.feature_mode == "adaptive_moe" and len(feature_list) > 1:
            feature_list = self._offset_adaptive_moe_slide_coords(feature_list)
        feat = torch.cat(feature_list, dim=0)
        label = torch.tensor(sample["label"], dtype=torch.long)
        event_time = torch.tensor(sample["event_time"], dtype=torch.float32)
        censorship = torch.tensor(sample["censorship"], dtype=torch.float32)
        if self.return_meta:
            return feat, label, event_time, censorship, {
                "sample_id": sample["sample_id"],
                "case_id": sample["case_id"],
                "cancer_type": "" if sample["cancer_type"] is None else sample["cancer_type"],
            }
        return feat, label, event_time, censorship


def nll_loss(hazards, survival, y, c, alpha=0.15, eps=1e-7):
    batch_size = len(y)
    y = y.view(batch_size, 1).long()
    c = c.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)
    survival_padded = torch.cat([torch.ones_like(c), survival], dim=1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(survival_padded, 1, y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(survival_padded, 1, y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    return ((1 - alpha) * neg_l + alpha * uncensored_loss).mean()


def ce_loss(hazards, survival, y, c, alpha=0.15, eps=1e-7):
    batch_size = len(y)
    y = y.view(batch_size, 1).long()
    c = c.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)
    survival_padded = torch.cat([torch.ones_like(c), survival], dim=1)
    reg = -(1 - c) * (
        torch.log(torch.gather(survival_padded, 1, y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, y).clamp(min=eps))
    )
    ce_l = -c * torch.log(torch.gather(survival, 1, y).clamp(min=eps)) - (1 - c) * torch.log(
        (1 - torch.gather(survival, 1, y)).clamp(min=eps)
    )
    return ((1 - alpha) * ce_l + alpha * reg).mean()


class NLLSurvLoss:
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, survival=None, y=None, c=None, alpha=None, **kwargs):
        y = kwargs.get("Y", y)
        survival = kwargs.get("S", survival)
        return nll_loss(hazards, survival, y, c, self.alpha if alpha is None else alpha)


class CrossEntropySurvLoss:
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, survival=None, y=None, c=None, alpha=None, **kwargs):
        y = kwargs.get("Y", y)
        survival = kwargs.get("S", survival)
        return ce_loss(hazards, survival, y, c, self.alpha if alpha is None else alpha)


def get_survival_loss(name, alpha=0.15):
    name = str(name).lower()
    if name == "nll_surv":
        return NLLSurvLoss(alpha=alpha)
    if name == "ce_surv":
        return CrossEntropySurvLoss(alpha=alpha)
    raise ValueError(f"Unknown survival loss: {name}. Supported: {sorted(SURVIVAL_LOSSES)}")


class EvidentialSurvivalLoss(nn.Module):
    """Evidential discrete survival loss for fused Dirichlet evidence.

    Events supervise the observed interval. Censored samples supervise survival
    beyond the censoring interval by summing probabilities over k > y. The KL
    term penalizes evidence on intervals incompatible with the observation.
    """

    def __init__(
        self,
        survival_weight=0.1,
        kl_weight=1e-3,
        kl_anneal_epochs=10,
        eps=1e-7,
        local_survival_weight=0.0,
        wikg_survival_weight=0.0,
        expert_kl_weight=0.0,
        wikg_evidence_reg_weight=0.0,
        wikg_evidence_reg_max_ratio=0.0,
        wikg_evidence_reg_target=0.0,
        wikg_evidence_reg_use_discounted=True,
        wikg_evidence_reg_warmup_epochs=0,
    ):
        super().__init__()
        self.survival_weight = float(survival_weight)
        self.kl_weight = float(kl_weight)
        self.kl_anneal_epochs = int(kl_anneal_epochs)
        self.eps = float(eps)
        self.local_survival_weight = float(local_survival_weight)
        self.wikg_survival_weight = float(wikg_survival_weight)
        self.expert_kl_weight = float(expert_kl_weight)
        self.wikg_evidence_reg_weight = float(wikg_evidence_reg_weight)
        self.wikg_evidence_reg_max_ratio = float(wikg_evidence_reg_max_ratio)
        self.wikg_evidence_reg_target = float(wikg_evidence_reg_target)
        self.wikg_evidence_reg_use_discounted = bool(wikg_evidence_reg_use_discounted)
        self.wikg_evidence_reg_warmup_epochs = int(wikg_evidence_reg_warmup_epochs)
        self.last_stats = {}

    @staticmethod
    def _dirichlet_kl_to_uniform(alpha):
        num_classes = alpha.shape[-1]
        alpha0 = alpha.sum(dim=-1, keepdim=True)
        return (
            torch.lgamma(alpha0)
            - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
            - torch.lgamma(alpha.new_tensor(float(num_classes)))
            + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha0))).sum(dim=-1, keepdim=True)
        ).mean()

    def _kl_mask(self, y, c, num_classes):
        class_index = torch.arange(num_classes, device=y.device).view(1, -1)
        y = y.view(-1, 1).long()
        c = c.view(-1, 1).float()
        event_allowed = class_index == y
        censored_allowed = class_index > y
        allowed = torch.where(c < 0.5, event_allowed, censored_allowed)
        return allowed

    def _survival_nll_from_probability(self, probabilities, y, c):
        probabilities = probabilities.clamp_min(self.eps)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        num_classes = probabilities.shape[-1]
        event_prob = torch.gather(probabilities, 1, y).clamp_min(self.eps)
        after_mask = torch.arange(num_classes, device=probabilities.device).view(1, -1) > y
        censored_prob = (probabilities * after_mask.float()).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return (-(1.0 - c) * torch.log(event_prob) - c * torch.log(censored_prob)).mean()

    def _masked_kl_to_uniform(self, alpha, y, c):
        alpha = alpha.clamp_min(1.0 + self.eps)
        allowed = self._kl_mask(y, c, alpha.shape[-1])
        alpha_tilde = torch.where(allowed, torch.ones_like(alpha), alpha)
        return self._dirichlet_kl_to_uniform(alpha_tilde)

    def _wikg_evidence_regularization(self, model_output, reference, epoch=None):
        zero = reference.new_tensor(0.0)
        if self.wikg_evidence_reg_warmup_epochs > 0 and epoch is not None:
            reg_scale = 0.0 if (int(epoch) + 1) <= self.wikg_evidence_reg_warmup_epochs else 1.0
        else:
            reg_scale = 1.0
        effective_weight = self.wikg_evidence_reg_weight * reg_scale
        if effective_weight <= 0:
            return zero, {
                "edl_wikg_evidence_reg_loss": 0.0,
                "edl_wikg_evidence_reg_weight": effective_weight,
                "edl_wikg_evidence_reg_anneal": reg_scale,
                "edl_wikg_evidence_reg_ratio_excess": 0.0,
                "edl_wikg_evidence_mass": 0.0,
                "edl_local_evidence_mass": 0.0,
            }

        if self.wikg_evidence_reg_use_discounted:
            wikg_key = "discounted_wikg_evidence"
            local_key = "discounted_local_evidence"
        else:
            wikg_key = "wikg_evidence"
            local_key = "local_evidence"
        wikg_evidence = model_output.get(wikg_key)
        if not torch.is_tensor(wikg_evidence):
            return zero, {
                "edl_wikg_evidence_reg_loss": 0.0,
                "edl_wikg_evidence_reg_weight": effective_weight,
                "edl_wikg_evidence_reg_anneal": reg_scale,
                "edl_wikg_evidence_reg_ratio_excess": 0.0,
                "edl_wikg_evidence_mass": 0.0,
                "edl_local_evidence_mass": 0.0,
            }

        wikg_mass = wikg_evidence.sum(dim=-1, keepdim=True)
        reg_terms = []
        ratio_excess_mean = zero
        local_mass_value = 0.0
        local_evidence = model_output.get(local_key)
        if self.wikg_evidence_reg_max_ratio > 0 and torch.is_tensor(local_evidence):
            local_mass = local_evidence.sum(dim=-1, keepdim=True).detach()
            allowed_mass = (self.wikg_evidence_reg_max_ratio * local_mass).clamp_min(1.0)
            ratio_excess = torch.relu(wikg_mass - allowed_mass)
            ratio_loss = torch.square(ratio_excess / allowed_mass).mean()
            reg_terms.append(ratio_loss)
            ratio_excess_mean = ratio_excess.mean()
            local_mass_value = float(local_mass.mean().detach().cpu())

        if self.wikg_evidence_reg_target > 0:
            target = wikg_mass.new_tensor(self.wikg_evidence_reg_target).clamp_min(1.0)
            target_excess = torch.relu(wikg_mass - target)
            reg_terms.append(torch.square(target_excess / target).mean())

        reg_loss = sum(reg_terms) if reg_terms else zero
        return effective_weight * reg_loss, {
            "edl_wikg_evidence_reg_loss": float(reg_loss.detach().cpu()),
            "edl_wikg_evidence_reg_weight": effective_weight,
            "edl_wikg_evidence_reg_anneal": reg_scale,
            "edl_wikg_evidence_reg_ratio_excess": float(ratio_excess_mean.detach().cpu()),
            "edl_wikg_evidence_mass": float(wikg_mass.mean().detach().cpu()),
            "edl_local_evidence_mass": local_mass_value,
        }

    def forward(self, model_output, y, c, epoch=None):
        if not isinstance(model_output, dict) or "fused_probability" not in model_output or "fused_alpha" not in model_output:
            raise ValueError("EvidentialSurvivalLoss requires model outputs 'fused_probability' and 'fused_alpha'.")

        probabilities = model_output["fused_probability"].clamp_min(self.eps)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        alpha = model_output["fused_alpha"].clamp_min(1.0 + self.eps)
        y = y.view(-1, 1).long()
        c = c.view(-1, 1).float()
        num_classes = probabilities.shape[-1]

        edl_surv_loss = self._survival_nll_from_probability(probabilities, y, c)
        kl_loss = self._masked_kl_to_uniform(alpha, y, c)
        if epoch is None or self.kl_anneal_epochs <= 0:
            anneal = 1.0
        else:
            anneal = min(1.0, max(0.0, float(epoch + 1) / float(self.kl_anneal_epochs)))
        local_surv_loss = probabilities.new_tensor(0.0)
        wikg_surv_loss = probabilities.new_tensor(0.0)
        expert_kl_loss = probabilities.new_tensor(0.0)
        expert_kl_terms = []
        local_prob = model_output.get("local_probability")
        wikg_prob = model_output.get("wikg_probability")
        local_alpha = model_output.get("local_alpha")
        wikg_alpha = model_output.get("wikg_alpha")
        if self.local_survival_weight > 0 and torch.is_tensor(local_prob):
            local_surv_loss = self._survival_nll_from_probability(local_prob, y, c)
            if self.expert_kl_weight > 0 and torch.is_tensor(local_alpha):
                expert_kl_terms.append(self._masked_kl_to_uniform(local_alpha, y, c))
        if self.wikg_survival_weight > 0 and torch.is_tensor(wikg_prob):
            wikg_surv_loss = self._survival_nll_from_probability(wikg_prob, y, c)
            if self.expert_kl_weight > 0 and torch.is_tensor(wikg_alpha):
                expert_kl_terms.append(self._masked_kl_to_uniform(wikg_alpha, y, c))
        if expert_kl_terms:
            expert_kl_loss = torch.stack(expert_kl_terms).mean()
        wikg_reg_loss, wikg_reg_stats = self._wikg_evidence_regularization(model_output, probabilities, epoch=epoch)
        total = (
            self.survival_weight * edl_surv_loss
            + self.local_survival_weight * local_surv_loss
            + self.wikg_survival_weight * wikg_surv_loss
            + self.kl_weight * anneal * kl_loss
            + self.expert_kl_weight * anneal * expert_kl_loss
            + wikg_reg_loss
        )
        self.last_stats = {
            "evidential_loss": float(total.detach().cpu()),
            "edl_surv_loss": float(edl_surv_loss.detach().cpu()),
            "edl_local_surv_loss": float(local_surv_loss.detach().cpu()),
            "edl_wikg_surv_loss": float(wikg_surv_loss.detach().cpu()),
            "edl_kl_loss": float(kl_loss.detach().cpu()),
            "edl_expert_kl_loss": float(expert_kl_loss.detach().cpu()),
            "edl_surv_weight": self.survival_weight,
            "edl_local_surv_weight": self.local_survival_weight,
            "edl_wikg_surv_weight": self.wikg_survival_weight,
            "edl_kl_weight": self.kl_weight * anneal,
            "edl_expert_kl_weight": self.expert_kl_weight * anneal,
            "edl_kl_anneal": anneal,
        }
        self.last_stats.update(wikg_reg_stats)
        return total


def survival_from_logits(logits):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1 - hazards, dim=1)
    return hazards, survival


def extract_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["logits"]
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def extract_wsi_feature(model_output, fallback):
    if isinstance(model_output, dict) and "WSI_feature" in model_output:
        feature = model_output["WSI_feature"]
    else:
        feature = fallback
    if feature.dim() == 1:
        feature = feature.unsqueeze(0)
    return feature.reshape(feature.shape[0], -1)


def extract_auxiliary_loss(model_output, fallback):
    if isinstance(model_output, dict) and "aux_loss" in model_output:
        return model_output["aux_loss"]
    return fallback.new_tensor(0.0)


def extract_model_metrics(model_output):
    if not isinstance(model_output, dict):
        return {}
    metrics = {}
    for key in (
        "aux_loss",
        "orth_loss",
        "expert_cosine",
        "alignment_norm_gap",
        "gate_entropy",
        "dse_entropy",
        "t_wsi_norm",
        "t_mean_norm",
            "t_mean_token_norm",
            "t_pheno_entropy",
            "fusion_feature_norm",
            "uses_physical_coords",
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
        value = model_output.get(key)
        if torch.is_tensor(value) and value.numel() > 0:
            metrics[key] = float(value.detach().mean().cpu())
    gate_weights = model_output.get("gate_weights")
    if torch.is_tensor(gate_weights) and gate_weights.shape[-1] >= 2:
        metrics["gate_local"] = float(gate_weights[..., 0].detach().mean().cpu())
        metrics["gate_wikg"] = float(gate_weights[..., 1].detach().mean().cpu())
    return metrics


class GateLoadBalanceQueue:
    """Tracks cross-sample routing usage without retaining WSI computation graphs."""

    def __init__(self, max_size=128, min_history=16, num_experts=2):
        self.max_size = int(max_size)
        self.min_history = int(min_history)
        self.num_experts = int(num_experts)
        if self.max_size <= 0:
            raise ValueError("Gate load-balance queue size must be positive.")
        if self.min_history <= 0 or self.min_history > self.max_size:
            raise ValueError("Gate load-balance min_history must be in [1, max_size].")
        if self.num_experts <= 1:
            raise ValueError("Gate load balancing requires at least two experts.")
        self._gates = []

    def __len__(self):
        return len(self._gates)

    def enqueue(self, gate_weights):
        gates = gate_weights.detach().reshape(-1, self.num_experts).float().cpu()
        self._gates.extend(gate.clone() for gate in gates)
        if len(self._gates) > self.max_size:
            self._gates = self._gates[-self.max_size :]

    def historical_usage(self, reference):
        if not self._gates:
            return None
        return torch.stack(self._gates).mean(dim=0).to(device=reference.device, dtype=reference.dtype)

    def loss(self, gate_weights):
        gates = gate_weights.reshape(-1, self.num_experts)
        if len(self._gates) < self.min_history:
            return gates.sum() * 0.0
        usage = self.historical_usage(gates)
        # Switch-style load-aware routing: a balanced history makes this constant,
        # while an overused expert receives a larger gradient penalty.
        return self.num_experts * torch.sum(usage.detach() * gates.mean(dim=0))

    def imbalance(self, gate_weights=None):
        if gate_weights is None and not self._gates:
            return 0.0
        if gate_weights is not None:
            reference = gate_weights.reshape(-1, self.num_experts)
            current = reference.detach().mean(dim=0)
        else:
            reference = self._gates[0]
            current = None
        usage = self.historical_usage(reference)
        if current is not None:
            usage = (usage * len(self._gates) + current) / (len(self._gates) + 1) if usage is not None else current
        return float((self.num_experts * usage.square().sum() - 1.0).detach().cpu())


def forward_with_wsi_feature(model, bag):
    try:
        return model(bag, return_WSI_feature=True)
    except TypeError:
        return model(bag)


def unpack_survival_batch(batch):
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected a tuple/list survival batch, got {type(batch)!r}.")
    if len(batch) == 4:
        bag, label, event_time, censorship = batch
        return bag, label, event_time, censorship, None
    if len(batch) == 5:
        bag, label, event_time, censorship, metadata = batch
        return bag, label, event_time, censorship, metadata
    raise ValueError(f"Expected a survival batch with 4 or 5 fields, got {len(batch)}.")


def normalize_cancer_type(cancer_type, unknown="UNKNOWN"):
    if cancer_type is None:
        return unknown
    if isinstance(cancer_type, bytes):
        cancer_type = cancer_type.decode("utf-8", errors="ignore")
    text = str(cancer_type).strip()
    return text if text else unknown


def metadata_cancer_types(metadata, batch_size, unknown="UNKNOWN"):
    if metadata is None:
        return [unknown] * int(batch_size)
    if isinstance(metadata, dict):
        values = metadata.get("cancer_type", None)
    else:
        values = metadata
    if values is None:
        return [unknown] * int(batch_size)
    if isinstance(values, (list, tuple)):
        cancer_types = [normalize_cancer_type(value, unknown=unknown) for value in values]
    else:
        cancer_types = [normalize_cancer_type(values, unknown=unknown)]
    if len(cancer_types) == 1 and batch_size > 1:
        cancer_types = cancer_types * int(batch_size)
    if len(cancer_types) != int(batch_size):
        raise ValueError(f"Expected {batch_size} cancer type value(s), got {len(cancer_types)}.")
    return cancer_types


class SurvivalRankedMemoryBank:
    def __init__(self, max_size=256, per_cancer=False, unknown_cancer="UNKNOWN"):
        self.max_size = int(max_size)
        self.per_cancer = bool(per_cancer)
        self.unknown_cancer = str(unknown_cancer)
        self.features = []
        self.event_times = []
        self.censorships = []
        self.cancer_types = []
        self._banks = {}

    def __len__(self):
        if self.per_cancer:
            return sum(len(bank["features"]) for bank in self._banks.values())
        return len(self.features)

    def _empty_bank(self):
        return {"features": [], "event_times": [], "censorships": [], "cancer_types": []}

    def _normalize_cancer(self, cancer_type):
        return normalize_cancer_type(cancer_type, unknown=self.unknown_cancer)

    def _trim(self, bank):
        overflow = len(bank["features"]) - self.max_size
        if overflow > 0:
            bank["features"] = bank["features"][overflow:]
            bank["event_times"] = bank["event_times"][overflow:]
            bank["censorships"] = bank["censorships"][overflow:]
            bank["cancer_types"] = bank["cancer_types"][overflow:]

    def _enqueue_one(self, bank, feature, event_time, censorship, cancer_type):
        bank["features"].append(feature.float())
        bank["event_times"].append(float(event_time))
        bank["censorships"].append(float(censorship))
        bank["cancer_types"].append(self._normalize_cancer(cancer_type))
        self._trim(bank)

    def enqueue(self, features, event_times, censorships, cancer_types=None):
        if self.max_size <= 0:
            return
        features = features.detach().cpu()
        event_times = event_times.detach().cpu().view(-1)
        censorships = censorships.detach().cpu().view(-1)
        cancer_types = metadata_cancer_types(cancer_types, features.shape[0], unknown=self.unknown_cancer)
        for feature, event_time, censorship, cancer_type in zip(features, event_times, censorships, cancer_types):
            cancer_type = self._normalize_cancer(cancer_type)
            if self.per_cancer:
                bank = self._banks.setdefault(cancer_type, self._empty_bank())
                self._enqueue_one(bank, feature, event_time, censorship, cancer_type)
            else:
                self.features.append(feature.float())
                self.event_times.append(float(event_time))
                self.censorships.append(float(censorship))
                self.cancer_types.append(cancer_type)
                overflow = len(self.features) - self.max_size
                if overflow > 0:
                    self.features = self.features[overflow:]
                    self.event_times = self.event_times[overflow:]
                    self.censorships = self.censorships[overflow:]
                    self.cancer_types = self.cancer_types[overflow:]

    def _select_bank(self, cancer_type=None):
        if self.per_cancer:
            if cancer_type is None:
                merged = self._empty_bank()
                for bank in self._banks.values():
                    merged["features"].extend(bank["features"])
                    merged["event_times"].extend(bank["event_times"])
                    merged["censorships"].extend(bank["censorships"])
                    merged["cancer_types"].extend(bank["cancer_types"])
                return merged
            return self._banks.get(self._normalize_cancer(cancer_type), self._empty_bank())

        bank = {
            "features": self.features,
            "event_times": self.event_times,
            "censorships": self.censorships,
            "cancer_types": self.cancer_types,
        }
        if cancer_type is None:
            return bank
        cancer_type = self._normalize_cancer(cancer_type)
        keep = [idx for idx, stored_cancer in enumerate(self.cancer_types) if stored_cancer == cancer_type]
        return {
            "features": [self.features[idx] for idx in keep],
            "event_times": [self.event_times[idx] for idx in keep],
            "censorships": [self.censorships[idx] for idx in keep],
            "cancer_types": [self.cancer_types[idx] for idx in keep],
        }

    def tensors(self, device, cancer_type=None):
        bank = self._select_bank(cancer_type=cancer_type)
        if len(bank["features"]) == 0:
            return None, None, None
        features = torch.stack(bank["features"], dim=0).to(device)
        event_times = torch.tensor(bank["event_times"], dtype=torch.float32, device=device)
        censorships = torch.tensor(bank["censorships"], dtype=torch.float32, device=device)
        return features, event_times, censorships

    def cancer_counts(self):
        if self.per_cancer:
            return {cancer: len(bank["features"]) for cancer, bank in self._banks.items()}
        return dict(Counter(self.cancer_types))


class SurvivalRankedContrastiveLoss(nn.Module):
    """Censor-aware continuous survival-time guided contrastive geometry.

    Event-event pairs regress feature distance to a continuous survival-time
    target. Event-censored pairs use a one-sided hinge because censored times
    are lower bounds rather than exact event times. Censored-censored pairs are
    ignored by default because they are usually too ambiguous for exact
    distance regression in heavily censored cohorts.
    """

    def __init__(
        self,
        time_scale=60.0,
        event_weight=2.0,
        censored_weight=0.25,
        ambiguous_weight=0.0,
        censored_pair_weight=0.0,
        max_target_distance=0.8,
        hinge_margin=0.0,
        smooth_l1_beta=0.05,
        min_pairs=1,
        within_cancer=False,
        unknown_cancer="UNKNOWN",
    ):
        super().__init__()
        self.time_scale = float(time_scale)
        self.event_weight = float(event_weight)
        self.censored_weight = float(censored_weight)
        self.ambiguous_weight = float(ambiguous_weight)
        self.censored_pair_weight = float(censored_pair_weight)
        self.max_target_distance = float(max_target_distance)
        self.hinge_margin = float(hinge_margin)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.min_pairs = int(min_pairs)
        self.within_cancer = bool(within_cancer)
        self.unknown_cancer = str(unknown_cancer)
        self.last_stats = {}

    def _smooth_l1(self, value, target):
        beta = max(self.smooth_l1_beta, 1e-6)
        diff = torch.abs(value - target)
        return torch.where(diff < beta, 0.5 * diff.pow(2) / beta, diff - 0.5 * beta)

    def _target_distance(self, delta):
        time_scale = max(self.time_scale, 1e-6)
        return torch.clamp(delta / time_scale, min=0.0, max=self.max_target_distance)

    @staticmethod
    def _add_group_loss(parts, weights, raw_loss, mask, group_weight):
        if group_weight <= 0 or not mask.any():
            return 0
        parts.append(raw_loss[mask].mean() * group_weight)
        weights.append(raw_loss.new_tensor(group_weight))
        return int(mask.sum().item())

    def forward(self, features, event_times, censorships, bank, cancer_types=None):
        features = F.normalize(features.float(), dim=1)
        event_times = event_times.float().view(-1)
        censorships = censorships.float().view(-1)
        cancer_types = metadata_cancer_types(cancer_types, features.shape[0], unknown=self.unknown_cancer)

        losses = []
        valid_pairs = 0
        stats = {
            "event_event_pairs": 0,
            "event_censored_pairs": 0,
            "ambiguous_pairs": 0,
            "censored_censored_pairs": 0,
            "within_cancer_pairs": 0,
            "skipped_empty_cancer_bank": 0,
        }

        global_bank = None
        if not self.within_cancer:
            global_bank = bank.tensors(features.device)
            if global_bank[0] is None:
                self.last_stats = stats
                return features.new_tensor(0.0)
            global_bank = (
                F.normalize(global_bank[0].float(), dim=1),
                global_bank[1],
                global_bank[2],
            )

        for idx in range(features.shape[0]):
            if self.within_cancer:
                bank_features, bank_times, bank_censorships = bank.tensors(
                    features.device,
                    cancer_type=cancer_types[idx],
                )
                if bank_features is None:
                    stats["skipped_empty_cancer_bank"] += 1
                    continue
                bank_features = F.normalize(bank_features.float(), dim=1)
            else:
                bank_features, bank_times, bank_censorships = global_bank

            feature = features[idx : idx + 1]
            t_cur = event_times[idx]
            c_cur = censorships[idx]
            delta = torch.abs(t_cur - bank_times)
            target_distance = self._target_distance(delta)
            feature_distance = 1.0 - torch.mm(feature, bank_features.t()).squeeze(0)

            cur_event = c_cur < 0.5
            ref_event = bank_censorships < 0.5
            event_event = cur_event & ref_event
            censored_censored = (~cur_event) & (~ref_event)

            one_censored = cur_event ^ ref_event
            cur_event_ref_censored = cur_event & (~ref_event)
            ref_event_cur_censored = ref_event & (~cur_event)
            comparable_event_censored = (
                (cur_event_ref_censored & (t_cur <= bank_times))
                | (ref_event_cur_censored & (bank_times <= t_cur))
            )
            ambiguous_event_censored = one_censored & (~comparable_event_censored)

            group_losses = []
            group_weights = []
            sample_valid_pairs = 0
            event_loss = self._smooth_l1(feature_distance, target_distance)
            count = self._add_group_loss(group_losses, group_weights, event_loss, event_event, self.event_weight)
            stats["event_event_pairs"] += count
            valid_pairs += count
            sample_valid_pairs += count

            hinge_target = torch.clamp(target_distance + self.hinge_margin, min=0.0, max=self.max_target_distance)
            censored_loss = F.relu(hinge_target - feature_distance).pow(2)
            count = self._add_group_loss(group_losses, group_weights, censored_loss, comparable_event_censored, self.censored_weight)
            stats["event_censored_pairs"] += count
            valid_pairs += count
            sample_valid_pairs += count

            ambiguous_loss = F.relu(0.5 * hinge_target - feature_distance).pow(2)
            count = self._add_group_loss(group_losses, group_weights, ambiguous_loss, ambiguous_event_censored, self.ambiguous_weight)
            stats["ambiguous_pairs"] += count
            valid_pairs += count
            sample_valid_pairs += count

            censored_censored_loss = self._smooth_l1(feature_distance, target_distance)
            count = self._add_group_loss(
                group_losses,
                group_weights,
                censored_censored_loss,
                censored_censored,
                self.censored_pair_weight,
            )
            stats["censored_censored_pairs"] += count
            valid_pairs += count
            sample_valid_pairs += count
            if self.within_cancer:
                stats["within_cancer_pairs"] += sample_valid_pairs

            if group_losses:
                losses.append(torch.stack(group_losses).sum() / torch.stack(group_weights).sum().clamp(min=1e-6))

        if valid_pairs < self.min_pairs or len(losses) == 0:
            self.last_stats = stats
            return features.new_tensor(0.0)
        loss = torch.stack(losses).mean()
        stats["valid_pairs"] = valid_pairs
        self.last_stats = stats
        return loss


# class SurvivalRankedContrastiveLoss(nn.Module):
#     """Censor-aware pairwise ranking loss aligned with C-index.

#     Core changes vs. original:
#       Fix 1 (misaligned objective): Replace smooth_l1 distance regression with
#         pairwise logistic ranking loss (RankNet). The loss directly maximises
#         the probability that the higher-risk sample has a larger risk score,
#         which is exactly what C-index measures.

#       Fix 2 (symmetric target): Construct *directed* pairs (i, j) where
#         t_i < t_j (i dies earlier), then require score_i > score_j.
#         Direction is baked into pair construction; the loss only needs to
#         push the margin in one direction.

#     Risk score convention: score = dot(feat_i, bank_feat_j).
#       Higher dot-product  →  more similar to high-risk bank samples  →  higher risk.
#       Caller should ensure bank contains a balanced mix of short/long survivors.
#       Alternatively, flip the sign if your feature space is inverted.

#     Pair types after directed construction (t_i < t_j):
#       event–event:   both observed  → hard logistic loss
#       event–censored: i is event, j is censored with t_j >= t_i
#                      (j is a *lower bound*, so we only penalise if margin
#                      already violated, via a soft hinge on the score gap)
#       censored–event / ambiguous: skipped or down-weighted

#     Parameters
#     ----------
#     margin : float
#         Minimum required score gap  score_i - score_j  for event–censored pairs.
#         Set to 0 for pure lower-bound enforcement, >0 for an explicit margin.
#     event_weight : float
#         Weight for event–event pairs.
#     censored_weight : float
#         Weight for (event, censored) comparable pairs.
#     time_weight_scale : float or None
#         When set, scale each pair's contribution by
#         sigmoid(|Δt| / time_weight_scale) so pairs with a larger survival-time
#         gap contribute more strongly.  None disables this.
#     min_pairs : int
#         Minimum number of valid pairs; return 0 loss if below threshold.
#     """

#     def __init__(
#         self,
#         time_scale: float = 60.0,
#         margin: float = 0.1,
#         event_weight: float = 1.0,
#         censored_weight: float = 0.5,
#         time_weight_scale: float | None = None,
#         min_pairs: int = 1,
#     ):
#         super().__init__()
#         self.time_scale = float(time_scale)
#         self.margin = float(margin)
#         self.event_weight = float(event_weight)
#         self.censored_weight = float(censored_weight)
#         self.time_weight_scale = time_weight_scale
#         self.min_pairs = int(min_pairs)
#         self.last_stats: dict = {}

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def _time_pair_weight(self, delta: torch.Tensor) -> torch.Tensor:
#         """Optional per-pair weight based on survival-time gap magnitude."""
#         if self.time_weight_scale is None:
#             return torch.ones_like(delta)
#         scale = max(float(self.time_weight_scale), 1e-6)
#         # sigmoid maps delta/scale → (0.5, 1); pairs with large |Δt| get ~1.
#         return torch.sigmoid(delta / scale)

#     @staticmethod
#     def _pairwise_logistic(score_i: torch.Tensor, score_j: torch.Tensor) -> torch.Tensor:
#         """RankNet loss: -log σ(score_i - score_j).

#         Numerically stable form: log(1 + exp(-(score_i - score_j))).
#         Gradient pushes score_i up and score_j down.
#         """
#         return F.softplus(-(score_i - score_j))  # softplus(x) = log(1+exp(x))

#     @staticmethod
#     def _hinge_ranking(score_i: torch.Tensor, score_j: torch.Tensor, margin: float) -> torch.Tensor:
#         """Squared hinge: penalise only when score_i - score_j < margin."""
#         return F.relu(margin - (score_i - score_j)).pow(2)

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------

#     def forward(
#         self,
#         features: torch.Tensor,       # (B, D)  current-batch features
#         event_times: torch.Tensor,    # (B,)    survival / censoring times
#         censorships: torch.Tensor,    # (B,)    1 = censored, 0 = event
#         bank,                         # momentum bank with .tensors() API
#     ) -> torch.Tensor:

#         bank_features, bank_times, bank_censorships = bank.tensors(features.device)
#         if bank_features is None:
#             self.last_stats = {}
#             return features.new_tensor(0.0)

#         # ---- normalise features → cosine similarity as risk proxy ----
#         feat = F.normalize(features.float(), dim=1)           # (B, D)
#         bank_feat = F.normalize(bank_features.float(), dim=1) # (N, D)
#         t_cur = event_times.float().view(-1)                  # (B,)
#         c_cur = censorships.float().view(-1)                  # (B,)
#         t_bank = bank_times.float().view(-1)                  # (N,)
#         c_bank = bank_censorships.float().view(-1)            # (N,)

#         # ---- risk scores: (B, N) matrix ----
#         # score[i, j] = dot(feat_i, bank_j); higher → feat_i is closer to
#         # bank samples → treated as higher risk.
#         scores = torch.mm(feat, bank_feat.t())  # (B, N)

#         # ---- boolean flags ----
#         cur_event  = c_cur  < 0.5   # (B,)
#         bank_event = c_bank < 0.5   # (N,)

#         # ---- FIX 2: build *directed* pairs where t_i < t_j ----
#         # Broadcast: t_i (B,1) vs t_j (1,N)
#         t_i = t_cur.unsqueeze(1)    # (B, 1)
#         t_j = t_bank.unsqueeze(0)   # (1, N)
#         delta = t_j - t_i           # (B, N); positive means t_j > t_i

#         # Only keep pairs where t_i < t_j  (strict; ties are ambiguous)
#         directed = delta > 0        # (B, N)

#         # score_i and score_j in pair (i, j):
#         # score_i = scores[i, j]  (current sample i vs bank sample j's features)
#         # But we want score_i to represent the *risk of sample i*.
#         # Convention: score[i, j] measures how much feat_i looks like bank_j.
#         # For a fair ranking loss, we need *one scalar risk per sample*.
#         # We use the per-row mean score as a risk estimate when constructing
#         # the pairwise targets.  More precisely:
#         #   risk_i = mean_j scores[i, ·]   (i is current sample)
#         #   risk_j = scores[i, j]           (j is the specific bank sample)
#         # This is equivalent to: for the pair (cur_i, bank_j), the "score gap"
#         # that the loss operates on is scores[i,j] viewed as score_j and the
#         # batch-level risk of sample i as score_i.
#         #
#         # Simpler and more direct: treat scores[i,j] as the *similarity*
#         # between cur_i and bank_j.  For pair (i, j) with t_i < t_j we want
#         # bank_j to be *less* similar to cur_i than a random reference,
#         # which maps awkwardly.  The cleanest approach: compare scores[i, j]
#         # (sim of i with bank_j) against scores[k, j] for some anchor k.
#         #
#         # ---- Practical simplification used here ----
#         # Define risk score for current sample i as its *negative* mean
#         # similarity to the bank (lower similarity → less typical → higher risk
#         # only if bank is biased toward long survivors; this assumption may not
#         # hold).  To avoid this coupling, we instead define the ranking target
#         # purely in terms of the *pair score gap*:
#         #
#         #   For directed pair (cur_i, bank_j) with t_cur_i < t_bank_j:
#         #     require  scores[i, j]  >  scores[i, j]   ← trivially true for same j
#         #
#         # The correct formulation uses two *different* bank samples as the
#         # "positive" (short survivor) and "negative" (long survivor) anchors.
#         # We implement this as: for each current sample i with t_i < t_j (bank),
#         # we want i to be *more similar* to other bank samples with t ≈ t_i
#         # than to bank sample j.  This requires a second anchor lookup.
#         #
#         # ---- Adopted design (practical for momentum-bank setting) ----
#         # For directed pair (cur_i, bank_j) with t_i < t_j:
#         #   score_high = scores[i, j_short]  where j_short is the closest bank
#         #                sample with t_bank ≤ t_i   (same-risk anchor)
#         #   score_low  = scores[i, j]                (long-survivor bank sample)
#         #   ranking loss: push score_high > score_low
#         #
#         # When no same-risk anchor exists (bank too small), fall back to:
#         #   treat -scores[i, j] as the score of the "wrong" reference and
#         #   +scores[i, j'] (j' closest short bank) as the positive.
#         #
#         # For simplicity and to keep the change minimal relative to the
#         # original structure, we adopt the following clean formulation:
#         #
#         #   For event–event directed pair (cur_i=event, bank_j=event, t_i<t_j):
#         #     loss = pairwise_logistic(scores[i, i_anchor], scores[i, j])
#         #   where i_anchor is the bank sample with smallest |t_bank - t_i|.
#         #
#         # This IS the fix:  it directly tells the network "feat_i should be
#         # closer to its own-time-anchor than to the longer-survivor bank_j".
#         # The original code never expressed this directed preference.

#         # Find per-sample "same-time anchor" in bank: argmin |t_bank - t_cur_i|
#         t_diff_anchor = torch.abs(t_i - t_j)             # (B, N), reuse broadcast
#         anchor_idx = t_diff_anchor.argmin(dim=1)          # (B,)
#         # anchor scores: scores[i, anchor_idx[i]]
#         anchor_scores = scores[torch.arange(feat.shape[0], device=feat.device), anchor_idx]  # (B,)
#         # Expand to (B, N) for broadcasting in pair-wise loss
#         anchor_scores_expanded = anchor_scores.unsqueeze(1).expand_as(scores)  # (B, N)

#         # ---- FIX 1 + 2 combined: directed pairwise ranking loss ----
#         # For each directed pair (i→j, t_i < t_j):
#         #   score_high = anchor_scores[i]  (similarity to same-risk bank sample)
#         #   score_low  = scores[i, j]       (similarity to longer-survivor bank_j)
#         #   Loss pushes score_high > score_low
#         score_high = anchor_scores_expanded   # (B, N)
#         score_low  = scores                    # (B, N)

#         # Per-pair time-gap weights (optional, addresses problem 5 implicitly)
#         pair_w = self._time_pair_weight(delta.clamp(min=0))  # (B, N)

#         # ---- Event–event pairs ----
#         mask_ee = directed & cur_event.unsqueeze(1) & bank_event.unsqueeze(0)  # (B, N)
#         loss_ee = torch.tensor(0.0, device=feat.device)
#         n_ee = int(mask_ee.sum().item())
#         if n_ee > 0:
#             raw_ee = self._pairwise_logistic(score_high[mask_ee], score_low[mask_ee])
#             loss_ee = (raw_ee * pair_w[mask_ee]).mean() * self.event_weight

#         # ---- Event–censored comparable pairs ----
#         # cur_i = event (observed), bank_j = censored, t_i < t_j
#         # t_j is a lower bound → bank_j survived at least t_j > t_i
#         # → i should still rank higher risk than j (soft constraint via hinge)
#         mask_ec = directed & cur_event.unsqueeze(1) & (~bank_event).unsqueeze(0)
#         loss_ec = torch.tensor(0.0, device=feat.device)
#         n_ec = int(mask_ec.sum().item())
#         if n_ec > 0 and self.censored_weight > 0:
#             raw_ec = self._hinge_ranking(score_high[mask_ec], score_low[mask_ec], self.margin)
#             loss_ec = (raw_ec * pair_w[mask_ec]).mean() * self.censored_weight

#         # ---- Aggregate ----
#         total_pairs = n_ee + n_ec
#         self.last_stats = {
#             "event_event_pairs":    n_ee,
#             "event_censored_pairs": n_ec,
#             "total_pairs":          total_pairs,
#         }

#         if total_pairs < self.min_pairs:
#             return features.new_tensor(0.0)

#         # Weighted sum; normalise by total weight to keep scale stable
#         total_w = self.event_weight * (n_ee > 0) + self.censored_weight * (n_ec > 0)
#         total_w = max(total_w, 1e-6)
#         loss = (loss_ee + loss_ec) / total_w

#         return loss


def concordance_index(event_times, risk_scores, censorships):
    times = np.asarray(event_times, dtype=float).reshape(-1)
    risks = np.asarray(risk_scores, dtype=float).reshape(-1)
    censorships = np.asarray(censorships, dtype=float).reshape(-1)
    events = censorships < 1
    valid = np.isfinite(times) & np.isfinite(risks) & np.isfinite(censorships)
    times, risks, events = times[valid], risks[valid], events[valid]

    concordant = 0.0
    comparable = 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if np.isclose(times[i], times[j]):
                continue
            if times[i] < times[j] and events[i]:
                comparable += 1
                concordant += 1 if risks[i] > risks[j] else 0.5 if np.isclose(risks[i], risks[j]) else 0
            elif times[j] < times[i] and events[j]:
                comparable += 1
                concordant += 1 if risks[j] > risks[i] else 0.5 if np.isclose(risks[i], risks[j]) else 0
    return float(concordant / comparable) if comparable > 0 else float("nan")


def _survival_time_grid_from_loader(loader):
    dataset = getattr(loader, "dataset", None)
    bins = getattr(dataset, "bins", None)
    if bins is None:
        return None
    bins = np.asarray(bins, dtype=float).reshape(-1)
    if len(bins) <= 1:
        return None
    return bins[1:]


def _prepare_survival_probability_inputs(event_times, survival_probabilities, censorships, time_grid=None):
    if survival_probabilities is None:
        return None
    times = np.asarray(event_times, dtype=float).reshape(-1)
    censoring = np.asarray(censorships, dtype=float).reshape(-1)
    curves = np.asarray(survival_probabilities, dtype=float)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)
    if curves.ndim != 2:
        raise ValueError("Survival probabilities must be a 2D array with shape [n_samples, n_times].")
    if not (len(times) == len(censoring) == curves.shape[0]):
        raise ValueError("IBS/INBLL require event_times, censorships, and survival curves with matching sample counts.")

    num_times = curves.shape[1]
    if time_grid is None:
        grid = np.arange(1, num_times + 1, dtype=float)
    else:
        grid = np.asarray(time_grid, dtype=float).reshape(-1)
        if len(grid) == num_times + 1:
            grid = grid[1:]
    if len(grid) != num_times:
        raise ValueError(f"Time grid length {len(grid)} does not match survival curve width {num_times}.")

    valid = np.isfinite(times) & np.isfinite(censoring) & np.isfinite(curves).all(axis=1)
    times, censoring, curves = times[valid], censoring[valid], curves[valid]
    if len(times) == 0 or num_times == 0:
        return None

    order = np.argsort(grid)
    grid = grid[order]
    curves = np.clip(curves[:, order], 1e-7, 1.0 - 1e-7)
    return times, curves, censoring, grid


def _kaplan_meier_censoring_survival(times, censorships, eps=1e-7):
    times = np.asarray(times, dtype=float).reshape(-1)
    censoring = np.asarray(censorships, dtype=float).reshape(-1)
    censor_events = censoring >= 1.0
    unique_times = np.sort(np.unique(times[np.isfinite(times)]))
    km_times = []
    km_survival = []
    survival = 1.0
    for time_point in unique_times:
        at_risk = np.sum(times >= time_point)
        if at_risk <= 0:
            continue
        censor_count = np.sum((times == time_point) & censor_events)
        survival *= max(0.0, 1.0 - float(censor_count) / float(at_risk))
        km_times.append(float(time_point))
        km_survival.append(float(survival))
    km_times = np.asarray(km_times, dtype=float)
    km_survival = np.asarray(km_survival, dtype=float)

    def predict(query_times, left=False):
        query = np.asarray(query_times, dtype=float)
        if len(km_times) == 0:
            return np.ones_like(query, dtype=float)
        side = "left" if left else "right"
        indices = np.searchsorted(km_times, query, side=side) - 1
        values = np.ones_like(query, dtype=float)
        valid = indices >= 0
        values[valid] = km_survival[indices[valid]]
        return np.clip(values, eps, 1.0)

    return predict


def _integrate_over_grid(values, time_grid):
    values = np.asarray(values, dtype=float).reshape(-1)
    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    finite = np.isfinite(values) & np.isfinite(time_grid)
    values, time_grid = values[finite], time_grid[finite]
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    duration = float(time_grid[-1] - time_grid[0])
    if duration <= 0:
        return float(np.mean(values))
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(values, time_grid)
    else:
        area = np.trapz(values, time_grid)
    return float(area / duration)


def survival_probability_metrics(event_times, survival_probabilities, censorships, time_grid=None, eps=1e-7):
    """IPCW Integrated Brier Score and Integrated Negative Binomial Log-Likelihood.

    `survival_probabilities[:, j]` is interpreted as S(t_j | x), where t_j is
    supplied by `time_grid`. Censorship is encoded as 1 for censored, 0 for an
    observed event, matching the rest of the survival pipeline.
    """

    prepared = _prepare_survival_probability_inputs(event_times, survival_probabilities, censorships, time_grid)
    if prepared is None:
        return {"ibs": float("nan"), "inbll": float("nan")}
    times, curves, censoring, grid = prepared
    observed_event = censoring < 1.0
    censor_survival = _kaplan_meier_censoring_survival(times, censoring, eps=eps)

    brier_scores = []
    nbll_scores = []
    for idx, time_point in enumerate(grid):
        survival_at_t = np.clip(curves[:, idx], eps, 1.0 - eps)
        survived_past_t = times > time_point
        event_by_t = (times <= time_point) & observed_event

        g_t = float(censor_survival(np.asarray([time_point]), left=False)[0])
        survived_weight = survived_past_t.astype(float) / max(g_t, eps)
        event_weights = np.zeros_like(times, dtype=float)
        if np.any(event_by_t):
            event_weights[event_by_t] = 1.0 / censor_survival(times[event_by_t], left=True)

        brier = np.mean(
            survived_weight * np.square(1.0 - survival_at_t)
            + event_weights * np.square(survival_at_t)
        )
        nbll = -np.mean(
            survived_weight * np.log(survival_at_t)
            + event_weights * np.log1p(-survival_at_t)
        )
        brier_scores.append(float(brier))
        nbll_scores.append(float(nbll))

    return {
        "ibs": _integrate_over_grid(brier_scores, grid),
        "inbll": _integrate_over_grid(nbll_scores, grid),
    }


def survival_metrics(event_times, risk_scores, censorships, survival_probabilities=None, time_grid=None):
    metrics = {"c_index": concordance_index(event_times, risk_scores, censorships)}
    if survival_probabilities is not None:
        metrics.update(survival_probability_metrics(event_times, survival_probabilities, censorships, time_grid=time_grid))
    return metrics


def survival_metrics_by_cancer(event_times, risk_scores, censorships, cancer_types):
    times = np.asarray(event_times, dtype=float).reshape(-1)
    risks = np.asarray(risk_scores, dtype=float).reshape(-1)
    censoring = np.asarray(censorships, dtype=float).reshape(-1)
    cancers = np.asarray(cancer_types, dtype=object).reshape(-1)
    if not (len(times) == len(risks) == len(censoring) == len(cancers)):
        raise ValueError("Cancer-stratified survival metrics require equally sized inputs.")

    metrics = {}
    cancer_c_indices = []
    valid_cancers = sorted(
        {
            str(cancer)
            for cancer in cancers
            if cancer not in [None, "", "null"] and not pd.isna(cancer)
        }
    )
    if not valid_cancers:
        return metrics
    for cancer in valid_cancers:
        mask = np.asarray([str(value) == cancer for value in cancers], dtype=bool)
        cancer_times = times[mask]
        cancer_risks = risks[mask]
        cancer_censoring = censoring[mask]
        c_index = concordance_index(cancer_times, cancer_risks, cancer_censoring)
        metrics[f"cancer_{cancer}_c_index"] = c_index
        metrics[f"cancer_{cancer}_n"] = int(mask.sum())
        metrics[f"cancer_{cancer}_events"] = int(np.sum(cancer_censoring < 1))
        if np.isfinite(c_index):
            cancer_c_indices.append(c_index)
    metrics["cancer_macro_c_index"] = float(np.mean(cancer_c_indices)) if cancer_c_indices else float("nan")
    return metrics


def survival_probability_metrics_by_cancer(event_times, survival_probabilities, censorships, cancer_types, time_grid=None):
    times = np.asarray(event_times, dtype=float).reshape(-1)
    censoring = np.asarray(censorships, dtype=float).reshape(-1)
    cancers = np.asarray(cancer_types, dtype=object).reshape(-1)
    curves = np.asarray(survival_probabilities, dtype=float)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)
    if not (len(times) == len(censoring) == len(cancers) == curves.shape[0]):
        raise ValueError("Cancer-stratified IBS/INBLL require equally sized inputs.")

    metrics = {}
    cancer_ibs = []
    cancer_inbll = []
    valid_cancers = sorted(
        {
            str(cancer)
            for cancer in cancers
            if cancer not in [None, "", "null"] and not pd.isna(cancer)
        }
    )
    if not valid_cancers:
        return metrics
    for cancer in valid_cancers:
        mask = np.asarray([str(value) == cancer for value in cancers], dtype=bool)
        cancer_metrics = survival_probability_metrics(
            times[mask],
            curves[mask],
            censoring[mask],
            time_grid=time_grid,
        )
        metrics[f"cancer_{cancer}_ibs"] = cancer_metrics["ibs"]
        metrics[f"cancer_{cancer}_inbll"] = cancer_metrics["inbll"]
        if np.isfinite(cancer_metrics["ibs"]):
            cancer_ibs.append(cancer_metrics["ibs"])
        if np.isfinite(cancer_metrics["inbll"]):
            cancer_inbll.append(cancer_metrics["inbll"])
    metrics["cancer_macro_ibs"] = float(np.mean(cancer_ibs)) if cancer_ibs else float("nan")
    metrics["cancer_macro_inbll"] = float(np.mean(cancer_inbll)) if cancer_inbll else float("nan")
    return metrics


def gate_distribution_metrics_by_cancer(gate_local_values, cancer_types, high_weight_threshold=0.7):
    gates = np.asarray(gate_local_values, dtype=float).reshape(-1)
    cancers = np.asarray(cancer_types, dtype=object).reshape(-1)
    if len(gates) != len(cancers):
        raise ValueError("Cancer-stratified gate metrics require equally sized inputs.")
    if len(gates) == 0:
        return {}
    threshold = float(high_weight_threshold)
    if threshold <= 0.5 or threshold > 1.0:
        raise ValueError("High gate-weight threshold must be in (0.5, 1.0].")

    valid_cancers = sorted(
        {
            str(cancer)
            for cancer in cancers
            if cancer not in [None, "", "null"] and not pd.isna(cancer)
        }
    )
    groups = [("ALL", np.ones(len(gates), dtype=bool))]
    groups.extend(
        (cancer, np.asarray([str(value) == cancer for value in cancers], dtype=bool))
        for cancer in valid_cancers
    )

    metrics = {}
    wikg_threshold = 1.0 - threshold
    for cancer, mask in groups:
        values = gates[mask]
        if len(values) == 0:
            continue
        prefix = f"cancer_{cancer}_gate_local"
        metrics[f"{prefix}_n"] = int(len(values))
        metrics[f"{prefix}_mean"] = float(np.mean(values))
        metrics[f"{prefix}_std"] = float(np.std(values))
        for quantile, name in ((0.10, "q10"), (0.25, "q25"), (0.50, "q50"), (0.75, "q75"), (0.90, "q90")):
            metrics[f"{prefix}_{name}"] = float(np.quantile(values, quantile))
        metrics[f"{prefix}_high_ab_ratio"] = float(np.mean(values >= threshold))
        metrics[f"{prefix}_high_wikg_ratio"] = float(np.mean(values <= wikg_threshold))
        metrics[f"{prefix}_soft_mixed_ratio"] = float(np.mean((values > wikg_threshold) & (values < threshold)))
    metrics["cancer_gate_high_weight_threshold"] = threshold
    return metrics


def survival_train_loop(
    device,
    model,
    loader,
    loss_fn,
    optimizer,
    scheduler=None,
    grad_accumulation=1,
    contrastive_loss_fn=None,
    contrastive_bank=None,
    contrastive_weight=0.0,
    gate_balance_queue=None,
    gate_balance_weight=0.0,
    evidential_loss_fn=None,
    survival_loss_weight=1.0,
    epoch=None,
):
    start = time.time()
    model.train()
    optimizer.zero_grad()
    loss_log = 0.0
    surv_loss_log = 0.0
    evidential_loss_log = 0.0
    evidential_steps = 0
    contrast_loss_log = 0.0
    gate_balance_loss_log = 0.0
    gate_load_imbalance_log = 0.0
    gate_balance_steps = 0
    contrast_stats_log = {}
    model_metrics_log = {}
    risks, event_times, censorships, cancer_types, survival_curves = [], [], [], [], []

    for batch_idx, batch in enumerate(loader):
        bag, label, event_time, c, metadata = unpack_survival_batch(batch)
        bag = bag.to(device).float()
        label = label.to(device).long()
        c = c.to(device).float()
        batch_cancer_types = metadata_cancer_types(metadata, label.numel()) if metadata is not None else None
        if contrastive_loss_fn is not None and contrastive_weight > 0:
            output = forward_with_wsi_feature(model, bag)
        else:
            output = model(bag)
        logits = extract_logits(output)
        hazards, survival = survival_from_logits(logits)
        surv_loss = loss_fn(hazards=hazards, survival=survival, y=label, c=c)
        aux_loss = extract_auxiliary_loss(output, logits)
        evidential_loss = bag.new_tensor(0.0)
        if evidential_loss_fn is not None:
            evidential_loss = evidential_loss_fn(output, label, c, epoch=epoch)
            evidential_loss_log += float(evidential_loss.detach().cpu())
            evidential_steps += 1
            for key, value in getattr(evidential_loss_fn, "last_stats", {}).items():
                model_metrics_log[key] = model_metrics_log.get(key, 0.0) + value
        contrast_loss = bag.new_tensor(0.0)
        if contrastive_loss_fn is not None and contrastive_bank is not None and contrastive_weight > 0:
            feature = extract_wsi_feature(output, logits)
            contrast_loss = contrastive_loss_fn(
                feature,
                event_time.to(device),
                c,
                contrastive_bank,
                cancer_types=batch_cancer_types,
            )
            for key, value in getattr(contrastive_loss_fn, "last_stats", {}).items():
                contrast_stats_log[key] = contrast_stats_log.get(key, 0) + value
        gate_balance_loss = bag.new_tensor(0.0)
        gate_weights = output.get("gate_weights") if isinstance(output, dict) else None
        gate_is_warmup = output.get("gate_is_warmup") if isinstance(output, dict) else None
        gate_balance_active = (
            gate_balance_queue is not None
            and gate_balance_weight > 0
            and torch.is_tensor(gate_weights)
            and not (torch.is_tensor(gate_is_warmup) and bool(gate_is_warmup.detach().item()))
        )
        if gate_balance_active:
            gate_balance_loss = gate_balance_queue.loss(gate_weights)
            gate_balance_loss_log += float(gate_balance_loss.detach().cpu())
            gate_load_imbalance_log += gate_balance_queue.imbalance(gate_weights)
            gate_balance_steps += 1
        weighted_surv_loss = float(survival_loss_weight) * surv_loss
        loss = (
            weighted_surv_loss
            + evidential_loss
            + contrastive_weight * contrast_loss
            + aux_loss
            + gate_balance_weight * gate_balance_loss
        )
        loss_log += loss.item()
        surv_loss_log += surv_loss.item()
        contrast_loss_log += contrast_loss.item()
        for key, value in extract_model_metrics(output).items():
            model_metrics_log[key] = model_metrics_log.get(key, 0.0) + value

        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        risks.extend(risk.tolist())
        survival_curves.extend(survival.detach().cpu().numpy().tolist())
        event_times.extend(event_time.detach().cpu().numpy().reshape(-1).tolist())
        censorships.extend(c.detach().cpu().numpy().reshape(-1).tolist())
        if batch_cancer_types is not None:
            cancer_types.extend(batch_cancer_types)

        (loss / grad_accumulation).backward()
        if (batch_idx + 1) % grad_accumulation == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        if contrastive_bank is not None and contrastive_weight > 0:
            feature = extract_wsi_feature(output, logits)
            contrastive_bank.enqueue(feature, event_time, c.detach().cpu(), cancer_types=batch_cancer_types)
        if gate_balance_active:
            gate_balance_queue.enqueue(gate_weights)

    if scheduler is not None:
        scheduler.step()
    loss_log /= max(len(loader), 1)
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
    metrics["surv_loss_weight"] = float(survival_loss_weight)
    metrics["evidential_loss"] = evidential_loss_log / max(evidential_steps, 1)
    metrics["contrast_loss"] = contrast_loss_log / max(len(loader), 1)
    metrics["contrast_weight"] = contrastive_weight
    metrics["gate_balance_loss"] = gate_balance_loss_log / max(gate_balance_steps, 1)
    metrics["gate_balance_weight"] = gate_balance_weight
    metrics["gate_load_imbalance"] = gate_load_imbalance_log / max(gate_balance_steps, 1)
    for key, value in contrast_stats_log.items():
        metrics[f"contrast_{key}"] = value
    for key, value in model_metrics_log.items():
        metrics[key] = value / max(len(loader), 1)
    return loss_log, time.time() - start, metrics


def survival_eval_loop(
    device,
    model,
    loader,
    loss_fn=None,
    alpha=0.0,
    return_results=False,
    gate_high_weight_threshold=0.7,
    evidential_loss_fn=None,
    survival_loss_weight=1.0,
):
    model.eval()
    loss_log = 0.0
    evidential_loss_log = 0.0
    evidential_steps = 0
    risks, event_times, censorships, labels, cancer_types, gate_local_values, survival_curves = [], [], [], [], [], [], []
    model_metrics_log = {}
    results = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            bag, label, event_time, c, metadata = unpack_survival_batch(batch)
            bag = bag.to(device).float()
            label = label.to(device).long()
            c = c.to(device).float()
            output = model(bag)
            logits = extract_logits(output)
            hazards, survival = survival_from_logits(logits)
            aux_loss = extract_auxiliary_loss(output, logits)
            evidential_loss = bag.new_tensor(0.0)
            if evidential_loss_fn is not None:
                evidential_loss = evidential_loss_fn(output, label, c)
                evidential_loss_log += float(evidential_loss.detach().cpu())
                evidential_steps += 1
                for key, value in getattr(evidential_loss_fn, "last_stats", {}).items():
                    model_metrics_log[key] = model_metrics_log.get(key, 0.0) + value
            surv_loss = bag.new_tensor(0.0)
            if loss_fn is not None:
                surv_loss = loss_fn(hazards=hazards, survival=survival, y=label, c=c, alpha=alpha)
                loss = float(survival_loss_weight) * surv_loss + evidential_loss + aux_loss
                loss_log += loss.item()
                model_metrics_log["surv_loss"] = model_metrics_log.get("surv_loss", 0.0) + float(surv_loss.detach().cpu())
            for key, value in extract_model_metrics(output).items():
                model_metrics_log[key] = model_metrics_log.get(key, 0.0) + value

            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            risks.extend(risk.tolist())
            survival_curves.extend(survival.detach().cpu().numpy().tolist())
            event_times.extend(event_time.detach().cpu().numpy().reshape(-1).tolist())
            censorships.extend(c.detach().cpu().numpy().reshape(-1).tolist())
            labels.extend(label.detach().cpu().numpy().reshape(-1).tolist())
            sample = loader.dataset.samples[batch_idx] if hasattr(loader.dataset, "samples") else {}
            if metadata is not None:
                batch_cancer_types = metadata_cancer_types(metadata, label.numel())
                cancer_types.extend(batch_cancer_types)
            else:
                cancer_types.append(sample.get("cancer_type"))
            if isinstance(output, dict) and torch.is_tensor(output.get("gate_weights")):
                gate_local_values.extend(output["gate_weights"].detach().cpu().reshape(-1, 2)[:, 0].tolist())

            if return_results:
                results[sample["sample_id"]] = {
                    "sample_id": sample["sample_id"],
                    "cancer_type": sample.get("cancer_type"),
                    "risk": float(risk[0]),
                    "disc_label": int(labels[-1]),
                    "event_time": float(event_times[-1]),
                    "censorship": float(censorships[-1]),
                    "survival_probability": survival.detach().cpu().reshape(-1, survival.shape[-1])[0].tolist(),
                }
                if isinstance(output, dict) and torch.is_tensor(output.get("gate_weights")):
                    gate_weights = output["gate_weights"].detach().cpu().reshape(-1, 2)[0]
                    results[sample["sample_id"]]["routing"] = {
                        "local": float(gate_weights[0]),
                        "wikg": float(gate_weights[1]),
                        "dse_entropy": float(output["dse_entropy"].detach().cpu()),
                        "uses_physical_coords": bool(output["uses_physical_coords"].detach().cpu().item()),
                    }
                if isinstance(output, dict) and torch.is_tensor(output.get("fused_uncertainty")):
                    def _first_row_list(key):
                        value = output.get(key)
                        if not torch.is_tensor(value):
                            return None
                        return value.detach().cpu().reshape(-1, value.shape[-1])[0].tolist()

                    def _mean_scalar(key):
                        value = output.get(key)
                        if not torch.is_tensor(value):
                            return None
                        return float(value.detach().cpu().mean())

                    results[sample["sample_id"]]["evidential"] = {
                        "local_probability": _first_row_list("local_probability"),
                        "wikg_probability": _first_row_list("wikg_probability"),
                        "fused_probability": _first_row_list("fused_probability"),
                        "discounted_local_probability": _first_row_list("discounted_local_probability"),
                        "discounted_wikg_probability": _first_row_list("discounted_wikg_probability"),
                        "fused_uncertainty": _mean_scalar("fused_uncertainty"),
                        "local_uncertainty": _mean_scalar("local_uncertainty"),
                        "wikg_uncertainty": _mean_scalar("wikg_uncertainty"),
                        "discounted_local_uncertainty": _mean_scalar("discounted_local_uncertainty"),
                        "discounted_wikg_uncertainty": _mean_scalar("discounted_wikg_uncertainty"),
                        "local_evidence_strength": _mean_scalar("local_evidence_strength"),
                        "wikg_evidence_strength": _mean_scalar("wikg_evidence_strength"),
                        "discounted_local_evidence_strength": _mean_scalar("discounted_local_evidence_strength"),
                        "discounted_wikg_evidence_strength": _mean_scalar("discounted_wikg_evidence_strength"),
                        "fused_evidence_strength": _mean_scalar("fused_evidence_strength"),
                        "expert_disagreement": _mean_scalar("expert_disagreement"),
                    }

    if loss_fn is not None and len(loader) > 0:
        loss_log /= len(loader)
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
    metrics["surv_loss_weight"] = float(survival_loss_weight)
    metrics["evidential_loss"] = evidential_loss_log / max(evidential_steps, 1)
    if len(gate_local_values) == len(cancer_types):
        metrics.update(
            gate_distribution_metrics_by_cancer(
                gate_local_values,
                cancer_types,
                high_weight_threshold=gate_high_weight_threshold,
            )
        )
    for key, value in model_metrics_log.items():
        metrics[key] = value / max(len(loader), 1)
    if return_results:
        return loss_log, metrics, results
    return loss_log, metrics
