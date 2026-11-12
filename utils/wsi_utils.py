import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_h5_features_and_coords(slide_path, require_coords=False):
    if not str(slide_path).lower().endswith(".h5"):
        raise ValueError(f"Direct H5 input is required, got: {slide_path}")
    with h5py.File(slide_path, "r") as h5_file:
        if "features" not in h5_file:
            raise ValueError(f"H5 input must contain 'features': {slide_path}")
        feat = torch.from_numpy(np.array(h5_file["features"]))
        coords = torch.from_numpy(np.array(h5_file["coords"])) if "coords" in h5_file else None
    if require_coords and coords is None:
        raise ValueError(f"H5 input must contain 'coords': {slide_path}")
    if feat.dim() == 3 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    if coords is not None and coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)
    if coords is not None and feat.shape[0] != coords.shape[0]:
        raise ValueError(
            f"H5 feature/coordinate shape mismatch: features={tuple(feat.shape)}, "
            f"coords={tuple(coords.shape)}, path={slide_path}"
        )
    return feat, coords


class WSI_Dataset(Dataset):
    def __init__(self, dataset_info_csv_path, group):
        assert group in ["train", "val", "test"], "group must be in [train,val,test]"
        self.dataset_info_csv_path = dataset_info_csv_path
        self.dataset_df = pd.read_csv(self.dataset_info_csv_path)
        self.slide_path_list = self.dataset_df[group + "_slide_path"].dropna().to_list()
        self.labels_list = self.dataset_df[group + "_label"].dropna().to_list()

    def __len__(self):
        return len(self.slide_path_list)

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        label = torch.tensor(int(self.labels_list[idx]))

        if str(slide_path).lower().endswith(".h5"):
            feat, _ = _load_h5_features_and_coords(slide_path)
        else:
            feat = torch.load(slide_path)
            if isinstance(feat, dict):
                if "feats" in feat:
                    feat = feat["feats"]
                elif "features" in feat:
                    feat = feat["features"]
                else:
                    raise ValueError(f"Unknown dict format in {slide_path}, keys: {list(feat.keys())}")
        if feat.dim() == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        return feat, label

    def is_None_Dataset(self):
        return self.__len__() == 0

    def is_with_labels(self):
        return len(self.labels_list) != 0

    def get_balanced_sampler(self, replacement=True):
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler

        label_counts = Counter(self.labels_list)
        weights = [1.0 / label_counts[label] for label in self.labels_list]
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=replacement)


class WSI_Coord_Dataset(WSI_Dataset):
    """Spatial MIL dataset reading coordinates from the feature file itself."""

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        label = torch.tensor(int(self.labels_list[idx]))
        coords = None

        if str(slide_path).lower().endswith(".h5"):
            feat, coords = _load_h5_features_and_coords(slide_path)
        else:
            loaded = torch.load(slide_path)
            if isinstance(loaded, dict):
                if "feats" in loaded:
                    feat = loaded["feats"]
                elif "features" in loaded:
                    feat = loaded["features"]
                else:
                    raise ValueError(f"Unknown dict format in {slide_path}, keys: {list(loaded.keys())}")
                coords = loaded.get("coords")
            else:
                feat = loaded
            if feat.dim() == 3 and feat.shape[0] == 1:
                feat = feat.squeeze(0)
            if coords is not None and coords.dim() == 3 and coords.shape[0] == 1:
                coords = coords.squeeze(0)

        if coords is not None:
            coords = coords.to(feat.device).float()
            if coords.shape[0] == feat.shape[0] and coords.shape[-1] >= 2:
                feat = torch.cat([feat, coords[:, :2]], dim=-1)
        return feat, label


class CDP_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self, dataset_info_csv_path, BeyesGuassian_pt_dir, group):
        super().__init__(dataset_info_csv_path, group)
        self.slide_path_list = [
            os.path.join(
                BeyesGuassian_pt_dir,
                os.path.basename(slide_path).replace(".pt", "_bayesian_gaussian.pt"),
            )
            for slide_path in self.slide_path_list
        ]


class LONG_MIL_WSI_Dataset(WSI_Dataset):
    """LONG_MIL dataset requiring direct H5 paths with features and coords."""

    def __getitem__(self, idx):
        feat, coords = _load_h5_features_and_coords(self.slide_path_list[idx], require_coords=True)
        return torch.cat([feat, coords.float()], dim=-1), torch.tensor(int(self.labels_list[idx]))


class SC_MIL_WSI_Dataset(WSI_Dataset):
    """SC_MIL dataset requiring direct H5 paths with features and coords."""

    def __getitem__(self, idx):
        feat, coords = _load_h5_features_and_coords(self.slide_path_list[idx], require_coords=True)
        return torch.cat([feat, coords.float()], dim=-1), torch.tensor(int(self.labels_list[idx]))
