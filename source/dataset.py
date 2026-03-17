import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path
from torchvision import transforms
from typing import Dict, Optional
from collections import defaultdict
from omegaconf import DictConfig

from source.dataset_utils import(
    RandomZeroing,
    GaussianNoise,
    RandomScaling,
    RandomCrop,
)

class ExtractedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
        label_mapping: Dict[int, int] = {},
        label_encoding: Optional[str] = None,
        slide_pos_embed: Optional[DictConfig] = None,
    ):
        self.features_dir = features_dir
        self.label_name = label_name
        self.label_mapping = label_mapping
        self.label_encoding = label_encoding
        self.slide_pos_embed = slide_pos_embed

        self.df = self.prepare_data(df)
        self.num_classes = len(self.df.label.value_counts(dropna=True))
        self.map_class_to_slide_ids()

    def prepare_data(self, df):
        if self.label_mapping:
            df["label"] = df[self.label_name].apply(lambda x: self.label_mapping[x])
        elif self.label_name != "label":
            df["label"] = df[self.label_name]
        filtered_slide_ids = []
        for slide_id in df.slide_id:
            if Path(self.features_dir, f"{slide_id}_features.pt").is_file():
                filtered_slide_ids.append(slide_id)
        df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        return df_filtered

    def map_class_to_slide_ids(self):
        # map each class to corresponding slide ids
        self.class_2_id = defaultdict(list)
        for i in range(self.num_classes):
            self.class_2_id[i] = np.asarray(self.df.label == i).nonzero()[0]

    def get_slide_id(self, idx):
        return self.df.loc[idx].slide_id
    
    def get_label(self, idx):
        return self.df.loc[idx].label

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        
        fp = Path(self.features_dir, f"{slide_id}_features.pt")
        # raw_features = torch.load(fp)
        raw_features = torch.load(fp, weights_only=False)
        
        # Unpack custom dictionary and stack the high-resolution features 
        # into the [N, 2304] tensor the transformer expects
        features = torch.stack([patch['hr_feature'] for patch in raw_features])

        label = row.label
        if self.label_encoding == "ordinal":
            label = [1] * (label + 1) + [0] * (self.num_classes - label - 1)

        return idx, slide_id, features, label

    def __len__(self):
        return len(self.df)

class PretrainFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        augmentation: str,
    ):
        self.features_dir = features_dir

        self.df = self.prepare_data(df)

        self.augmentation = augmentation
        
    def prepare_data(self, df):
        filtered_slide_ids = []
        for slide_id in df.slide_id:
            if Path(self.features_dir, f"{slide_id}_features.pt").is_file():
                filtered_slide_ids.append(slide_id)

        df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        print(f"Total training WSIs: {len(df_filtered)}")
        return df_filtered

    def get_slide_id(self, idx):
        return self.df.loc[idx].slide_id
    
    def get_label(self, idx):
        return self.df.loc[idx].label

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        fp = Path(self.features_dir, f"{slide_id}_features.pt")
        
        # 1. Instantly load the pure tensor (safe and fast!)
        fast_data = torch.load(fp, weights_only=True)

        # 2. Extract the pre-concatenated HR+LR features
        features = fast_data['features']

        # --- ADDED THIS TO PREVENT SEQUENCE LENGTH CRASHES AND GPU OOM ---
        max_patches = 4000
        if features.shape[0] > max_patches:
            # Randomly sample max_patches without replacement
            indices = torch.randperm(features.shape[0])[:max_patches]
            features = features[indices]

        if self.augmentation == "slide_aug":
            
            def fast_vectorized_augment(x, zero_prob=0.5, noise_prob=1.0, scale_prob=0.5, crop_prob=0.5):
                x_aug = x.clone()
                
                # 1. Random Zeroing
                if torch.rand(1).item() < zero_prob:
                    mask = torch.rand_like(x_aug) > 0.5
                    x_aug = x_aug * mask
                    
                # 2. Gaussian Noise
                if torch.rand(1).item() < noise_prob:
                    x_aug = x_aug + torch.randn_like(x_aug) * 0.1
                    
                # 3. Random Scaling
                if torch.rand(1).item() < scale_prob:
                    scale = torch.empty(1).uniform_(0.9, 1.1).item()
                    x_aug = x_aug * scale
                    
                # 4. Random Crop (Drops 30% of the sequence)
                if torch.rand(1).item() < crop_prob:
                    seq_len = x_aug.shape[0]
                    crop_len = int(seq_len * 0.7)
                    start_idx = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
                    x_aug = x_aug[start_idx : start_idx + crop_len]
                    
                return x_aug

            # features1 gets standard augmentation (100% noise chance based on original config)
            features1 = fast_vectorized_augment(features, zero_prob=0.5, noise_prob=1.0, scale_prob=0.5, crop_prob=0.5)
            
            # features2 gets prime augmentation (10% noise chance based on original config)
            features2 = fast_vectorized_augment(features, zero_prob=0.5, noise_prob=0.1, scale_prob=0.5, crop_prob=0.5)
        
        elif self.augmentation == "random_qtr":
            # apply random quarter from overlap 0.5 (insufficient number of regions if it was taken from overlap 0)
            r1_indices = torch.randperm(features.shape[0])[:round(features.shape[0]/4)]
            r2_indices = torch.randperm(features.shape[0])[:round(features.shape[0]/4)]
            
            # take random quarter from features
            features1 = features[r1_indices]
            features2 = features[r2_indices]

        return slide_id, features1, features2

    def __len__(self):
        return len(self.df)

class RegionFilepathsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        region_root_dir: Path,
        region_size: int,
        fmt: str,
        is_quarter_data: str,

    ):
        self.df = df
        self.region_root_dir = region_root_dir
        self.region_size = region_size
        self.format = fmt
        self.is_quarter_data = is_quarter_data

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        region_dir = Path(
            self.region_root_dir, slide_id, str(self.region_size), self.format
        )
        regions = [str(fp) for fp in region_dir.glob(f"*.{self.format}")]
        if self.is_quarter_data:
            num_regions = len(regions)
            num_selected_regions = round(num_regions * 0.25)
            regions = random.sample(regions, num_selected_regions)
        
        return idx, regions

    def __len__(self):
        return len(self.df)
