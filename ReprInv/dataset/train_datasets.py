import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .image_datasets_segmentation import SegLocImageDataset, SamImageDataset, DptImageDataset
from .image_datasets_sparse_descriptors import (
    SparseDescriptorImageDataset,
    NinjadescImageDataset,
    _collect_image_paths,
)

# Supported training conditions, ie. a dataset extraction class is available for the condition
# Channel count each condition feeds into the UNet's conv_in, i.e. add_lora_layers(input_dims=...).
CONDITION_INPUT_DIMS = {
    "SIFT": 128,
    "SUPERPOINT": 256,
    "XFEAT": 64,
    "ALIKED": 128,
    "LOMA": 256,
    "SIFT_ninjadesc": 128,
    "segloc": 3,
    "sam": 3,
    "adek": 3,
}

_MAX_SAMPLE_RETRIES = 10
SEGMENTATION_CONDITIONS = {"segloc", "sam", "adek"}
MODEL_BACKED_CONDITIONS = {"segloc", "sam", "adek", "SIFT_ninjadesc", "SIFT", "SUPERPOINT", "XFEAT", "ALIKED"}


def condition_input_dims(conditions) -> list:
    unknown = [c for c in conditions if c not in CONDITION_INPUT_DIMS]
    if unknown:
        raise ValueError(
            f"Unknown condition(s) {unknown}. Known: {sorted(CONDITION_INPUT_DIMS)}."
        )
    return [CONDITION_INPUT_DIMS[c] for c in conditions]


def _fixed_size_positions(positions: torch.Tensor, num_positions: int) -> torch.Tensor:
    ### Pad positions to a constant (num_positions, 2) for batched training.
    out = torch.full((num_positions, 2), -1.0, dtype=torch.float32)
    if positions is None or positions.numel() == 0:
        return out
    positions = positions.to(dtype=torch.float32)
    n = positions.shape[0]
    if n > num_positions:
        keep = torch.randperm(n)[:num_positions]
        out.copy_(positions[keep])
    else:
        out[:n] = positions
    return out


class _ConditionExtractor:
    def __init__(self, condition, resolution, downsample_factor, device,
                 noise_std_dev=0.0, aggregate=False, base_desc="sift", num_positions=2048, loma_features=()):
        self.condition = condition
        self.num_positions = num_positions
        common = dict(resolution=resolution, downsample_factor=downsample_factor)

        if condition == "segloc":
            self._ds = SegLocImageDataset([], condition, device=device, **common)
        elif condition == "sam":
            self._ds = SamImageDataset([], condition, device=device, **common)
        elif condition == "adek":
            self._ds = DptImageDataset([], condition, device=device, **common)
        elif condition == "SIFT_ninjadesc":
            self._ds = NinjadescImageDataset([], condition, aggregate=aggregate, device=device,
                                             base_desc=base_desc, **common)
        else:
            # ALIKED runs in the training process (dataloader_num_workers=0), so it can use the GPU.
            self._ds = SparseDescriptorImageDataset([], condition, aggregate=aggregate, aliked_device=device,
                                                    loma_features=loma_features, **common)

        if not self.is_segmentation:
            self._ds.noise_std_dev = noise_std_dev
            self._ds.num_keypoints_training_mid = num_positions

    @property
    def is_segmentation(self) -> bool:
        return self.condition in SEGMENTATION_CONDITIONS

    def __call__(self, image: Image.Image, training: bool, path=None):
        if self.is_segmentation:
            if self.condition == "segloc":
                seg = self._ds._extract_segloc(image)
            elif self.condition == "sam":
                seg = self._ds._extract_sam(image)
            else:
                seg = self._ds._extract_dpt(image)
            conditioning = self._ds.image_transforms_mask(seg)
            positions = self._ds.tensor_centroids_from_mask(seg, max_centroids=self.num_positions).to(dtype=torch.float32)
            return conditioning, positions

        if self.condition == "SIFT_ninjadesc":
            keypoints, descriptors = self._ds.extract_NINJADESC(image)
        else:
            keypoints, descriptors = self._ds._extract(image, path)

        return self._ds.create_conditioning_tensor(keypoints, descriptors, mode="direct", training=training, aggregate=self._ds.aggregate, max_keypoints=self.num_positions)


class ConditionImageDataset(Dataset):
    """
    For training mode : length is n_images * n_conditions and the condition is derived from the index, so that PureBatchSampler can build pure batches from the condition
    Validation mode: length is n_images and every item uses active_condition
    """

    def __init__(
        self,
        source: Union[str, Path, list],
        conditions,
        resolution: int = 512,
        downsample_factor: int = 8,
        device: str = "cuda",
        training: bool = True,
        noise_std_dev: float = 0.0,
        aggregate: bool = False,
        base_desc: str = "sift",
        max_images: Optional[int] = None,
        num_positions: int = 2048,
        loma_features=(),
    ):
        conditions = list(conditions)
        if not conditions:
            raise ValueError("At least one condition is required.")
        condition_input_dims(conditions)  # validates

        self.image_paths = _collect_image_paths(source)
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        if not self.image_paths:
            raise ValueError(f"No images found under {source!r}.")

        self.conditions = conditions
        self.num_conditions = len(conditions)
        self.training = training
        self.resolution = resolution
        self.num_positions = num_positions
        self.active_condition = conditions[0]

        self._extractor_kwargs = dict(
            resolution=resolution, downsample_factor=downsample_factor, device=device,
            noise_std_dev=noise_std_dev, aggregate=aggregate,
            base_desc=base_desc, num_positions=num_positions, loma_features=loma_features,
        )
        self._extractors = {}

        self.image_transforms = T.Compose([
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        n_images = len(self.image_paths)
        for i, condition in enumerate(conditions):
            indices = torch.arange(i, n_images * self.num_conditions, self.num_conditions)
            setattr(self, f"{condition.lower()}_indices", indices)
            if condition == "adek":
                self.ade20k_indices = indices

    def _get_extractor(self, condition: str) -> _ConditionExtractor:
        if condition not in self._extractors:
            self._extractors[condition] = _ConditionExtractor(condition, **self._extractor_kwargs)
        return self._extractors[condition]

    def set_active_condition(self, condition: str):
        assert condition in self.conditions, f"'{condition}' not in {self.conditions}"
        self.active_condition = condition

    def __len__(self):
        if self.training:
            return len(self.image_paths) * self.num_conditions
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        step = self.num_conditions if self.training else 1
        for _ in range(_MAX_SAMPLE_RETRIES):
            try:
                return self._load(idx)
            except Exception as exc:
                path = self.image_paths[idx // self.num_conditions if self.training else idx]
                logging.warning("skipping unreadable sample %s: %s", path, exc)
                idx = (idx + step) % len(self)
        raise RuntimeError(
            f"{_MAX_SAMPLE_RETRIES} consecutive samples failed to load; "
            f"check the dataset under {self.image_paths[0].parent}"
        )

    def _load(self, idx: int) -> dict:
        if self.training:
            image_idx = idx // self.num_conditions
            condition = self.conditions[idx % self.num_conditions]
        else:
            image_idx = idx
            condition = self.active_condition

        path = self.image_paths[image_idx]
        image = Image.open(path).convert("RGB")

        conditioning_tensor, positions = self._get_extractor(condition)(image, training=self.training, path=path)
        positions = _fixed_size_positions(positions, self.num_positions)
        image_ = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        return {
            "input_image": self.image_transforms(image),
            "conditioning_tensor": conditioning_tensor,
            "positions": positions,
            "condition": condition,
            "input_image_np": np.array(image_),
            "image_path": str(path),
        }
