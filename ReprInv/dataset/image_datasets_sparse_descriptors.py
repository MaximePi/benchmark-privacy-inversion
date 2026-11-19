import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from lightglue import ALIKED
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from submodules.superpoint.superpoint_fs import SuperPoint
from submodules.ninjadesc.loader import BASE_DESCRIPTORS, ensure_2d, load_ninjadesc_models

ImageFile.LOAD_TRUNCATED_IMAGES = True 
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DESCRIPTOR_DIMS = {"SIFT": 128, "SUPERPOINT": 256, "XFEAT": 64, "ALIKED": 128, "LOMA": 256}
# Upright ALIKED: LightGlue aliked-n16, 2048-keypoint cap, on the aspect-preserving frame.
ALIKED_MODEL = "aliked-n16"
ALIKED_MAX_KEYPOINTS = 2048
_NINJADESC_PATCH_MARGIN = 16


def _collect_image_paths(source: Union[str, Path, list]) -> list:
    if isinstance(source, list):
        return [Path(p) for p in source]
    source = Path(source)
    if source.is_file():
        return [source]
    paths = []
    for root, _dirs, files in os.walk(source, followlinks=True):
        for name in files:
            if Path(name).suffix.lower() in _IMG_EXTS:
                paths.append(Path(root) / name)
    return sorted(set(paths))


def _ninjadesc_runtime_device(preferred: torch.device) -> torch.device:
    try:
        from torch.utils.data import get_worker_info
    except Exception:
        return preferred
    if get_worker_info() is not None and preferred.type == "cuda":
        return torch.device("cpu")
    return preferred


def _pil_rgb_to_gray_hw1_like_prep(input_image: Image.Image) -> np.ndarray:
    rgb = np.array(input_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img_norm = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    return np.float32(cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY))[..., None]


def _extract_patches_32_aligned(keypoints_xy: np.ndarray, gray_hw1: np.ndarray):
    m = _NINJADESC_PATCH_MARGIN
    h, w = gray_hw1.shape[0], gray_hw1.shape[1]
    patches = []
    kpts = []
    for kp in keypoints_xy:
        c = int(round(float(kp[0])))
        r = int(round(float(kp[1])))
        if r > m - 1 and c > m - 1 and r + m < h and c + m < w:
            p = gray_hw1[r - m : r + m, c - m : c + m, :]
            patches.append(p.transpose(2, 0, 1))
            kpts.append([float(c), float(r)])
    if not patches:
        return np.zeros((1, 1, 32, 32), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)
    return np.stack(patches, axis=0).astype(np.float32), np.asarray(kpts, dtype=np.float32)


class BaseSparseDescriptorDataset(Dataset, ABC):
    def __init__(self,
                 resolution=512,
                 downsample_factor=8,
                 noise_std_dev=0.0,
                 max_keypoints_extraction=8192,
                 num_keypoints_training_mid=1024):

        self.resolution = resolution
        self.downsample_factor = downsample_factor
        self.noise_std_dev = noise_std_dev
        self.max_keypoints_extraction = max_keypoints_extraction
        self.num_keypoints_training_mid = num_keypoints_training_mid

        self.xfeat = None
        self.superpoint = None
        self.aliked = None

        self.image_transforms = T.Compose([
            T.Resize((self.resolution, self.resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        self.image_transforms_superpoint = T.Compose([
            T.Resize((self.resolution, self.resolution)),
            T.ToTensor(),
            T.Grayscale(num_output_channels=1),
        ])

        self.extractor = {
            'SIFT': self.extract_SIFT,
            'SUPERPOINT': self.extract_SUPERPOINT,
            'XFEAT': self.extract_XFEAT,
        }

    _extractors = {
        "SIFT": "extract_SIFT",
        "SUPERPOINT": "extract_SUPERPOINT",
        "XFEAT": "extract_XFEAT",
        "ALIKED": "extract_ALIKED",
    }

    def _extract(self, image: Image.Image, path=None):
        if self.condition == "LOMA":
            return self.extract_LOMA(image, path)
        return getattr(self, self._extractors[self.condition])(image)

    def _frame_size(self, w, h):
        """Aspect-preserving frame with the longest side at the resolution."""
        scale = self.resolution / max(w, h)
        return max(1, round(w * scale)), max(1, round(h * scale))

    def _frame_to_square(self, xy, frame_w, frame_h):
        """Keypoints from the aspect-preserving frame into the resolution x resolution square."""
        return xy * np.array([self.resolution / frame_w, self.resolution / frame_h], dtype=np.float32)

    def extract_LOMA(self, input_image, path):
        """LoMa-B (DaD keypoints, DeDoDe-G descriptors, 256-d), precomputed on the aspect-preserving frame like
        extract_ALIKED and scaled into the square. LoMa itself is not run here."""
        feats = np.load(self.loma_features[Path(path).stem])
        xy = self._frame_to_square(feats["xy"].astype(np.float32), *self._frame_size(*input_image.size))
        return xy, feats["desc"].astype(np.float32)

    def _init_xfeat(self):
        if self.xfeat is None:
            self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat',
                                        pretrained=True, top_k=self.max_keypoints_extraction)
            self.xfeat.to("cpu")
            self.xfeat.dev = "cpu"
            self.xfeat.net = self.xfeat.net.to("cpu")
            self.xfeat.eval()

    def _init_superpoint(self):
        if self.superpoint is None:
            self.superpoint = SuperPoint()
            self.superpoint.to("cpu")
            self.superpoint.eval()

    def _init_aliked(self):
        if self.aliked is None:
            self.aliked = ALIKED(model_name=ALIKED_MODEL, max_num_keypoints=ALIKED_MAX_KEYPOINTS).eval()
            self.aliked.to(self.aliked_device)

    @torch.no_grad()
    def extract_ALIKED(self, input_image):
        """ALIKED (LightGlue, default preprocessing) on the aspect-preserving frame with the longest side at the
        resolution; keypoints are then scaled into the resolution x resolution square that the other conditions
        and the target image use, so ALIKED never sees the squashed image."""
        self._init_aliked()
        frame_w, frame_h = self._frame_size(*input_image.size)
        image = input_image.resize((frame_w, frame_h), Image.Resampling.BILINEAR)
        image_torch = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        feats = self.aliked.extract(image_torch.to(self.aliked_device))
        xy = feats["keypoints"][0].float().cpu().numpy()
        desc = feats["descriptors"][0].float().cpu().numpy()
        return self._frame_to_square(xy, frame_w, frame_h), desc

    @torch.no_grad()
    def extract_XFEAT(self, input_image):
        self._init_xfeat()
        input_image_torch = self.image_transforms(input_image)[None].cpu()
        out = self.xfeat.detectAndCompute(input_image_torch, top_k=self.max_keypoints_extraction)[0]
        return out['keypoints'].cpu().numpy(), out['descriptors'].cpu().numpy()

    @torch.no_grad()
    def extract_SIFT(self, input_image):
        sift = cv2.SIFT_create(nfeatures=self.max_keypoints_extraction, contrastThreshold=0.04, edgeThreshold=20)
        input_image_cv = np.array(input_image.convert("L"))
        if input_image_cv.shape[0] != self.resolution or input_image_cv.shape[1] != self.resolution:
            input_image_cv = cv2.resize(input_image_cv, (self.resolution, self.resolution),
                                        interpolation=cv2.INTER_LINEAR)
        if input_image_cv.dtype != np.uint8:
            input_image_cv = (input_image_cv * 255).astype(np.uint8)

        keypoints_cv = sift.detect(input_image_cv, None)
        for kp in keypoints_cv:
            kp.angle = 0
        keypoints_cv, descriptors = sift.compute(input_image_cv, keypoints_cv)
        if descriptors is None:
            return np.zeros((self.max_keypoints_extraction, 2)), np.zeros((self.max_keypoints_extraction, 128))
        eps = 1e-7
        descriptors = descriptors / (np.sum(descriptors, axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        keypoints = np.array([kp.pt for kp in keypoints_cv], dtype=np.float32)
        return keypoints, descriptors

    @torch.no_grad()
    def extract_SUPERPOINT(self, input_image):
        self._init_superpoint()
        input_image_torch = self.image_transforms_superpoint(input_image)[None]
        out = self.superpoint({"image": input_image_torch})
        return out['keypoints'][0].numpy(), out['descriptors'][0].numpy().T

    @torch.no_grad()
    def create_conditioning_tensor(self, keypoints, descriptors, mode='direct', training=False, aggregate=False, max_keypoints=None):
        h = w = self.resolution
        eps = 1e-7

        n_keep = None
        if training:
            random_factor = np.random.rand()
            n_keep = int((self.num_keypoints_training_mid // 2) * (1 + random_factor))
        if max_keypoints is not None:
            n_keep = max_keypoints if n_keep is None else min(n_keep, max_keypoints)
        if n_keep is not None:
            n_keep = min(n_keep, self.max_keypoints_extraction)
            if n_keep < len(keypoints):
                idxs = np.random.permutation(len(keypoints))[:n_keep]
                keypoints = keypoints[idxs]
                descriptors = descriptors[idxs]

        if isinstance(descriptors, torch.Tensor):
            descriptors = descriptors.cpu().numpy()

        descriptors = (descriptors - descriptors.min(axis=1, keepdims=True)) / (
            descriptors.max(axis=1, keepdims=True) - descriptors.min(axis=1, keepdims=True) + eps)
        descriptors = descriptors * 2 - 1

        if self.noise_std_dev > 0:
            noise = np.random.normal(0, self.noise_std_dev, keypoints.shape)
            keypoints = keypoints + noise
            keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
            keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)

        if mode == 'sum_pool':
            dim = descriptors.shape[-1]
            conditioning_tensor = torch.zeros((dim, h, w))
            for kp, desc in zip(keypoints, descriptors):
                x, y = int(kp[0]), int(kp[1])
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                conditioning_tensor[:, y, x] = torch.tensor(desc, dtype=torch.float32).clone()
            conditioning_tensor = torch.nn.functional.avg_pool2d(
                conditioning_tensor,
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
            ) * (self.downsample_factor ** 2)
        elif mode == 'direct':
            dim = descriptors.shape[-1]
            lh, lw = h // self.downsample_factor, w // self.downsample_factor
            conditioning_tensor = torch.zeros((dim, lh, lw))
            if aggregate:
                count_t = torch.zeros((lh, lw))
                for kp, desc in zip(keypoints, descriptors):
                    x, y = int(kp[0] // self.downsample_factor), int(kp[1] // self.downsample_factor)
                    if x < 0 or y < 0 or x >= lw or y >= lh:
                        continue
                    conditioning_tensor[:, y, x] += torch.tensor(desc, dtype=torch.float32)
                    count_t[y, x] += 1.0
                mask = count_t > 0
                conditioning_tensor[:, mask] /= count_t[mask]
            else:
                for kp, desc in zip(keypoints, descriptors):
                    x, y = int(kp[0] // self.downsample_factor), int(kp[1] // self.downsample_factor)
                    if x < 0 or y < 0 or x >= lw or y >= lh:
                        continue
                    conditioning_tensor[:, y, x] = torch.tensor(desc, dtype=torch.float32).clone()

        if len(keypoints) > self.max_keypoints_extraction:
            keypoints = np.random.permutation(keypoints)[:self.max_keypoints_extraction]

        keypoints = torch.from_numpy(keypoints[:, :2].copy()).to(conditioning_tensor.dtype).clone()
        keypoints /= torch.tensor([w, h], dtype=conditioning_tensor.dtype)

        return conditioning_tensor, keypoints

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class SparseDescriptorImageDataset(BaseSparseDescriptorDataset):
    def __init__(
        self,
        source: Union[str, Path, list],
        condition: str,
        resolution: int = 512,
        downsample_factor: int = 8,
        aggregate: bool = False,
        aliked_device: str = "cpu",
        loma_features=(),
    ):
        if condition not in DESCRIPTOR_DIMS:
            raise ValueError(f"Unknown condition '{condition}'. Choose from {list(DESCRIPTOR_DIMS)}.")
        super().__init__(resolution=resolution,downsample_factor=downsample_factor)
        self.condition = condition
        self.aggregate = aggregate
        self.aliked_device = aliked_device
        if condition == "LOMA":  # one .npz (xy, desc) per image, found by the image's file stem
            paths = [p for d in loma_features for p in Path(d).glob("*.npz")]
            self.loma_features = {p.stem: p for p in paths}
            if len(self.loma_features) != len(paths):
                raise ValueError(f"duplicate image stems across LoMa feature folders {list(loma_features)}")
        self.image_paths = _collect_image_paths(source)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        keypoints, descriptors = self._extract(image, path)
        condition, positions = self.create_conditioning_tensor(
            keypoints, descriptors, mode='direct', aggregate=self.aggregate
        )

        return {
            "image_path": str(path),
            "input_image": np.array(image),
            "condition": condition,     # (D, L, L)
            "positions": positions,     # (N, 2) in [0, 1]
        }


class NinjadescImageDataset(BaseSparseDescriptorDataset):
    """
    SIFT keypoints, 32x32 patches around them, and NinjaDesc privacy descriptors
    from the vendored ``submodules.ninjadesc`` loader (base model -> privacy head).
    """

    def __init__(
        self,
        source: Union[str, Path, list],
        condition: str,
        resolution: int = 512,
        downsample_factor: int = 8,
        aggregate: bool = False,
        base_desc: str = "sift",
        device: str = "cuda",
        privacy_batch_size: int = 256):
        
        super().__init__(resolution=resolution, downsample_factor=downsample_factor)
        self.condition = condition
        self.aggregate = aggregate
        self.base_desc = base_desc
        self.image_paths = _collect_image_paths(source)
        self.privacy_batch_size = privacy_batch_size
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._runtime_device = None
        self._privacy = None
        self._base_model = None

    def _load_privacy_models(self):
        if self._privacy is not None:
            return
        runtime_device = _ninjadesc_runtime_device(self._device)
        self._runtime_device = runtime_device
        priv, base = load_ninjadesc_models(self.base_desc)
        self._privacy = priv.to(runtime_device).eval()
        self._base_model = base.to(runtime_device).eval()

    @torch.no_grad()
    def _privacy_descriptors_from_patches(self, patches: torch.Tensor) -> np.ndarray:
        """patches: (N, 1, 32, 32) float CPU tensor."""
        self._load_privacy_models()
        dev = self._runtime_device or _ninjadesc_runtime_device(self._device)
        outs = []
        for i in range(0, patches.shape[0], self.privacy_batch_size):
            chunk = patches[i : i + self.privacy_batch_size].to(dev)
            base_descs = self._base_model(chunk)
            outs.append(self._privacy(ensure_2d(base_descs)).cpu())
        return torch.cat(outs, dim=0).numpy()

    @torch.no_grad()
    def extract_NINJADESC(self, input_image):
        input_image = input_image.resize((self.resolution, self.resolution), Image.LANCZOS)
        keypoints_xy, _sift_desc = self.extract_SIFT(input_image)
        gray = _pil_rgb_to_gray_hw1_like_prep(input_image)
        patches_np, keypoints = _extract_patches_32_aligned(keypoints_xy, gray)
        descriptors = self._privacy_descriptors_from_patches(torch.from_numpy(patches_np))
        return keypoints, descriptors

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        keypoints, descriptors = self.extract_NINJADESC(image)
        condition, positions = self.create_conditioning_tensor(keypoints, descriptors, mode='direct', aggregate=self.aggregate)

        return {
            "image_path": str(path),
            "input_image": np.array(image),
            "condition": condition,     
            "positions": positions,     
        }
