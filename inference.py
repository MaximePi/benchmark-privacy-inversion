import os

import torch
from diffusers import DDPMScheduler, AutoencoderKL
from huggingface_hub import hf_hub_download

from ReprInv.model.cond_lora_unet import UNet2DConditionModelConditionalLora
from ReprInv.pipeline import Representation_Inversion_Diffusion
from ReprInv.scheduler.adaptive_ddpm import AdaptiveDDPMScheduler
from ReprInv.scheduler.spatial_ddpm import SpatialDDPMScheduler

_SEGMENTATION_CONDITIONS = {"segloc", "sam", "gsff", "adek", "ppnesf"}
_DESCRIPTOR_DIMS = {
    "SIFT": 128, "SUPERPOINT": 256, "XFEAT": 64,
    "SIFT_privacy": 128, "SUPERPOINT_privacy": 256,
    "SIFT_ninjadesc": 128,
    "ALIKED": 128,
    "LOMA": 256,
}

_CONDITION_ALIASES = {
    "AASE_SIFT": "SIFT_privacy",
    "AASE_SUPERPOINT": "SUPERPOINT_privacy",
    "SIFT_permutation": "SIFT",
    "SIFT_random_lines": "SIFT",
    "SUPERPOINT_permutation": "SUPERPOINT",
    "SUPERPOINT_random_lines": "SUPERPOINT",
}


def resolve_internal_condition(config: dict, condition: str) -> str:
    ### Map a user entered condition to the task key the UNet was built with.
    if config.get("ninjadesc"):
        return "SIFT"
    return _CONDITION_ALIASES.get(condition, condition)


def _remap_conv_in_keys(state_dict: dict, condition: str) -> dict:
    remapped = state_dict.copy()
    for key, value in state_dict.items():
        for suffix in ("weight", "bias"):
            fragment = f"convs_in.{condition}.{suffix}"
            if fragment in key:
                remapped[key.replace(fragment, f"conv_in.{suffix}")] = value
    return remapped


def _parse_tasks(config: dict) -> tuple:
    if config.get("ninjadesc"):
        return ["SIFT"], [128]

    raw = config.get("condition") or config.get("default_condition", "")
    tasks = [_CONDITION_ALIASES.get(t.strip(), t.strip()) for t in raw.split(",")]

    if all(t in _SEGMENTATION_CONDITIONS for t in tasks):
        return tasks, [3] * len(tasks)

    return tasks, [_DESCRIPTOR_DIMS[t] for t in tasks]


def load_model(config: dict, condition: str, device: str) -> Representation_Inversion_Diffusion:
    if config.get("model_type") != "diffusion":
        raise ValueError(
            f"load_model only supports diffusion models, "
            f"got model_type='{config.get('model_type')}'"
        )

    parts = config["base_model_name"].split("/", 2)
    pretrained = f"{parts[0]}/{parts[1]}"
    sf_prefix = parts[2] if len(parts) > 2 else ""
    lora_rank = config.get("lora_rank", 64)
    network_alpha = config.get("network_alpha", 24)
    schedule_type = config.get("schedule_type", "default")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def _sf(name):
        return f"{sf_prefix}/{name}" if sf_prefix else name

    unet = UNet2DConditionModelConditionalLora.from_pretrained(pretrained, subfolder=_sf("unet"), token=token)

    tasks, input_dims = _parse_tasks(config)
    unet.add_lora_layers(
        lora_rank=lora_rank,
        tasks=tasks,
        input_dims=input_dims,
        network_alpha=network_alpha,
    )
    unet.add_sparse_pixel_graph_encoder()

    if config.get("weights_path"):  # a locally trained checkpoint
        ckpt_path = config["weights_path"]
    else:
        ckpt_path = hf_hub_download(
            repo_id=config["hf_repo"],
            filename=config["weights_file"],
            token=token,
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["unet_state_dict"]
    state_dict = _remap_conv_in_keys(state_dict, condition)
    state_dict = {
        k: v for k, v in state_dict.items()
        if ".lora_layer." not in k and "conv_in." not in k
    }
    unet.load_state_dict(state_dict, strict=False)

    base_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder=_sf("scheduler"), token=token)
    if schedule_type == "spatial":
        scheduler = SpatialDDPMScheduler(base_scheduler)
    elif schedule_type == "adaptive":
        scheduler = AdaptiveDDPMScheduler(base_scheduler)
    else:
        scheduler = base_scheduler

    vae = AutoencoderKL.from_pretrained(pretrained, subfolder=_sf("vae"), token=token)
    vae.requires_grad_(False)

    pipe = Representation_Inversion_Diffusion(unet, vae, scheduler)
    pipe.to(device, torch.float32)
    pipe.set_condition(resolve_internal_condition(config, condition))
    pipe.eval()

    return pipe
