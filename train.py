import argparse
import datetime
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from diffusers import DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler

from ReprInv.model.cond_lora_unet import UNet2DConditionModelConditionalLora
from ReprInv.pipeline import Representation_Inversion_Diffusion
from ReprInv.utils.batch_sampler import PureBatchSampler
from ReprInv.dataset.train_datasets import (
    ConditionImageDataset,
    condition_input_dims,
    CONDITION_INPUT_DIMS,
    MODEL_BACKED_CONDITIONS,
)
from ReprInv.eval import compute_image_reconstruction_metrics


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(port))
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=7200),
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def build_datasets(args):
    train_set = ConditionImageDataset(
        args.train_images,
        conditions=args.conditions,
        resolution=args.resolution,
        downsample_factor=args.downsample_factor,
        device=args.extractor_device,
        training=True,
        noise_std_dev=args.noise_std_dev,
        aggregate=args.aggregate,
        num_positions=args.num_positions,
        loma_features=args.loma_features,
    )
    val_set = ConditionImageDataset(
        args.val_images or args.train_images,
        conditions=args.conditions,
        resolution=args.resolution,
        downsample_factor=args.downsample_factor,
        device=args.extractor_device,
        training=False,
        noise_std_dev=0.0,
        aggregate=args.aggregate,
        max_images=args.num_val_images,
        num_positions=args.num_positions,
        loma_features=args.loma_val_features or args.loma_features,
    )
    return train_set, val_set


@torch.no_grad()
def validate(pipe, dataloader_val, dataset_val, condition, args, device, step, rank_output_dir):
    dataset_val.set_active_condition(condition)
    pipe.set_condition(condition)
    pipe.eval()

    vae_dtype, unet_dtype = pipe.vae.dtype, pipe.unet.module.dtype
    pipe.vae.to(dtype=torch.float32)
    pipe.unet.module.to(dtype=torch.float32)

    out_dir = Path(rank_output_dir) / f"val_step{step}_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    recons, gts = [], []
    for i, batch in enumerate(dataloader_val):
        image = pipe(
            batch["conditioning_tensor"][0:1].to(device),
            batch["positions"][0:1].to(device),
            num_inference_steps=args.val_inference_steps,
            guidance_scale=args.val_guidance_scale,
            seed=args.seed,
            dtype=torch.float32,
        )
        pred = image.detach().cpu().squeeze(0).permute(1, 2, 0).float().numpy()
        gt = batch["input_image_np"][0].cpu().numpy()
        recons.append(pred)
        gts.append(gt)

        pred_pil = Image.fromarray((pred * 255).round().clip(0, 255).astype("uint8"))
        gt_pil = Image.fromarray(gt).resize(pred_pil.size, Image.LANCZOS)
        w, h = pred_pil.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(gt_pil, (0, 0))
        combined.paste(pred_pil, (w, 0))
        combined.save(out_dir / f"val_{i:04d}.png")

    agg, _ = compute_image_reconstruction_metrics(recons, gts, device=device, include_fid=False)
    with open(out_dir / "metrics.txt", "w") as f:
        for k, v in agg.items():
            f.write(f"{k}: {v}\n")
    logging.info(f"[val step {step}] {condition}: " + ", ".join(f"{k}={v}" for k, v in agg.items()))

    pipe.vae.to(dtype=vae_dtype)
    pipe.unet.module.to(dtype=unet_dtype)
    pipe.train()
    return agg


def save_checkpoint(path, epoch, overall_step, unet, optimizer, lr_scheduler, scaler, args):
    torch.save({
        "epoch": epoch,
        "overall_step": overall_step,
        "conditions": args.conditions,
        "unet_state_dict": unet.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
    }, path)
    logging.info(f"Saved checkpoint to {path}")


def init_from_checkpoint(unet, path, source_tasks, source_task, rank):
    """Warm-start new tasks from a trained multi-task model (weights only; fresh optimizer and schedule).

    Every weight whose name and shape match is copied. Per-task weights take the ``source_task`` entry:
    each task-embedding row and each ``convs_in.<task>``. ``conv_in`` keeps its value: in a checkpoint it
    is a copy of the last active task's ``convs_in``, and ``switch_lora`` replaces it with the current
    task's before every forward pass. Any other mismatch is an error, so nothing is left silently at its
    random initialization.
    """
    source = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["unet_state_dict"]
    row = source_tasks.index(source_task)
    target = unet.state_dict()
    state, n_copied, n_remapped = {}, 0, 0
    for name, value in target.items():
        if name in source and source[name].shape == value.shape:
            state[name] = source[name]
            n_copied += 1
        elif name.endswith("embedding.weight") and name in source and source[name].shape[0] == len(source_tasks):
            state[name] = source[name][row : row + 1].expand_as(value).clone()
            n_remapped += 1
        elif name.startswith("conv_in."):
            state[name] = value
        elif name.startswith("convs_in."):
            suffix = name.split(".")[-1]
            state[name] = source[f"convs_in.{source_task}.{suffix}"]
            if state[name].shape != value.shape:
                raise ValueError(f"{name}: {tuple(value.shape)} cannot start from {source_task} {tuple(state[name].shape)}")
            n_remapped += 1
        else:
            raise ValueError(f"no initialization for {name} {tuple(value.shape)}")
    unet.load_state_dict(state, strict=True)
    if rank == 0:
        print(f"Initialized from {path}: {n_copied} tensors copied, {n_remapped} per-task tensors from {source_task}")


def load_checkpoint(path, unet, optimizer, lr_scheduler, scaler, rank):
    ckpt = torch.load(path, map_location=lambda storage, loc: storage.cuda(rank))
    state_dict = ckpt["unet_state_dict"]
    clean = {k: v for k, v in state_dict.items() if ".lora_layer." not in k and not k.startswith("conv_in.")}
    unet.module.load_state_dict(clean, strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"] + 1, ckpt["overall_step"]


def train(rank, world_size, args):
    setup(rank, world_size, args.master_port)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0

    input_dims = condition_input_dims(args.conditions)
    if is_main:
        print(f"Conditions: {list(zip(args.conditions, input_dims))}")

    unet = UNet2DConditionModelConditionalLora.from_pretrained(
        args.pretrained_model_name, subfolder="unet")
    unet.add_lora_layers(
        lora_rank=args.lora_rank,
        tasks=args.conditions,
        input_dims=input_dims,
        network_alpha=args.network_alpha,
    )
    unet.add_sparse_pixel_graph_encoder()
    if args.init_from_checkpoint:
        init_from_checkpoint(unet, args.init_from_checkpoint, args.init_source_tasks, args.init_source_task, rank)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name, subfolder="vae")
    vae.requires_grad_(False)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision, torch.float32)

    unet.to(device)
    unet = DDP(unet, device_ids=[rank], find_unused_parameters=True)
    pipe = Representation_Inversion_Diffusion(unet, vae, noise_scheduler)
    pipe.to(device, weight_dtype=weight_dtype)
    pipe.train()

    dataset, dataset_val = build_datasets(args)
    per_gpu_batch_size = max(1, args.batch_size // world_size)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    batch_sampler = PureBatchSampler(train_sampler, dataset, batch_size=per_gpu_batch_size, conditions=args.conditions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.dataloader_num_workers, pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, shuffle=False, batch_size=1, num_workers=0)

    if is_main:
        print(f"Train images: {len(dataset.image_paths)} | val images: {len(dataset_val.image_paths)}")
        print(f"Global batch size: {args.batch_size} | per GPU: {per_gpu_batch_size}")

    optimizer = pipe.unet.module.configure_optimizers(args)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles if args.lr_scheduler == "cosine_with_restarts" else 1,
        power=args.lr_power if args.lr_scheduler == "polynomial" else 1.0)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    start_epoch, overall_step = 0, 0
    if args.resume_from_checkpoint:
        if is_main:
            print(f"Resuming from {args.resume_from_checkpoint}")
        start_epoch, overall_step = load_checkpoint(
            args.resume_from_checkpoint, unet, optimizer, lr_scheduler, scaler, rank)

    epoch = start_epoch
    while overall_step < args.num_train_steps:
        pipe.train()

        for batch in dataloader:
            pipe.set_condition(batch["condition"][0]) # PureBatchSampler guarantees one condition per batch.

            loss = pipe.forward_train(
                batch["input_image"], batch["conditioning_tensor"], batch["positions"])

            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (overall_step + 1) % args.gradient_accumulation_steps == 0:
                if args.mixed_precision == "fp16":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.module.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(unet.module.parameters(), 1.0)
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if is_main and overall_step > 0 and overall_step % args.log_interval == 0:
                logging.info(f"epoch {epoch} step {overall_step}: loss {loss.item():.4f}, "
                             f"lr {lr_scheduler.get_last_lr()[0]:.2e}")

            if overall_step > 0 and overall_step % args.val_interval == 0:
                if is_main:
                    condition = args.conditions[np.random.randint(len(args.conditions))]
                    validate(pipe, dataloader_val, dataset_val, condition, args,
                             device, overall_step, args.output_dir)
                dist.barrier()

            if overall_step > 0 and overall_step % args.save_interval == 0:
                if is_main:
                    save_checkpoint(
                        Path(args.output_dir) / f"checkpoint_epoch{epoch+1}_step{overall_step}.pth",
                        epoch, overall_step, unet, optimizer, lr_scheduler, scaler, args)
                dist.barrier()

            overall_step += 1
            if overall_step >= args.num_train_steps:
                break

        epoch += 1

    if is_main:
        logging.info(f"finished at step {overall_step} of {args.num_train_steps} "
                     f"after {epoch - start_epoch} epoch(s)")
        save_checkpoint(Path(args.output_dir) / "checkpoint_final.pth",
                        epoch, overall_step, unet, optimizer, lr_scheduler, scaler, args)
    cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / conditions
    parser.add_argument("--train_images", required=True, help="Folder of training images")
    parser.add_argument("--val_images", default=None, help="Folder of validation images (default: train_images)")
    parser.add_argument("--conditions", nargs="+", required=True,
                        choices=sorted(CONDITION_INPUT_DIMS), metavar="COND",
                        help=f"One or more of: {sorted(CONDITION_INPUT_DIMS)}")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_val_images", type=int, default=50)
    parser.add_argument("--num_positions", type=int, default=2048)
    parser.add_argument("--noise_std_dev", type=float, default=4.0)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--extractor_device", default="cuda")
    parser.add_argument("--loma_features", nargs="+", default=[],
                        help="folders of precomputed LoMa features (one .npz per image stem, with xy and desc) for LOMA training")
    parser.add_argument("--loma_val_features", nargs="+", default=[],
                        help="LoMa feature folders of the validation images (default: --loma_features)")

    # should not be modified
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--downsample_factor", type=int, default=8)
    
    # model
    parser.add_argument("--pretrained_model_name", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--network_alpha", type=float, default=24)

    # optimisation main
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_train_steps", type=int, default=1000000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)

    # optimisation optional
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lora_plus", action="store_true")
    parser.add_argument("--lora_lr_A", type=float, default=5e-5)
    parser.add_argument("--lora_lr_B", type=float, default=5e-4)
    parser.add_argument("--lora_wd_A", type=float, default=1e-2)
    parser.add_argument("--lora_wd_B", type=float, default=1e-2)

    # run control
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--val_interval", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=25000)
    parser.add_argument("--val_inference_steps", type=int, default=25)
    parser.add_argument("--val_guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--init_from_checkpoint", default=None,
                        help="warm-start weights (not optimizer) from a trained checkpoint, e.g. the released model")
    parser.add_argument("--init_source_tasks", nargs="+", default=["SIFT", "SUPERPOINT", "XFEAT"],
                        help="task order of --init_from_checkpoint (released model: SIFT SUPERPOINT XFEAT)")
    parser.add_argument("--init_source_task", default="SIFT",
                        help="source task whose per-task weights initialize the new tasks")
    parser.add_argument("--master_port", type=int, default=12102)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("Training requires at least one CUDA device.")

    model_backed = [c for c in args.conditions if c in MODEL_BACKED_CONDITIONS]
    if args.dataloader_num_workers > 0 and model_backed:
        print(f"Warning: conditions {model_backed} load a model per DataLoader worker; "
              f"--dataloader_num_workers 0 is safer.")

    print(f"Using {world_size} GPU(s).")
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
