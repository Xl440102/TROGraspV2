import os
import argparse
import math
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from dataset.TROMixedHandZarrDataset import create_dataloader_zarr
from model.troe_graph import RobotGraph
from model.vqvae_encoder import VQVAEEncoder


def build_step_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
):
    warmup_steps = int(max(0, warmup_steps))
    total_steps = int(max(1, total_steps))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        denom = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def sync_scheduler_to_step(scheduler, step: int):
    if scheduler is None:
        return

    step = int(max(0, step))
    scheduler.last_epoch = step
    scheduler._step_count = step + 1

    lrs = [
        base_lr * lr_lambda(step)
        for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    for param_group, lr in zip(scheduler.optimizer.param_groups, lrs):
        param_group["lr"] = lr
    scheduler._last_lr = lrs


def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    save_dir: str,
    step_in_epoch: int,
):
    ckpt_dir = os.path.join(save_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"step_{step_in_epoch}.pth")

    unwrapped = accelerator.unwrap_model(model)
    payload = {
        "step_in_epoch": int(step_in_epoch),
        "model_state": unwrapped.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }

    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    ckpt_path: str,
    load_optimizer: bool = True,
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    unwrapped = accelerator.unwrap_model(model)

    unwrapped.load_state_dict(ckpt["model_state"], strict=True)

    step_in_epoch = int(ckpt.get("step_in_epoch", 0))
    if load_optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])

        if scheduler is not None and ckpt.get("scheduler_state", None) is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
    else:
        sync_scheduler_to_step(scheduler, step_in_epoch)

    return step_in_epoch


def train(config):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    ddp_kwargs = DistributedDataParallelKwargs(
        gradient_as_bucket_view=True,
        bucket_cap_mb=50,
        find_unused_parameters=False
    )
    accelerator = Accelerator(
        mixed_precision=config.train.mixed_precision,
        gradient_accumulation_steps=config.train.accumulator,
        kwargs_handlers=[ddp_kwargs]
    )

    save_dir = config.train.save_dir
    total_steps = int(config.train.total_steps)

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    tb_writer = None
    if accelerator.is_main_process:
        tb_dir = os.path.join(save_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_dir)

    model = RobotGraph(**config.model)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        total_params = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
        print(f"Total params: {total_params}", flush=True)


    vqvae_encoder = VQVAEEncoder(**config.vqvae)
    vqvae_encoder.eval()
    vqvae_encoder.to(accelerator.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.train.lr),
        fused=False,
    )

    dataloader = create_dataloader_zarr(config.dataset)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    scheduler = build_step_scheduler(
        optimizer,
        warmup_steps=int(config.train.warmup_steps),
        total_steps=total_steps,
    )

    resume_from = str(getattr(config.train, "resume_from", "")).strip()
    load_optimizer = bool(config.train.get("load_optimizer", True))
    step_in_epoch = 0
    if resume_from:
        step_in_epoch = load_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            resume_from,
            load_optimizer=load_optimizer,
        )
        if accelerator.is_main_process:
            resume_mode = "model+optimizer+scheduler" if load_optimizer else "model-only"
            print(
                f"Resumed {resume_mode} from {resume_from}, "
                f"step_in_epoch={step_in_epoch}, lr={optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )

    save_every = int(config.train.save_every)
    log_every = int(config.train.log_every)
    grad_clip = float(config.train.grad_clip)
    max_loss = float(config.train.get("max_loss", float("inf")))

    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss_total_sum = 0.0
    running_loss_trans_sum = 0.0
    running_loss_rot_sum = 0.0
    running_loss_count = 0
    skipped_batches = 0

    data_iter = iter(dataloader)

    while step_in_epoch < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with torch.no_grad():
            batch = vqvae_encoder(batch)

        if config.train.mixed_precision == "bf16":
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    batch[k] = v.to(torch.bfloat16)

        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss_dict = model(batch)
                loss_total = loss_dict["loss_total"]
                loss_trans = loss_dict["loss_trans"]
                loss_rot = loss_dict["loss_rot"]

            raw_loss_total = loss_total.detach()
            raw_loss_trans = loss_trans.detach()
            raw_loss_rot = loss_rot.detach()

            local_valid = (
                torch.isfinite(raw_loss_total).all()
                and raw_loss_total.item() <= max_loss
            )
            local_valid = torch.tensor(
                int(local_valid),
                device=loss_total.device,
                dtype=torch.int32,
            )
            global_valid = accelerator.reduce(local_valid, reduction="min").item()
            should_skip = (global_valid == 0)

            if should_skip:
                safe_loss = torch.nan_to_num(
                    loss_total,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ) * 0.0
            else:
                safe_loss = loss_total

            accelerator.backward(safe_loss)

            stepped = False
            if accelerator.sync_gradients:
                if should_skip:
                    optimizer.zero_grad(set_to_none=True)
                    skipped_batches += 1
                    report_loss = torch.nan_to_num(
                        raw_loss_total.float(),
                        nan=float("inf"),
                        posinf=float("inf"),
                        neginf=float("-inf"),
                    )
                    max_loss_global = accelerator.reduce(report_loss, reduction="max").item()
                    if accelerator.is_main_process:
                        print(
                            f"[skip {skipped_batches:>7d} at step {step_in_epoch:>7d}] "
                            f"loss_total={max_loss_global:.6f} > max_loss={max_loss:.6f}",
                            flush=True,
                        )
                else:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_in_epoch += 1
                    stepped = True

        if should_skip:
            continue

        log_loss_total_local = raw_loss_total
        log_loss_trans_local = raw_loss_trans
        log_loss_rot_local = raw_loss_rot

        running_loss_total_sum += float(log_loss_total_local.item())
        running_loss_trans_sum += float(log_loss_trans_local.item())
        running_loss_rot_sum += float(log_loss_rot_local.item())
        running_loss_count += 1

        if stepped and step_in_epoch > 0 and step_in_epoch % log_every == 0:
            loss_total_global = accelerator.gather_for_metrics(log_loss_total_local).mean().item()
            loss_trans_global = accelerator.gather_for_metrics(log_loss_trans_local).mean().item()
            loss_rot_global = accelerator.gather_for_metrics(log_loss_rot_local).mean().item()

            if accelerator.is_main_process:
                avg_total = running_loss_total_sum / max(1, running_loss_count)
                avg_trans = running_loss_trans_sum / max(1, running_loss_count)
                avg_rot = running_loss_rot_sum / max(1, running_loss_count)

                tb_writer.add_scalar("train/loss_total", loss_total_global, step_in_epoch)
                tb_writer.add_scalar("train/loss_trans", loss_trans_global, step_in_epoch)
                tb_writer.add_scalar("train/loss_rot", loss_rot_global, step_in_epoch)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step_in_epoch)

                print(
                    f"[step {step_in_epoch:>7d}/{total_steps}] "
                    f"loss_total={loss_total_global:.6f} | "
                    f"loss_trans={loss_trans_global:.6f} | "
                    f"loss_rot={loss_rot_global:.6f} | "
                    f"avg_total={avg_total:.6f} | "
                    f"avg_trans={avg_trans:.6f} | "
                    f"avg_rot={avg_rot:.6f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )

                running_loss_total_sum = 0.0
                running_loss_trans_sum = 0.0
                running_loss_rot_sum = 0.0
                running_loss_count = 0

        if (
            accelerator.is_main_process
            and save_every > 0
            and step_in_epoch > 0
            and (step_in_epoch % save_every == 0)
        ):
            save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                save_dir,
                step_in_epoch=step_in_epoch,
            )

    if accelerator.is_main_process:
        save_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            save_dir,
            step_in_epoch=step_in_epoch,
        )
        tb_writer.close()

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train(config)


if __name__ == "__main__":
    main()
