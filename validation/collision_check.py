import argparse
import json
import os
import sys
import traceback
from collections import defaultdict

import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from utils.tro_hand_model import create_hand_model


HAND_NAMES = [
    "allegro_hand",
    "leaphand",
    "shadow_hand",
    "dclaw_gripper",
    "barrett_hand",
    "ability_hand",
    "xhand",
    "schunk_hand"
]


def hand_model_surface_points(hand_model, q):
    """Return transformed hand link surface points for a batch of q."""
    link_pc_dict, _ = hand_model.batch_get_transformed_links_pc(q)
    return torch.cat(list(link_pc_dict.values()), dim=1).to(dtype=torch.float32)


def nearest_env_distance(hand_pc, env_pc, chunk_size=4096):
    """
    Compute the nearest environment point distance for every sampled hand point.

    Args:
        hand_pc: [B, H, 3]
        env_pc: [B, E, 3]
    Returns:
        min_dist: [B, H]
    """
    chunks = []
    for start in range(0, hand_pc.shape[1], chunk_size):
        hand_chunk = hand_pc[:, start : start + chunk_size]
        chunks.append(torch.cdist(hand_chunk, env_pc).amin(dim=-1))
    return torch.cat(chunks, dim=1)


def check_hand_surface_against_env_pc(
    hand_pc,
    env_pc,
    proximity_thresh=0.006,
    min_close_points=16,
    close_ratio_thresh=0.006,
    ground_z=0.0,
    ground_penetration_thresh=0.015,
    min_ground_points=4,
    ground_ratio_thresh=0.001,
    chunk_size=4096,
):
    """
    Point-cloud collision check that does not use env points inside the hand mesh.

    The old strategy could miss a hand that penetrates the floor if no sampled
    floor point happens to be inside the hand mesh. This checker uses two tests:
      1. hand surface samples close to the environment point cloud;
      2. hand surface samples clearly below the known ground plane.

    Args:
        hand_pc: torch.Tensor [B, H, 3], transformed hand surface/link points.
        env_pc: torch.Tensor [B, E, 3], environment point cloud.

    Returns:
        collision: torch.BoolTensor [B]
        stats: dict of torch.Tensor values.
    """
    if hand_pc.ndim != 3 or hand_pc.shape[-1] != 3:
        raise ValueError(f"hand_pc must have shape [B, H, 3], got {tuple(hand_pc.shape)}")
    if env_pc.ndim != 3 or env_pc.shape[-1] != 3:
        raise ValueError(f"env_pc must have shape [B, E, 3], got {tuple(env_pc.shape)}")
    if hand_pc.shape[0] != env_pc.shape[0]:
        raise ValueError(f"batch mismatch: hand_pc={hand_pc.shape[0]}, env_pc={env_pc.shape[0]}")

    env_pc = env_pc.to(device=hand_pc.device, dtype=torch.float32)
    hand_pc = hand_pc.to(dtype=torch.float32)
    num_hand_points = hand_pc.shape[1]

    min_dist = nearest_env_distance(hand_pc, env_pc, chunk_size=chunk_size)
    close = min_dist <= proximity_thresh
    close_count = close.sum(dim=1)
    close_ratio = close_count.to(torch.float32) / float(num_hand_points)
    proximity_collision = (
        (close_count >= min_close_points)
        & (close_ratio >= close_ratio_thresh)
    )

    ground_z_tensor = torch.full(
        (hand_pc.shape[0],),
        float(ground_z),
        device=hand_pc.device,
        dtype=torch.float32,
    )
    below_ground = hand_pc[..., 2] < (ground_z_tensor[:, None] - ground_penetration_thresh)
    below_ground_count = below_ground.sum(dim=1)
    below_ground_ratio = below_ground_count.to(torch.float32) / float(num_hand_points)
    ground_collision = (
        (below_ground_count >= min_ground_points)
        & (below_ground_ratio >= ground_ratio_thresh)
    )

    collision = proximity_collision | ground_collision
    stats = {
        "close_count": close_count,
        "close_ratio": close_ratio,
        "min_surface_distance": min_dist.amin(dim=1),
        "ground_z": ground_z_tensor,
        "below_ground_count": below_ground_count,
        "below_ground_ratio": below_ground_ratio,
        "proximity_collision": proximity_collision,
        "ground_collision": ground_collision,
    }
    return collision, stats


def load_sr_lift(sim_data_root):
    path = os.path.join(sim_data_root, "sr_lift.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=False)


def process_one_setting(args, robot_name, step, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    sim_data_root = os.path.join(args.sim_root, step, robot_name)
    lift_sr_dict = load_sr_lift(sim_data_root)
    valid_meta_names = set(lift_sr_dict.keys())

    vis_file_path = os.path.join(args.pred_root, robot_name, f"{step}.pth")
    vis_info = torch.load(vis_file_path, weights_only=False)

    hand_model = create_hand_model(robot_name, device=device)
    success_dict = {}
    diversity_dict = {}
    collision_stats = defaultdict(list)

    for info in tqdm(vis_info, desc=f"{robot_name} {step} gpu{gpu_id}"):
        meta_name = info["meta_name"]
        if meta_name not in valid_meta_names:
            continue

        env_pc = info["env_pc"].to(device=device, dtype=torch.float32)
        predict_q = info["predict_q"].to(device=device, dtype=torch.float32)

        hand_pc = hand_model_surface_points(hand_model, predict_q)
        collision, stats = check_hand_surface_against_env_pc(
            hand_pc=hand_pc,
            env_pc=env_pc,
            proximity_thresh=args.proximity_thresh,
            min_close_points=args.min_close_points,
            close_ratio_thresh=args.close_ratio_thresh,
            ground_z=args.ground_z,
            ground_penetration_thresh=args.ground_penetration_thresh,
            min_ground_points=args.min_ground_points,
            ground_ratio_thresh=args.ground_ratio_thresh,
            chunk_size=args.cdist_chunk_size,
        )

        for key, value in stats.items():
            if value.dtype == torch.bool:
                collision_stats[key].extend(value.cpu().tolist())
            else:
                collision_stats[key].extend(value.detach().cpu().to(torch.float32).tolist())

        non_collision = ~collision
        lifted = torch.tensor(
            lift_sr_dict[meta_name],
            dtype=torch.bool,
            device=non_collision.device,
        )
        success = non_collision & lifted

        success_q = predict_q[success.to(device=predict_q.device)]
        diversity_dict[meta_name] = success_q
        success_dict[meta_name] = {
            "lifted": lifted.cpu().tolist(),
            "non_collision": non_collision.cpu().tolist(),
            "success": success.cpu().tolist(),
            # per-candidate ids, aligned with the lists above
            "scene_ids": torch.as_tensor(
                info.get("valid_scene_ids", torch.arange(len(predict_q)))
            ).tolist(),
            "sample_ids": torch.as_tensor(
                info.get("sample_ids", torch.zeros(len(predict_q), dtype=torch.long))
            ).tolist(),
            # per-candidate continuous collision statistics (candidate 4a signals)
            "collision_stats": {
                key: value.detach().cpu().to(torch.float32).tolist()
                for key, value in stats.items()
            },
        }

    output_pt = os.path.join(sim_data_root, f"{args.output_name}.pt")
    output_json = os.path.join(sim_data_root, f"{args.output_name}.json")
    stats_json = os.path.join(sim_data_root, f"{args.output_name}_collision_stats.json")

    torch.save(success_dict, output_pt)

    if len(diversity_dict) == 0:
        all_success_q = torch.empty((0, hand_model.dof), device=device)
    else:
        all_success_q = torch.cat(list(diversity_dict.values()), dim=0)
    if all_success_q.shape[0] == 0:
        diversity_std = torch.tensor(0.0, device=device)
    else:
        diversity_std = torch.std(all_success_q, dim=0).mean()

    sr_dict = {}
    for meta_name, result in success_dict.items():
        meta_success = result["success"]
        sr_dict[meta_name] = sum(meta_success) / len(meta_success)

    avg_sr = sum(sr_dict.values()) / len(sr_dict) if len(sr_dict) > 0 else 0.0
    sr_dict["avg_sr"] = avg_sr
    sr_dict["diversity"] = diversity_std.item()
    with open(output_json, "w") as f:
        json.dump(sr_dict, f, indent=2)

    stats_summary = {}
    for key, values in collision_stats.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        stats_summary[key] = {
            "mean": tensor.mean().item() if tensor.numel() else 0.0,
            "max": tensor.max().item() if tensor.numel() else 0.0,
        }
    stats_summary["collision_rule"] = {
        "strategy": "hand surface points near env_pc OR hand surface points below estimated ground z",
        "proximity_thresh": args.proximity_thresh,
        "min_close_points": args.min_close_points,
        "close_ratio_thresh": args.close_ratio_thresh,
        "ground_z": args.ground_z,
        "ground_penetration_thresh": args.ground_penetration_thresh,
        "min_ground_points": args.min_ground_points,
        "ground_ratio_thresh": args.ground_ratio_thresh,
    }
    with open(stats_json, "w") as f:
        json.dump(stats_summary, f, indent=2)

    return (
        f"[DONE] GPU {gpu_id} | {robot_name} | {step} | "
        f"avg_sr={avg_sr:.6f} | diversity={diversity_std.item():.6f} | "
        f"saved={output_pt}"
    )


def process_one_setting_worker(task):
    args, robot_name, step, gpu_id = task
    try:
        print(process_one_setting(args, robot_name, step, gpu_id), flush=True)
    except Exception as exc:
        print(
            f"[WARN] worker failed, skip | GPU {gpu_id} | "
            f"{robot_name} | {step} | {repr(exc)}",
            flush=True,
        )
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim-root",
        default="/mnt/fx/tro_exp/train_0.2B_multi_hand_v6_noise/sim",
    )
    parser.add_argument(
        "--pred-root",
        default="/mnt/fx/tro_exp/train_0.2B_multi_hand_v6_noise/val",
    )
    parser.add_argument("--steps", nargs="+", default=["step_820000"])
    parser.add_argument("--robot-names", nargs="+", default=HAND_NAMES)
    parser.add_argument("--output-name", default="sr_final_surface_pc")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--proximity-thresh", type=float, default=0.003) # 0.004
    parser.add_argument("--min-close-points", type=int, default=48) # 32
    parser.add_argument("--close-ratio-thresh", type=float, default=0.006)
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--ground-penetration-thresh", type=float, default=0.03)  # 0.02
    parser.add_argument("--min-ground-points", type=int, default=8)
    parser.add_argument("--ground-ratio-thresh", type=float, default=0.001)
    parser.add_argument("--cdist-chunk-size", type=int, default=4096)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp

    args = parse_args()
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA device is available.")

    tasks = []
    for step in args.steps:
        for robot_name in args.robot_names:
            gpu_id = len(tasks) % num_gpus
            sim_data_root = os.path.join(args.sim_root, step, robot_name)
            output_path = os.path.join(sim_data_root, f"{args.output_name}.pt")
            tasks.append((args, robot_name, step, gpu_id))

    ctx = mp.get_context("spawn")
    max_workers = min(len(tasks), num_gpus, args.max_workers)
    pending_tasks = list(tasks)
    active_processes = []

    while pending_tasks or active_processes:
        while pending_tasks and len(active_processes) < max_workers:
            task = pending_tasks.pop(0)
            _, robot_name, step, gpu_id = task
            process = ctx.Process(target=process_one_setting_worker, args=(task,))
            process.start()
            active_processes.append((task, process))
            print(f"[START] GPU {gpu_id} | {robot_name} | {step}", flush=True)

        still_active = []
        for task, process in active_processes:
            process.join(timeout=1.0)
            if process.is_alive():
                still_active.append((task, process))
                continue

            _, robot_name, step, gpu_id = task
            if process.exitcode != 0:
                print(
                    f"[WARN] process exited with code {process.exitcode}, continue | "
                    f"GPU {gpu_id} | {robot_name} | {step}",
                    flush=True,
                )
            process.close()

        active_processes = still_active
