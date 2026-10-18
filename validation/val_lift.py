import os
import json
import sys
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


def validate_isaac(
    robot_name,
    bsz_target_object_names,
    bsz_object_urdf_roots,
    bsz_val_data_paths,
    bsz_meta_names,
    gpu_id,
):
    tmp_dir = os.path.join(ROOT_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    result_path = os.path.join(
        tmp_dir,
        f"res_{robot_name}_{os.getpid()}_{uuid.uuid4().hex}.pt"
    )

    args = [
        'python', '-u',
        os.path.join(ROOT_DIR, 'validation/isaac_main.py'),
        '--robot_name', robot_name,
        '--bsz_target_object_names',
    ] + bsz_target_object_names + [
        '--bsz_object_urdf_roots',
    ] + bsz_object_urdf_roots + [
        '--bsz_val_data_paths',
    ] + bsz_val_data_paths + [
        '--result_path', result_path,
        '--bsz_meta_names',
    ] + bsz_meta_names
    

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["JAX_PLATFORM_NAME"] = "cpu"

    ret = subprocess.run(args, capture_output=True, text=True, env=env)

    if ret.returncode != 0:
        print("isaac_main failed:", ret.returncode, flush=True)
        print("STDOUT:\n", ret.stdout, flush=True)
        print("STDERR:\n", ret.stderr, flush=True)
        raise RuntimeError("isaac_main crashed or exited with error")

    if not os.path.exists(result_path):
        raise RuntimeError(f"Result file not found: {result_path}")

    save_data = torch.load(result_path, weights_only=False)
    success = save_data['success']
    os.remove(result_path)
    return success


def process_one_setting(args):
    robot_name, step, gpu_id = args

    sim_data_root = f"/mnt/fx/tro_exp/train_0.2B_multi_hand_v6_noise/sim/{step}/{robot_name}"
    sr_dict = {}
    success_dict = {}
    bsz = 50
    meta_names = [file for file in os.listdir(sim_data_root) if "sr" not in file]

    for batch_start in range(0, len(meta_names), bsz):
        batch_meta_names = meta_names[batch_start: batch_start + bsz]
 
        bsz_target_object_names = []
        bsz_object_urdf_roots = []
        bsz_val_data_paths = []
        bsz_meta_names = []
 
        for meta_name in batch_meta_names:

            target_object_name = meta_name
            object_urdf_root = f"{sim_data_root}/{meta_name}/object_urdf"
            val_data_path = f"{sim_data_root}/{meta_name}/val_data.json"
 
            bsz_target_object_names.append(target_object_name)
            bsz_object_urdf_roots.append(object_urdf_root)
            bsz_val_data_paths.append(val_data_path)
            bsz_meta_names.append(meta_name)

        batch_success_dict = validate_isaac(
            robot_name=robot_name,
            bsz_target_object_names=bsz_target_object_names,
            bsz_object_urdf_roots=bsz_object_urdf_roots,
            bsz_val_data_paths=bsz_val_data_paths,
            bsz_meta_names=bsz_meta_names,
            gpu_id=gpu_id,
        )
        for meta_name, success in batch_success_dict.items():
            sr = sum(success) / len(success)
            sr_dict[meta_name] = sr
            success_dict[meta_name] = success

        cur_success_rate = sum(sr_dict.values()) / len(sr_dict) if len(sr_dict) > 0 else 0.0
        processed_cnt = min(batch_start + len(batch_meta_names), len(meta_names))
        print(
            f"[GPU {gpu_id}] Hand {robot_name} | Step {step} | "
            f"Processed {processed_cnt}/{len(meta_names)} | "
            f"Current Success Rate: {cur_success_rate:.4f} | "
            f"Current Batch: {len(batch_meta_names)} objects",
            flush=True
        )

    avg_sr = sum(sr_dict.values()) / len(sr_dict) if len(sr_dict) > 0 else 0.0
    sr_dict["avg_sr"] = avg_sr
    with open(f"{sim_data_root}/sr_lift.json", 'w') as f:
        json.dump(sr_dict, f, indent=2)
    torch.save(success_dict, f"{sim_data_root}/sr_lift.pt")
    return f"[DONE] GPU {gpu_id} | {robot_name} | {step} | avg_sr={avg_sr:.6f}"


if __name__ == "__main__":

    robot_names = ["allegro_hand", "barrett_hand", "shadow_hand", "leaphand", "xhand", "dclaw_gripper", "schunk_hand", "ability_hand"]
    steps = ["step_820000"]
    tasks = []
    gpu_id = 0
    for step in steps:
        for robot_name in robot_names:
            tasks.append((robot_name, step, gpu_id))
            gpu_id += 1

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_one_setting, task) for task in tasks]

        for future in as_completed(futures):
            print(future.result(), flush=True)