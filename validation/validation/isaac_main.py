import os
import sys
import json
import argparse
import warnings
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from validation.isaac_validator_lift import IsaacValidator  # IsaacGym must be imported before PyTorch
from utils.tro_hand_model import create_hand_model
import torch

def isaac_main(
    robot_name: str,
    bsz_target_object_names: list,
    bsz_object_urdf_roots: list,
    bsz_val_data_paths: list,
    bsz_meta_names: list,
    gpu: int = 0,
    use_gui: bool = False
):
    if use_gui:  # for unknown reason otherwise will segmentation fault :(
        gpu = 0

    hand = create_hand_model(robot_name)
    joint_orders = hand.get_joint_orders()
    urdf_assets_meta = json.load(open('data/data_urdf/robot/urdf_assets_meta.json'))
    robot_urdf_path = urdf_assets_meta['urdf_path'][robot_name]

    palm_normal_local_dict = {
        "leaphand": (0.0, 0.0, -1.0),
        "allegro_hand": (1.0, 0.0, 0.0),
        "barrett_hand": (0.0, 0.0, 1.0),
        "shadow_hand": (1.0, 0.0, 0.0),
        "schunk_hand": (0.0, 1.0, 0.0),
        "xhand": (0.0, -1.0, 0.0),
        "dclaw_gripper": (1.0, 0.0, 0.0),
        "ability_hand": (0.0, 1.0, 0.0),
        "wuji_hand": (1.0, 0.0, 0.0)
    }

    simulator = IsaacValidator(
        robot_name=robot_name,
        bsz_val_data_paths=bsz_val_data_paths,
        bsz_target_object_names=bsz_target_object_names,
        bsz_meta_names=bsz_meta_names,
        joint_orders=joint_orders,
        gpu=gpu, 
        use_gui=use_gui,
        palm_normal_local=palm_normal_local_dict[robot_name]
    )
    print("[Isaac] IsaacValidator is created.")

    simulator.set_asset(
        robot_path=os.path.dirname(robot_urdf_path),
        robot_file=os.path.basename(robot_urdf_path),
        object_paths=bsz_object_urdf_roots,
        object_files=[f"{object_name}.urdf" for object_name in bsz_target_object_names]
    )
    simulator.create_envs()
    print("[Isaac] IsaacValidator preparation is done.")

    q_batch = torch.cat([
        torch.tensor(json.load(open(val_data_path))['joint'], dtype=torch.float32) for val_data_path in bsz_val_data_paths
    ], dim=0)

    # q_batch_pd = q_batch.clone()
    # q_batch_pd[:, 6:] = torch.cat([
    #     torch.tensor(json.load(open(val_data_path))['target_joint'], dtype=torch.float32) for val_data_path in bsz_val_data_paths
    # ], dim=0)

    simulator.set_actor_pose_dof(q_batch)
    success = simulator.run_sim()

    simulator.destroy()
    return success

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', type=str, required=True)
    parser.add_argument('--bsz_target_object_names', nargs='+', required=True)
    parser.add_argument('--bsz_object_urdf_roots', nargs='+', required=True)
    parser.add_argument('--bsz_val_data_paths', nargs='+', required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--bsz_meta_names', nargs='+', required=True)
    args = parser.parse_args()

    success = isaac_main(
        robot_name=args.robot_name,
        bsz_target_object_names=args.bsz_target_object_names,
        bsz_object_urdf_roots=args.bsz_object_urdf_roots,
        bsz_val_data_paths=args.bsz_val_data_paths,
        bsz_meta_names=args.bsz_meta_names
    )

    save_data = {'success': success}
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    torch.save(save_data, args.result_path)
    os._exit(0)


