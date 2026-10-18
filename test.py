import torch
import argparse
from time import time
from omegaconf import OmegaConf

import os
import sys
import jax
import json
import yourdfpy
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
from accelerate import Accelerator
from accelerate.utils import set_seed
from scipy.spatial.transform import Rotation as R

from utils.tro_hand_model import create_hand_model
from utils.pyroki_ik import PyrokiRetarget

from dataset.TROFilteredDatasetTestV2 import create_dataloader
from model.troe_graph import RobotGraph
from model.vqvae_encoder import VQVAEEncoder


def extract_target_wxyzs(full_transform, target_link_names):
    # (B, N, 3, 3)
    rotmats = np.stack([
        full_transform[name][:, :3, :3].detach().cpu().numpy()
        for name in target_link_names
    ], axis=1)

    B, N = rotmats.shape[:2]

    # scipy Rotation expects (M, 3, 3)
    quat_wxyz = R.from_matrix(
        rotmats.reshape(B * N, 3, 3)
    ).as_quat(scalar_first=True).astype(np.float32)

    return quat_wxyz.reshape(B, N, 4)


def compute_position_error(pred_se3_dict, target_se3_dict):
    errors = {}
    total, count = 0.0, 0

    for link_name, target_T in target_se3_dict.items():
        if link_name not in pred_se3_dict:
            continue

        pred_T = pred_se3_dict[link_name]
        pred_pos = pred_T[..., :3, 3]
        target_pos = target_T[..., :3, 3]

        err = torch.norm(pred_pos - target_pos, dim=-1).mean()

        errors[link_name] = err.item()
        total += err.item()
        count += 1

    avg_error = total / count if count else 0.0
    return errors, avg_error


def compute_orientation_error(pred_se3_dict, target_se3_dict):
    errors = {}
    total, count = 0.0, 0

    for link_name, target_T in target_se3_dict.items():
        if link_name not in pred_se3_dict:
            continue

        pred_T = pred_se3_dict[link_name]

        pred_R = pred_T[..., :3, :3]
        target_R = target_T[..., :3, :3]

        R_diff = pred_R @ target_R.transpose(-1, -2)

        cos_theta = (
            torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1) - 1.0
        ) / 2.0

        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        err = torch.acos(cos_theta).mean()

        errors[link_name] = err.item()
        total += err.item()
        count += 1

    avg_error = total / count if count else 0.0
    return errors, avg_error


def add_aux_link_pose(
    predict_link_pose_dict,
    aux_links,
    urdf_path,
):
    urdf = yourdfpy.URDF.load(urdf_path)
    out = dict(predict_link_pose_dict)

    for aux_name in aux_links:
        aux_joint = next((joint for joint in urdf.robot.joints if joint.child == aux_name), None)
        parent_name = aux_joint.parent
        T_world_parent = out[parent_name]

        T_parent_aux = torch.from_numpy(aux_joint.origin).to(
            device=T_world_parent.device,
            dtype=torch.float32,
        )

        T_world_aux = T_world_parent @ T_parent_aux
        out[aux_name] = T_world_aux
    return out


def finite_pose_mask(pose_dict, names=None):
    if names is None:
        names = list(pose_dict.keys())

    mask = None
    for name in names:
        current = torch.isfinite(pose_dict[name]).flatten(start_dim=1).all(dim=1)
        mask = current if mask is None else (mask & current)
    return mask


def index_pose_dict(pose_dict, mask):
    return {name: pose[mask] for name, pose in pose_dict.items()}


def make_multistart_initial_q_from_root_pose(
    hand_model,
    root_T,
    K,
    pk_to_pyroki_idx,
    device,
    use_canonical_first=True,
):
    root_T = root_T.to(device=device, dtype=torch.float32)
    B = root_T.shape[0]
    dof = hand_model.dof

    lower, upper = hand_model.pk_chain.get_joint_limits()
    lower = torch.as_tensor(lower, dtype=torch.float32, device=device)
    upper = torch.as_tensor(upper, dtype=torch.float32, device=device)

    q = lower.view(1, 1, dof) + torch.rand(
        B, K, dof,
        device=device,
        dtype=torch.float32,
    ) * (upper - lower).view(1, 1, dof)

    # Use one stable canonical seed as the first initialization.
    if use_canonical_first:
        q[:, 0, :] = lower + 0.25 * (upper - lower)

    q[:, :, :3] = root_T[:, None, :3, 3]
    root_R = root_T[:, :3, :3]
    root_rpy = torch.from_numpy(
        R.from_matrix(root_R.detach().cpu().numpy()).as_euler("XYZ")
    ).to(device=device, dtype=torch.float32)
    q[:, :, 3:6] = root_rpy[:, None, :]

    q = q[:, :, pk_to_pyroki_idx]
    return q.reshape(B * K, dof)


def solve_ik_multistart(
    hand_model,
    batch_retarget,
    predict_link_pose_dict,
    predict_link_pose_dict_aux,
    target_links,
    aux_links,
    urdf_path,
    pk_to_pyroki_idx,
    pyroki_to_pk_idx,
    root_link_name,
    num_multistart,
    device,
):
    assert root_link_name in predict_link_pose_dict
    target_pos_list = [predict_link_pose_dict_aux[name][:, :3, 3] for name in target_links]
    target_pos = torch.stack(target_pos_list, dim=1).to(device=device, dtype=torch.float32)

    B, L, _ = target_pos.shape
    root_T = predict_link_pose_dict[root_link_name].to(device=device, dtype=torch.float32)
    initial_q_flat = make_multistart_initial_q_from_root_pose(
        hand_model=hand_model,
        root_T=root_T,
        K=num_multistart,
        pk_to_pyroki_idx=pk_to_pyroki_idx,
        device=device,
        use_canonical_first=True,
    )  # [BK, DOF]

    target_pos_flat = target_pos[:, None].expand(
        B, num_multistart, L, 3
    ).reshape(B * num_multistart, L, 3).contiguous()  # [BK, L, 3]

    predict_q_jnp = batch_retarget(
        initial_q=jnp.array(initial_q_flat.detach().cpu().numpy()),
        target_pos=jnp.array(target_pos_flat.detach().cpu().numpy()),
    )
    jax.block_until_ready(predict_q_jnp)

    predict_q_flat = torch.from_numpy(np.array(predict_q_jnp)).to(
        device=device,
        dtype=torch.float32,
    )

    # Convert from Pyroki joint order back to pytorch_kinematics joint order.
    predict_q_flat = predict_q_flat[:, pyroki_to_pk_idx]

    # Evaluate every candidate by FK.
    ik_link_pose_dict_flat = hand_model.get_link_se3_batch(predict_q_flat)
    ik_link_pose_dict_aux_flat = add_aux_link_pose(
        ik_link_pose_dict_flat,
        aux_links,
        urdf_path,
    )

    # Select the best candidate by average target-link position error.
    candidate_errs = []

    for name in target_links:
        pred_pos = ik_link_pose_dict_aux_flat[name][:, :3, 3]             # [BK, 3]

        target_pos_one_link = predict_link_pose_dict_aux[name][:, :3, 3]  # [B, 3]
        target_pos_one_link = target_pos_one_link[:, None].expand(
            B, num_multistart, 3
        ).reshape(B * num_multistart, 3)                                  # [BK, 3]

        err = torch.norm(pred_pos - target_pos_one_link, dim=-1)          # [BK]
        candidate_errs.append(err)

    candidate_err = torch.stack(candidate_errs, dim=0).mean(dim=0)
    candidate_err = candidate_err.reshape(B, num_multistart)
    best_k = torch.argmin(candidate_err, dim=1)

    predict_q_candidates = predict_q_flat.reshape(B, num_multistart, -1)
    batch_idx = torch.arange(B, device=device)
    best_predict_q = predict_q_candidates[batch_idx, best_k]

    return best_predict_q


@torch.no_grad()
def test(config):

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    root_link_name_dict = {
        "allegro_hand": "base_link",
        "barrett_hand": "base_link",
        "shadow_hand": "palm",
        "leaphand": "palm_lower",
        "dclaw_gripper": "base_link",
        "ability_hand": "base",
        "xhand": "right_hand_link",
        "schunk_hand": "right_hand_e1",
        "wuji_hand": "palm_link"
    }

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print("Device:", device)

    print("Building dataloader...")
    dataloader = create_dataloader(config.dataset)

    #### Compile Jax for Pyroki ####
    robot_name = config.test.embodiment
    hand_model = create_hand_model(robot_name, device=device)

    urdf_path_dict = json.load(
        open("data/data_urdf/robot/urdf_assets_meta.json", "r")
    )
    urdf_path = urdf_path_dict["urdf_path"][robot_name]

    aux_links = json.load(
        open("data/data_urdf/robot/auxiliary_links.json", "r")
    )[robot_name]

    target_links = list(hand_model.link_names) + aux_links

    ik_solver = PyrokiRetarget(urdf_path, target_links)
    batch_retarget = jax.jit(ik_solver.solve_retarget)

    # Joint Alignment
    pyroki_joint_names = ik_solver.robot.joints.actuated_names
    pk_joint_names = hand_model.pk_chain.get_joint_parameter_names()

    pk_to_pyroki_idx = [
        pk_joint_names.index(name)
        for name in pyroki_joint_names
    ]
    pyroki_to_pk_idx = [
        pyroki_joint_names.index(name)
        for name in pk_joint_names
    ]

    # Multi-start settings
    num_multistart = int(config.test.get("num_multistart", 8))
    root_link_name = root_link_name_dict[robot_name]

    print(f"Using num_multistart = {num_multistart}")
    print(f"Using root_link_name = {root_link_name}")

    # VQVAE encoder
    vqvae_encoder = VQVAEEncoder(**config.vqvae)
    vqvae_encoder.eval()
    vqvae_encoder.to(accelerator.device)

    print("Building model...")
    model = RobotGraph(**config.model).to(device)

    ckpt = torch.load(config.test.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)

    print(f"Model loaded from {config.test.ckpt}.")

    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    print("Start Evaluation")

    vis_info = []
    test_batch_idx = 0
    avg_pos_error = 0.0
    avg_rot_error = 0.0
    valid_batch_count = 0
    total_skipped_samples = 0

    for batch in dataloader:

        num_object, sample_per_object = batch["robot_id"].shape
        meta_names = list(batch["meta_name"])

        for key in ["robot_id", "env_pc", "object_pc"]:
            x = batch[key]
            batch[key] = x.reshape(
                num_object * sample_per_object,
                *x.shape[2:],
            )

        # Data preprocessing
        batch = vqvae_encoder(batch)

        # Inference
        with accelerator.autocast():
            predict_link_pose_list = model.inference(batch)

        predict_link_pose_dict = predict_link_pose_list[-1]
        predict_link_pose_dict_aux = add_aux_link_pose(
            predict_link_pose_dict,
            aux_links,
            urdf_path,
        )

        root_valid_mask = finite_pose_mask(predict_link_pose_dict, [root_link_name])
        target_valid_mask = finite_pose_mask(predict_link_pose_dict_aux, target_links)
        valid_mask = root_valid_mask & target_valid_mask
        skipped_samples = int((~valid_mask).sum().item())
        total_skipped_samples += skipped_samples

        if skipped_samples > 0:
            skipped_ids = (~valid_mask).nonzero().flatten().detach().cpu().tolist()
            skipped_desc = [
                f"{idx}:{meta_names[idx // sample_per_object]}#{idx % sample_per_object}"
                for idx in skipped_ids[:10]
            ]
            suffix = " ..." if len(skipped_ids) > 10 else ""
            print(
                f"Skip {skipped_samples}/{valid_mask.numel()} samples with non-finite predicted poses: "
                + ", ".join(skipped_desc)
                + suffix
            )

        if not valid_mask.any():
            test_batch_idx += 1
            print(
                f"Test batch {test_batch_idx} / {len(dataloader)} | "
                f"all {valid_mask.numel()} samples skipped due to non-finite predicted poses"
            )
            continue

        predict_link_pose_dict_valid = index_pose_dict(predict_link_pose_dict, valid_mask)
        predict_link_pose_dict_aux_valid = index_pose_dict(predict_link_pose_dict_aux, valid_mask)
        
        # Multi-start Pyroki IK
        predict_q = solve_ik_multistart(
            hand_model=hand_model,
            batch_retarget=batch_retarget,
            predict_link_pose_dict=predict_link_pose_dict_valid,
            predict_link_pose_dict_aux=predict_link_pose_dict_aux_valid,
            target_links=target_links,
            aux_links=aux_links,
            urdf_path=urdf_path,
            pk_to_pyroki_idx=pk_to_pyroki_idx,
            pyroki_to_pk_idx=pyroki_to_pk_idx,
            root_link_name=root_link_name,
            num_multistart=num_multistart,
            device=device,
        )

        # IK error estimation
        ik_link_pose_dict = hand_model.get_link_se3_batch(predict_q)

        _, pos_error = compute_position_error(
            ik_link_pose_dict,
            predict_link_pose_dict_valid,
        )

        _, rot_error = compute_orientation_error(
            ik_link_pose_dict,
            predict_link_pose_dict_valid,
        )

        avg_pos_error += pos_error
        avg_rot_error += rot_error
        valid_batch_count += 1

        valid_indices = valid_mask.nonzero().flatten()
        compact_index = torch.full(
            (valid_mask.numel(),),
            -1,
            dtype=torch.long,
            device=valid_mask.device,
        )
        compact_index[valid_indices] = torch.arange(
            valid_indices.numel(),
            dtype=torch.long,
            device=valid_mask.device,
        )

        for meta_id in range(num_object):
            start = meta_id * sample_per_object
            end = (meta_id + 1) * sample_per_object
            object_valid_mask = valid_mask[start:end]
            if not object_valid_mask.any():
                continue
            object_valid_global = valid_indices[
                (valid_indices >= start) & (valid_indices < end)
            ]
            object_valid_compact = compact_index[object_valid_global]

            vis_info.append({
                "predict_q": predict_q[object_valid_compact].detach().cpu(),
                "object_pc": batch["object_pc"][start:end][object_valid_mask].detach().cpu(),
                "env_pc": batch["env_pc"][start:end][object_valid_mask].detach().cpu(),
                "meta_name": meta_names[meta_id],
                "valid_scene_ids": object_valid_mask.nonzero().flatten().detach().cpu(),
                "diffusion": {
                    k: v[start:end][object_valid_mask].detach().cpu()
                    for k, v in predict_link_pose_dict.items()
                },
            })

        test_batch_idx += 1

        print(
            f"Test batch {test_batch_idx} / {len(dataloader)} | "
            f"pos_error = {pos_error:.6f} m | "
            f"rot_error = {rot_error * 180.0 / 3.1415926:.4f} deg"
        )

    denom = max(valid_batch_count, 1)
    print(f"Total skipped samples: {total_skipped_samples}.")
    print(f"Total trans IK error: {avg_pos_error / denom} meters.")
    print(f"Total rot IK error: {avg_rot_error / denom * 180.0 / 3.1415926} degrees.")

    os.makedirs(os.path.dirname(config.test.save_path), exist_ok=True)
    torch.save(vis_info, config.test.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    overrides = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, overrides)

    test(config)
