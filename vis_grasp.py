"""
Close-up visualization for TRO comparison scenes.

Unlike vis_compare.py, this viewer crops the environment and surrounding object
meshes around the target object and predicted hand. This makes screenshots for
papers much easier to read when the full scene is large.
"""

import argparse
import json
import os
import random
import sys
import time
from functools import lru_cache

import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from utils.tro_hand_model_with_mount import create_hand_model_with_mount

def get_se3_from_trans_and_rpy(trans, rpy):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = trans
    R_mat = R.from_euler('XYZ', rpy).as_matrix()
    T[:3, :3] = R_mat
    return T

def add_hand_mount(q, T_base_palm):

    trans, rpy, joint_vals = q[:3], q[3:6], q[6:]
    T_world_palm = get_se3_from_trans_and_rpy(trans, rpy)
    T_world_base = T_world_palm @ np.linalg.inv(T_base_palm)

    trans_base = T_world_base[:3, 3]
    rpy_base = R.from_matrix(T_world_base[:3, :3]).as_euler('XYZ')
    q_aligned = np.concatenate([trans_base, rpy_base, joint_vals], axis=0)
    return q_aligned

align_matrixes = torch.load("data/data_urdf/robot/align_ft_shadow.pt")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

OBJECT_URDF_ROOT = "/mnt/fx/TRO-1B-Dataset/object_urdf"
EXP_ROOT = "/mnt/fx/tro_exp/train_0.2B_multi_hand_v6_noise"

DATASET_ROOTS = {
    "val": "/mnt/fx/TRO-1B-Dataset/TRO-val-dataset",
}

TRO_PRED_ROOTS = {
    "val": "val",
}

METHODS = {
    "TRO": {
        "color": (58, 128, 255),      # polished blue
        "opacity": 0.8,
    },
    "AnyDexGrasp": {
        "hand": "allegro_hand",
        "val_root": "/mnt/fx/AnyDexGrasp_exp/sim/filtered/allegro_hand",
        "color": (245, 166, 75),      # elegant orange
        "opacity": 0.86,
    },
    "DexGraspNet2": {
        "hand": "leaphand",
        "val_root": "/mnt/fx/DexGraspNet2_exp/sim/wo_filtered/leaphand",
        "color": (65, 190, 145),      # elegant green
        "opacity": 0.86,
    },
}

TARGET_COLOR = (236, 72, 104)          # rose red
CLUTTER_COLOR = (196, 118, 94)         # muted terracotta
ENV_COLOR = (232, 214, 185)            # warm cream
BOX_ENV_COLOR = (176, 194, 210)        # cool gray-blue for box walls
SHELF_ENV_COLOR = (190, 152, 108)      # warm wood-like shelf color
POINT_CLOUD_COLOR = (240, 226, 202)    # light cream

def pose_to_matrix(pose):
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose[:3]
    T[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
    return T


def q_to_tensor(q):
    return torch.as_tensor(q, dtype=torch.float32)


@lru_cache(maxsize=512)
def load_object_mesh(object_name):
    mesh_path = os.path.join(OBJECT_URDF_ROOT, f"{object_name}.stl")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        )
    return mesh


def transformed_object_mesh(object_name, pose):
    mesh = load_object_mesh(object_name).copy()
    mesh.apply_transform(pose_to_matrix(pose))
    return mesh


def add_mesh(server, name, mesh, color, opacity=1.0, visible=True):
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    return server.scene.add_mesh_simple(
        name=name,
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        color=color,
        opacity=opacity,
        visible=visible,
    )


def translated_mesh(mesh, offset):
    if mesh is None:
        return None
    mesh = mesh.copy()
    mesh.apply_translation(np.asarray(offset, dtype=np.float32))
    return mesh


def mesh_bounds_center(meshes):
    meshes = [m for m in meshes if m is not None and len(m.vertices) > 0]
    if not meshes:
        return np.zeros(3, dtype=np.float32)
    mins = np.stack([m.bounds[0] for m in meshes], axis=0).min(axis=0)
    maxs = np.stack([m.bounds[1] for m in meshes], axis=0).max(axis=0)
    return ((mins + maxs) * 0.5).astype(np.float32)


def as_numpy_points(points):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    return np.asarray(points, dtype=np.float32)


def make_box(center, extents):
    return trimesh.creation.box(
        extents=extents,
        transform=trimesh.transformations.translation_matrix(center),
    )


def combine_meshes(meshes):
    meshes = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def crop_mesh_xy(mesh, center_xy, radius, z_min=-0.05, z_max=1.5):
    """Keep faces whose centroids fall inside an XY square crop."""
    if mesh is None or len(mesh.faces) == 0:
        return None
    center_xy = np.asarray(center_xy, dtype=np.float32)
    tri = np.asarray(mesh.triangles_center)
    mask = (
        (np.abs(tri[:, 0] - center_xy[0]) <= radius)
        & (np.abs(tri[:, 1] - center_xy[1]) <= radius)
        & (tri[:, 2] >= z_min)
        & (tri[:, 2] <= z_max)
    )
    if not np.any(mask):
        return None
    cropped = mesh.submesh([np.flatnonzero(mask)], append=True, repair=False)
    if isinstance(cropped, list):
        cropped = combine_meshes(cropped)
    return cropped


def infer_env_info_from_pc(env_pc):
    pc = as_numpy_points(env_pc)
    mins = pc.min(axis=0)
    maxs = pc.max(axis=0)
    span = maxs - mins
    eps = 0.012
    if span[2] < 0.05:
        return {"env_type": "tabletop_dense"}
    near_top = np.mean(np.abs(pc[:, 2] - maxs[2]) < eps)
    near_y_min = np.mean(np.abs(pc[:, 1] - mins[1]) < eps)
    near_y_max = np.mean(np.abs(pc[:, 1] - maxs[1]) < eps)
    if near_top > 0.12 and max(near_y_min, near_y_max) > min(near_y_min, near_y_max) * 1.8:
        return {"env_type": "shelf_dense", "z_size": float(maxs[2])}
    return {"env_type": "box_dense", "z_size": float(maxs[2])}


def resolved_env_info(env_info, env_pc):
    if env_info:
        return env_info
    return infer_env_info_from_pc(env_pc)


def env_color_from_type(env_type):
    if env_type in ("box", "box_dense"):
        return BOX_ENV_COLOR
    if env_type in ("shelf", "shelf_dense"):
        return SHELF_ENV_COLOR
    return ENV_COLOR


def cropped_environment_mesh(env_info, env_pc, center_xy, radius):
    env_info = resolved_env_info(env_info, env_pc)

    env_type = env_info["env_type"]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r = float(radius)
    t = 0.008

    pc = as_numpy_points(env_pc)
    mins, maxs = pc.min(axis=0), pc.max(axis=0)

    # Clamp crop window to the actual environment XY range.
    x0 = max(cx - r, float(mins[0]))
    x1 = min(cx + r, float(maxs[0]))
    y0 = max(cy - r, float(mins[1]))
    y1 = min(cy + r, float(maxs[1]))

    # If crop window does not overlap with env bounds, fall back to a tiny patch.
    if x1 <= x0 or y1 <= y0:
        x0, x1 = cx - 0.01, cx + 0.01
        y0, y1 = cy - 0.01, cy + 0.01

    floor_cx = 0.5 * (x0 + x1)
    floor_cy = 0.5 * (y0 + y1)
    floor_sx = x1 - x0
    floor_sy = y1 - y0

    meshes = [make_box([floor_cx, floor_cy, -t / 2], [floor_sx, floor_sy, t])]

    if env_type in ("tabletop", "tabletop_dense"):
        return combine_meshes(meshes)

    if env_type in ("box", "box_dense"):
        z = float(env_info.get("z_size", maxs[2]))

        if abs(maxs[0] - cx) <= r + 0.02:
            meshes.append(make_box([maxs[0] + t / 2, floor_cy, z / 2], [t, floor_sy, z]))
        if abs(mins[0] - cx) <= r + 0.02:
            meshes.append(make_box([mins[0] - t / 2, floor_cy, z / 2], [t, floor_sy, z]))
        if abs(maxs[1] - cy) <= r + 0.02:
            meshes.append(make_box([floor_cx, maxs[1] + t / 2, z / 2], [floor_sx, t, z]))
        if abs(mins[1] - cy) <= r + 0.02:
            meshes.append(make_box([floor_cx, mins[1] - t / 2, z / 2], [floor_sx, t, z]))

        return combine_meshes(meshes)

    if env_type in ("shelf", "shelf_dense"):
        z = float(env_info.get("z_size", maxs[2]))

        # Top board should also be clamped to actual env bounds.
        meshes.append(make_box([floor_cx, floor_cy, z + t / 2], [floor_sx, floor_sy, t]))

        if abs(maxs[0] - cx) <= r + 0.02:
            meshes.append(make_box([maxs[0] + t / 2, floor_cy, z / 2], [t, floor_sy, z]))
        if abs(mins[0] - cx) <= r + 0.02:
            meshes.append(make_box([mins[0] - t / 2, floor_cy, z / 2], [t, floor_sy, z]))
        if abs(mins[1] - cy) <= r + 0.02:
            meshes.append(make_box([floor_cx, mins[1] - t / 2, z / 2], [floor_sx, t, z]))

        return combine_meshes(meshes)

    raise ValueError(f"Unsupported env_type: {env_type}")


def crop_points_xy(points, center_xy, radius, max_points):
    points = as_numpy_points(points)
    d = np.max(np.abs(points[:, :2] - np.asarray(center_xy)[None, :2]), axis=1)
    cropped = points[d <= radius]
    if cropped.shape[0] == 0:
        cropped = points[np.argsort(d)[:max_points]]
    if cropped.shape[0] > max_points:
        idx = np.linspace(0, cropped.shape[0] - 1, max_points).astype(np.int64)
        cropped = cropped[idx]
    return cropped


def load_scene(dataset, hand_name, object_name):
    dataset_root = DATASET_ROOTS[dataset]
    path = os.path.join(dataset_root, hand_name, f"{object_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache(maxsize=16)
def load_tro_infos(dataset, hand_name, step):
    path = os.path.join(ROOT_DIR, "ckpt", "test_meta", f"test_{hand_name}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_tro_info(dataset, hand_name, step, object_name):
    for info in load_tro_infos(dataset, hand_name, step):
        if info["meta_name"] == object_name:
            return info
    raise KeyError(f"{object_name} not found in TRO predictions")


def load_tro_q(dataset, hand_name, step, object_name, sample_idx):
    return q_to_tensor(load_tro_info(dataset, hand_name, step, object_name)["predict_q"][sample_idx])


def baseline_methods_for_mode(mode, hand_name):
    if mode != "with-baseline":
        return []
    if hand_name == "allegro_hand":
        return ["AnyDexGrasp"]
    if hand_name == "leaphand":
        return ["DexGraspNet2"]
    return []


def baseline_root(method, dataset):
    key = f"{dataset}_root"
    return METHODS[method][key]


def load_baseline_q(method, dataset, object_name, sample_idx):
    root = baseline_root(method, dataset)
    path = os.path.join(root, object_name, "val_data.json")
    if not os.path.exists(path):
        return None, f"missing {path}"
    with open(path, "r") as f:
        data = json.load(f)
    joints = data.get("joint", [])
    valid_mask = data.get("valid_mask", [True] * len(joints))
    if sample_idx >= len(joints):
        return None, f"sample {sample_idx} out of range"
    if sample_idx < len(valid_mask) and not valid_mask[sample_idx]:
        return None, "invalid baseline sample"
    return q_to_tensor(joints[sample_idx]), None


def list_candidate_objects(dataset, hand_name, mode, step):
    dataset_root = DATASET_ROOTS[dataset]
    hand_dir = os.path.join(dataset_root, hand_name)
    candidates = {name[:-3] for name in os.listdir(hand_dir) if name.endswith(".pt")}
    candidates &= {info["meta_name"] for info in load_tro_infos(dataset, hand_name, step)}
    for method in baseline_methods_for_mode(mode, hand_name):
        root = baseline_root(method, dataset)
        if os.path.isdir(root):
            candidates &= {name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))}
    return sorted(candidates)


def sample_random_scene(dataset, hand_name, mode, step, max_tries=200):
    candidates = list_candidate_objects(dataset, hand_name, mode, step)
    if not candidates:
        raise RuntimeError(f"No candidate objects for dataset={dataset}, hand={hand_name}, mode={mode}")
    baseline_methods = baseline_methods_for_mode(mode, hand_name)
    for _ in range(max_tries):
        object_name = random.choice(candidates)
        scene = load_scene(dataset, hand_name, object_name)
        sample_idx = random.randrange(len(scene["meta"]))
        tro_info = load_tro_info(dataset, hand_name, step, object_name)
        if sample_idx >= len(tro_info["predict_q"]):
            continue
        ok = True
        for method in baseline_methods:
            q, _ = load_baseline_q(method, dataset, object_name, sample_idx)
            ok = ok and q is not None
        if ok:
            return object_name, sample_idx
    object_name = random.choice(candidates)
    scene = load_scene(dataset, hand_name, object_name)
    return object_name, random.randrange(len(scene["meta"]))


@lru_cache(maxsize=16)
def get_hand_model(hand_name):
    return create_hand_model_with_mount(hand_name, device=torch.device("cpu"))


def hand_mesh(hand_name, q):
    return get_hand_model(hand_name).get_trimesh_q(q)["visual"]


def mounted_hand_mesh(hand_name, q):
    q = add_hand_mount(q, align_matrixes[hand_name])
    q = torch.from_numpy(q).to(dtype=torch.float32)
    return hand_mesh(hand_name, q)


def add_prediction(server, method, hand_name, mesh, recenter_offset, visible=True, show_label=False):
    mesh = translated_mesh(mesh, recenter_offset)
    handle = add_mesh(
        server,
        f"/predictions/{method}/{hand_name}",
        mesh,
        METHODS[method]["color"],
        opacity=METHODS[method]["opacity"],
        visible=visible,
    )
    handles = [handle]
    if show_label:
        center = mesh.bounds.mean(axis=0)
        handles.append(
            server.scene.add_label(
                f"/labels/{method}",
                f"{method} ({hand_name})",
                position=center + np.array([0.0, 0.0, 0.08]),
                font_size_mode="screen",
                font_screen_scale=0.9,
                depth_test=False,
                visible=visible,
            )
        )
    return mesh, tuple(h for h in handles if h is not None)


def build_scene_closeup(meta, env_pc, object_name, crop_center_xy, crop_radius, args):
    env_info = resolved_env_info(meta.get("challenge_env_info"), env_pc)
    env_mesh = cropped_environment_mesh(env_info, env_pc, crop_center_xy, crop_radius)
    env_color = env_color_from_type(env_info["env_type"])
    target_mesh = transformed_object_mesh(object_name, meta["target"])

    clutter_meshes = []
    clutter_idx = 0
    for name, pose in meta.items():
        if name in {"target", "challenge_env_info"}:
            continue
        mesh = transformed_object_mesh(name, pose)
        cropped = crop_mesh_xy(mesh, crop_center_xy, crop_radius, z_min=-0.02, z_max=args.crop_z_max)
        if cropped is not None:
            clutter_meshes.append((clutter_idx, name, cropped))
            clutter_idx += 1

    cropped_pc = None
    if args.show_env_points:
        cropped_pc = crop_points_xy(env_pc, crop_center_xy, crop_radius, args.max_env_points)

    return env_mesh, env_color, target_mesh, clutter_meshes, cropped_pc


def add_scene_closeup(server, env_mesh, env_color, target_mesh, clutter_meshes, cropped_pc, recenter_offset, args):
    add_mesh(server, "/scene/env_crop", translated_mesh(env_mesh, recenter_offset), env_color, opacity=args.env_opacity)
    add_mesh(server, "/objects/target", translated_mesh(target_mesh, recenter_offset), TARGET_COLOR, opacity=1.0)

    for clutter_idx, name, mesh in clutter_meshes:
        add_mesh(
            server,
            f"/objects/clutter/{clutter_idx:03d}_{name}",
            translated_mesh(mesh, recenter_offset),
            CLUTTER_COLOR,
            opacity=args.clutter_opacity,
        )

    if cropped_pc is not None:
        server.scene.add_point_cloud(
            "/pointclouds/env_crop",
            cropped_pc + np.asarray(recenter_offset, dtype=np.float32)[None, :],
            point_size=args.point_size,
            point_shape="circle",
            colors=POINT_CLOUD_COLOR,
            visible=True,
        )


def add_visibility_controls(server, prediction_handles):
    checkboxes = {}
    with server.gui.add_folder("Prediction visibility"):
        for method in ("TRO", "AnyDexGrasp", "DexGraspNet2"):
            checkbox = server.gui.add_checkbox(method, initial_value=True)
            checkboxes[method] = checkbox

            @checkbox.on_update
            def _(_, method=method, checkbox=checkbox):
                visible = bool(checkbox.value)
                for handle in prediction_handles.get(method, ()):
                    handle.visible = visible
    return checkboxes


def render_scene(server, args, object_name, sample_idx, checkbox_handles=None):
    scene = load_scene(args.dataset, args.tro_hand, object_name)
    meta = scene["meta"][sample_idx]
    env_pc = scene["env_pc"][sample_idx]
    tro_q = load_tro_q(args.dataset, args.tro_hand, args.step, object_name, sample_idx)

    tro_mesh = mounted_hand_mesh(args.tro_hand, tro_q)
    target_mesh = transformed_object_mesh(object_name, meta["target"])
    target_center = target_mesh.bounds.mean(axis=0)
    hand_center = tro_mesh.bounds.mean(axis=0)
    crop_center_xy = (
        np.asarray(target_center[:2]) * (1.0 - args.hand_center_weight)
        + np.asarray(hand_center[:2]) * args.hand_center_weight
    )

    env_mesh, env_color, target_mesh, clutter_meshes, cropped_pc = build_scene_closeup(
        meta,
        env_pc,
        object_name,
        crop_center_xy,
        args.crop_radius,
        args,
    )

    prediction_meshes = {"TRO": (args.tro_hand, tro_mesh)}
    for method in baseline_methods_for_mode(args.mode, args.tro_hand):
        q, reason = load_baseline_q(method, args.dataset, object_name, sample_idx)
        if q is None:
            print(f"[SKIP] {method}: {reason}", flush=True)
            continue
        prediction_meshes[method] = (METHODS[method]["hand"], mounted_hand_mesh(METHODS[method]["hand"], q))

    remaining_meshes = [env_mesh, target_mesh]
    remaining_meshes.extend(mesh for _, _, mesh in clutter_meshes)
    remaining_meshes.extend(mesh for _, mesh in prediction_meshes.values())
    recenter_center = mesh_bounds_center(remaining_meshes)
    recenter_offset = -recenter_center

    server.scene.reset()
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False

    add_scene_closeup(server, env_mesh, env_color, target_mesh, clutter_meshes, cropped_pc, recenter_offset, args)

    prediction_handles = {}
    METHODS["TRO"]["hand"] = args.tro_hand
    _, prediction_handles["TRO"] = add_prediction(
        server,
        "TRO",
        args.tro_hand,
        prediction_meshes["TRO"][1],
        recenter_offset,
        visible=checkbox_handles is None or bool(checkbox_handles["TRO"].value),
        show_label=args.show_labels,
    )

    for method, (hand_name, mesh) in prediction_meshes.items():
        if method == "TRO":
            continue
        _, prediction_handles[method] = add_prediction(
            server,
            method,
            hand_name,
            mesh,
            recenter_offset,
            visible=checkbox_handles is None or bool(checkbox_handles[method].value),
            show_label=args.show_labels,
        )

    server.scene.add_frame(
        "/crop_center",
        position=np.array(
            [crop_center_xy[0], crop_center_xy[1], target_center[2]],
            dtype=np.float32,
        ) + recenter_offset,
        axes_length=0.025,
        axes_radius=0.0015,
        visible=args.show_crop_center,
    )

    print(
        f"[scene] dataset={args.dataset} object={object_name} sample_idx={sample_idx} "
        f"crop_center=({crop_center_xy[0]:.3f}, {crop_center_xy[1]:.3f}) "
        f"recenter=({recenter_center[0]:.3f}, {recenter_center[1]:.3f}, {recenter_center[2]:.3f}) "
        f"radius={args.crop_radius:.3f}",
        flush=True,
    )
    return prediction_handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["val"], default="val")
    parser.add_argument("--tro-hand", default="xhand")
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--step", default="step_820000")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--mode", choices=["tro-only", "with-baseline"], default="with-baseline")
    parser.add_argument("--crop-radius", type=float, default=0.40)
    parser.add_argument("--crop-z-max", type=float, default=0.8)
    parser.add_argument("--hand-center-weight", type=float, default=0.25)
    parser.add_argument("--env-opacity", type=float, default=0.75)
    parser.add_argument("--clutter-opacity", type=float, default=0.75)
    parser.add_argument("--show-env-points", action="store_true")
    parser.add_argument("--max-env-points", type=int, default=1800)
    parser.add_argument("--point-size", type=float, default=0.006)
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--show-crop-center", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.crop_radius <= 0:
        parser.error("--crop-radius must be positive")
    if not (0.0 <= args.hand_center_weight <= 1.0):
        parser.error("--hand-center-weight must be in [0, 1]")
    if args.seed is not None:
        random.seed(args.seed)

    server = viser.ViserServer(port=args.port)
    prediction_handles = {}
    checkbox_handles = add_visibility_controls(server, prediction_handles)

    if args.object_name is None:
        object_name, sample_idx = sample_random_scene(args.dataset, args.tro_hand, args.mode, args.step)
    else:
        object_name = args.object_name
        if args.sample_idx is None:
            scene = load_scene(args.dataset, args.tro_hand, object_name)
            tro_info = load_tro_info(args.dataset, args.tro_hand, args.step, object_name)
            sample_idx = random.randrange(min(len(scene["meta"]), len(tro_info["predict_q"])))
        else:
            sample_idx = args.sample_idx

    prediction_handles.update(render_scene(server, args, object_name, sample_idx, checkbox_handles))

    with server.gui.add_folder("Scene"):
        random_button = server.gui.add_button("Random sample")

        @random_button.on_click
        def _(_):
            prediction_handles.clear()
            object_name, sample_idx = sample_random_scene(args.dataset, args.tro_hand, args.mode, args.step)
            prediction_handles.update(render_scene(server, args, object_name, sample_idx, checkbox_handles))

    print(f"[viser] http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
