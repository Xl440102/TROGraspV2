import os
import json
import torch
import shutil
import trimesh
from tqdm import tqdm
from pymeshlab import MeshSet
from concurrent.futures import ProcessPoolExecutor, as_completed


def prepare_mesh_with_fallback(
    stl_path, 
    target_vertex_count, 
    max_iter=10,
    max_verts=100000
):
    raw_mesh = trimesh.load(stl_path, force="mesh")
    raw_v = len(raw_mesh.vertices)

    if raw_v <= target_vertex_count:
        return True, {
            "mode": "copy",
            "orig_vertices": raw_v,
            "new_vertices": raw_v,
        }

    ms = MeshSet()
    ms.load_new_mesh(stl_path)
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_unreferenced_vertices")

    mesh = ms.current_mesh()
    orig_v = mesh.vertex_number()
    orig_f = mesh.face_number()

    if orig_v > max_verts:
        return False, {
            "reason": "mesh too complex to simplify",
            "orig_vertices": orig_v,
        }

    prev_v = orig_v
    target_face_count = max(int(orig_f * target_vertex_count / max(orig_v, 1)), 4)

    for _ in range(max_iter):
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=max(target_face_count, 4),
            preservenormal=True,
            preserveboundary=True,
            preservetopology=False,
            autoclean=True,
        )

        cur_mesh = ms.current_mesh()
        cur_v = cur_mesh.vertex_number()
        cur_f = cur_mesh.face_number()

        if cur_v <= target_vertex_count:
            return True, {
                "mode": "simplified",
                "orig_vertices": raw_v,
                "new_vertices": cur_v,
                "meshset": ms,
            }

        if cur_v >= prev_v:
            target_face_count = max(int(target_face_count * 0.5), 4)
        else:
            target_face_count = max(int(cur_f * target_vertex_count / max(cur_v, 1)), 4)

        prev_v = cur_v

    final_v = ms.current_mesh().vertex_number()
    if final_v <= target_vertex_count:
        return True, {
            "mode": "simplified",
            "orig_vertices": raw_v,
            "new_vertices": final_v,
            "meshset": ms,
        }

    return False, {
        "reason": "still_too_large_after_simplification",
        "orig_vertices": raw_v,
        "new_vertices": final_v,
    }


def process_one_setting(args):
    step, save_robot_name, val_data_root, val_save_root, exp_root, target_vertex_count = args

    try:
        sim_data_root = f"{exp_root}/sim/step_{step}"
        os.makedirs(sim_data_root, exist_ok=True)

        object_urdf_root = "/mnt/fx/TRO-1B-Dataset/object_urdf"

        hand_sim_data_root = f"{sim_data_root}/{save_robot_name}"
        os.makedirs(hand_sim_data_root, exist_ok=True)

        val_data_name = f"step_{step}.pth"
        val_data = torch.load(
            f"{val_save_root}/{save_robot_name}/{val_data_name}",
            weights_only=False
        )

        load_robot_name = save_robot_name.replace("_left", "") if "left" in save_robot_name else save_robot_name

        skipped_count = 0
        kept_count = 0
        simplified_count = 0
        copied_count = 0

        for data_id, data in tqdm(
            enumerate(val_data),
            total=len(val_data),
            desc=f"{save_robot_name}-step_{step}"
        ):  

            meta_name = data["meta_name"]
            stl_path = os.path.join(object_urdf_root, f"{meta_name}.stl")
            urdf_path = os.path.join(object_urdf_root, f"{meta_name}.urdf")

            ok, mesh_info = prepare_mesh_with_fallback(
                stl_path=stl_path,
                target_vertex_count=target_vertex_count,
            )

            if not ok:
                print(
                    f"[SKIP] {save_robot_name}, step_{step}, meta={meta_name}, info={mesh_info}",
                    flush=True
                )
                skipped_count += 1
                continue

            meta_dir = f"{hand_sim_data_root}/{meta_name}"
            object_urdf_dir = f"{meta_dir}/object_urdf"
            save_stl_path = f"{object_urdf_dir}/{meta_name}.stl"
            save_urdf_path = f"{object_urdf_dir}/{meta_name}.urdf"

            os.makedirs(object_urdf_dir, exist_ok=True)

            if mesh_info["mode"] == "copy":
                shutil.copy2(stl_path, save_stl_path)
                copied_count += 1
            else:
                mesh_info["meshset"].save_current_mesh(save_stl_path)
                simplified_count += 1

            env_meta = torch.load(
                f"{val_data_root}/{load_robot_name}/{meta_name}.pt",
                weights_only=False
            )
            env_meta = env_meta["meta"]

            predict_q = data["predict_q"]
            valid_scene_ids = data.get("valid_scene_ids", None)
            if valid_scene_ids is None:
                valid_scene_ids = torch.arange(len(predict_q), dtype=torch.long)
            elif isinstance(valid_scene_ids, torch.Tensor):
                valid_scene_ids = valid_scene_ids.to(dtype=torch.long).cpu()
            else:
                valid_scene_ids = torch.as_tensor(valid_scene_ids, dtype=torch.long)

            if len(valid_scene_ids) != len(predict_q):
                raise ValueError(
                    f"valid_scene_ids/predict_q length mismatch for {save_robot_name}/{meta_name}: "
                    f"{len(valid_scene_ids)} vs {len(predict_q)}"
                )

            joint = predict_q.cpu().tolist()
            env_meta_list = [
                env_meta[int(scene_id)]["target"].tolist()
                for scene_id in valid_scene_ids.tolist()
            ]

            sim_dict = dict(
                joint=joint,
                meta=env_meta_list,
                valid_scene_ids=valid_scene_ids.tolist(),
            )
            with open(f"{meta_dir}/val_data.json", "w") as f:
                json.dump(sim_dict, f)

            shutil.copy2(urdf_path, save_urdf_path)

            kept_count += 1

        return (
            f"[OK] {save_robot_name}, step_{step}, total={len(val_data)}, "
            f"kept={kept_count}, copied={copied_count}, "
            f"simplified={simplified_count}, skipped={skipped_count}"
        )

    except Exception as e:
        return f"[ERROR] {save_robot_name}, step_{step}: {e}"


if __name__ == '__main__':
    val_data_root = "/mnt/fx/TRO-1B-Dataset/TRO-val-dataset"
    exp_root = "/mnt/fx/tro_exp/train_0.2B_multi_hand_v6_noise"
    val_save_root = f"{exp_root}/val"

    steps = ["820000"]
    robot_names = ["xhand", "allegro_hand", "barrett_hand", "leaphand", "shadow_hand", "dclaw_gripper", "schunk_hand", "ability_hand"]
    target_vertex_count = 8192

    tasks = []
    for step in steps:
        for robot_name in robot_names:
            tasks.append((step, robot_name, val_data_root, val_save_root, exp_root, target_vertex_count))

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_one_setting, task) for task in tasks]

        for future in as_completed(futures):
            print(future.result(), flush=True)
