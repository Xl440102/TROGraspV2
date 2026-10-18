import os
import json
import math
import torch
import bisect
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Sampler
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from utils.tro_hand_model import create_hand_model


import os
import json
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader, get_worker_info
from utils.tro_hand_model import create_hand_model


class TROFilteredIterDataset(Dataset):
    def __init__(
        self,
        align_meta_path: str,
        data_root: str,
        hand_type: str,
        hand_list: list,
        num_object_points: int = 512,
        num_env_points: int = 2048,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        super().__init__()
        self.data_root = f"{data_root}/{hand_type}"
        self.test_metas = [file.split('.')[0] for file in os.listdir(self.data_root)]
        self.align_meta = torch.load(align_meta_path, map_location="cpu", weights_only=False)
        self.object_pc_root = f"/mnt/fx/TRO-1B-Dataset/object_pc"
        self.hand_type = hand_type
        self.hand_model = create_hand_model(hand_type, device="cpu")
        self.robot_id = hand_list.index(hand_type)

        self.num_object_points = num_object_points
        self.num_env_points = num_env_points
        self.dtype = dtype
        self.seed = seed

    def __len__(self):
        return len(self.test_metas)

    def get_se3_from_trans_and_quat(self, pose):
        trans, q_xyzw = pose[:3], pose[3:]
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = trans
        T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
        return T

    def get_se3_from_trans_and_rpy(self, trans, rpy):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = trans
        T[:3, :3] = R.from_euler("XYZ", rpy).as_matrix()
        return T

    def remove_hand_mount(self, q, hand_type):
        trans, rpy, joint_vals = q[:3], q[3:6], q[6:]
        T_world_base = self.get_se3_from_trans_and_rpy(trans, rpy)
        T_base_palm = self.align_meta[hand_type]
        T_world_palm = T_world_base @ T_base_palm

        trans_palm = T_world_palm[:3, 3]
        rpy_palm = R.from_matrix(T_world_palm[:3, :3]).as_euler("XYZ")
        q_aligned = np.concatenate([trans_palm, rpy_palm, joint_vals], axis=0)
        return q_aligned

    def _matrix_to_pose_(self, rotation_matrix):
        poses = torch.zeros((self.max_links, 6), dtype=torch.float32)
        for link_id, T in enumerate(rotation_matrix):
            rot = T[:3, :3].numpy()
            trans = T[:3, 3].numpy()
            axis_angle = R.from_matrix(rot).as_rotvec()
            pose = np.concatenate((trans, axis_angle), axis=0)
            poses[link_id] = torch.from_numpy(pose)
        return poses

    def __getitem__(self, idx):
      
        object_name = self.test_metas[idx]
        current_data = torch.load(f"{self.data_root}/{object_name}.pt", map_location="cpu", weights_only=False)
        object_pc = torch.load(
            f"{self.object_pc_root}/{object_name}.pt",
            map_location="cpu",
            weights_only=False
        )[:, :3].to(self.dtype)

        ones = torch.ones((object_pc.shape[0], 1), dtype=self.dtype)
        object_pc_homo = torch.cat([object_pc, ones], dim=1)

        file_env_pc = current_data["env_pc"]
        file_meta = current_data["meta"]
        num_scene = len(file_env_pc)

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        env_pc_list = []
        object_pc_list = []

        for idx in range(num_scene):
            # object pc transform
            target_object_se3 = self.get_se3_from_trans_and_quat(file_meta[idx]["target"])
            target_object_se3 = torch.from_numpy(target_object_se3).to(self.dtype)
            transformed_object_pc = (object_pc_homo @ target_object_se3.T)[:, :3]

            # sample object pc
            if transformed_object_pc.shape[0] >= self.num_object_points:
                object_pc_idx = torch.randperm(
                    transformed_object_pc.shape[0],
                    generator=generator
                )[:self.num_object_points]
                sampled_object_pc = transformed_object_pc[object_pc_idx]
            else:
                pad = torch.randint(
                    0,
                    transformed_object_pc.shape[0],
                    (self.num_object_points - transformed_object_pc.shape[0],),
                    generator=generator
                )
                sampled_object_pc = torch.cat([transformed_object_pc, transformed_object_pc[pad]], dim=0)

            c_object = sampled_object_pc.mean(dim=0)

            # env pc
            env_pc = torch.from_numpy(file_env_pc[idx]).to(self.dtype)

            d2 = ((env_pc - c_object[None, :]) ** 2).sum(dim=1)
            if env_pc.shape[0] >= self.num_env_points:
                env_pc_idx = torch.topk(d2, k=self.num_env_points, largest=False).indices
                sampled_env_pc = env_pc[env_pc_idx]
            else:
                pad = torch.randint(
                    0,
                    env_pc.shape[0],
                    (self.num_env_points - env_pc.shape[0],),
                    generator=generator
                )
                sampled_env_pc = torch.cat([env_pc, env_pc[pad]], dim=0)
        
            env_pc_list.append(sampled_env_pc)
            object_pc_list.append(sampled_object_pc)

        return {
            "robot_id": torch.ones((num_scene,), dtype=torch.long) * self.robot_id,
            "env_pc": torch.stack(env_pc_list, dim=0),
            "object_pc": torch.stack(object_pc_list, dim=0),
            "meta_name": object_name
        }
            

def create_dataloader(cfg):
    dataset = TROFilteredIterDataset(
        align_meta_path=cfg.align_meta_path,
        data_root=cfg.data_root,
        hand_type=cfg.hand_type,
        hand_list=cfg.hand_list,
        seed=cfg.seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    return dataloader