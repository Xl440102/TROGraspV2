import json
import os
import random

import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from utils.tro_hand_model import create_hand_model


DEFAULT_ZARR_HAND_NAMES = [
    "ability_hand",
    "allegro_hand",
    "barrett_hand",
    "dclaw_gripper",
    "leaphand",
    "shadow_hand",
    "schunk_hand",
    "xhand",
    "wuji_hand"
]


class TROMixedHandZarrDataset(IterableDataset):
    def __init__(
        self,
        sources,
        align_meta_path,
        hand_type,
        batch_size=48,
        max_links=25,
        max_joint_dim=30,
        num_object_points=512,
        num_env_points=2048,
        dtype=torch.float32,
        seed=0,
        max_store_cache_size=64,
        max_object_pc_cache_size=256,
        hand_resample_trials=4,
        is_mirror=False,
    ):
        super().__init__()
        assert not is_mirror, "This loader currently supports non-mirror only."

        self.sources = list(sources)
        self.hand_type = list(hand_type)
        self.batch_size = int(batch_size)
        self.max_links = int(max_links)
        self.max_joint_dim = int(max_joint_dim)
        self.num_object_points = int(num_object_points)
        self.num_env_points = int(num_env_points)
        self.dtype = dtype
        self.seed = int(seed)
        self.hand_resample_trials = int(hand_resample_trials)

        self.align_meta = torch.load(align_meta_path, map_location="cpu", weights_only=False)
        self.hand_model = {h: create_hand_model(h, device="cpu") for h in self.hand_type}
        self.hand_name_to_local_id = {h: i for i, h in enumerate(self.hand_type)}

        self.max_store_cache_size = int(max_store_cache_size)
        self.max_object_pc_cache_size = int(max_object_pc_cache_size)
        self._store_cache = {}
        self._store_cache_order = []
        self._object_pc_cache = {}
        self._object_pc_cache_order = []

        self.object_records_by_source = self._build_object_records_by_source()
        self.source_names = list(self.object_records_by_source.keys())
        if not self.source_names:
            raise ValueError("No zarr records found for any configured source.")

        weights = [float(self.object_records_by_source[name]["weight"]) for name in self.source_names]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError("At least one source weight must be positive.")
        self.source_probs = [w / weight_sum for w in weights]

    def _build_source_hand_id_map(self, source):
        force_hand_name = source.get("force_hand_type", None)
        if force_hand_name:
            return None

        zarr_hand_names = list(source.get("zarr_hand_names", DEFAULT_ZARR_HAND_NAMES))
        return {
            zarr_hand_names.index(h): h
            for h in source.get("hand_type", self.hand_type)
            if h in zarr_hand_names
        }

    def _build_object_records_by_source(self):
        records_by_source = {}

        for source_id, source in enumerate(self.sources):

            source_name = source["name"]
            data_root = source["data_root"]
            object_pc_root = source["object_pc_root"]
            
            if source_name == "original_8_hands":
                zarr_root = os.path.join(data_root, "object_zarrs")
            else:
                zarr_root = source["zarr_root"]
            
            force_hand_name = source.get("force_hand_type", None)
            hand_id_map = self._build_source_hand_id_map(source)
            weight = float(source.get("weight", 1.0))

            if force_hand_name is not None and force_hand_name not in self.hand_name_to_local_id:
                raise ValueError(f"{source_name}: force_hand_type={force_hand_name} is not in hand_type.")
            if force_hand_name is None and not hand_id_map:
                raise ValueError(f"{source_name}: no hand IDs match configured hand_type.")

            records = []
            for fname in sorted(os.listdir(zarr_root)):
                if not fname.endswith(".zarr"):
                    continue
                records.append(
                    {
                        "zarr_path": os.path.join(zarr_root, fname),
                        "object_name": fname[:-5],
                        "object_pc_root": object_pc_root,
                        "source_name": source_name,
                        "force_hand_name": force_hand_name,
                        "hand_id_map": hand_id_map,
                    }
                )

            if records:
                records_by_source[source_name] = {"records": records, "weight": weight}

        return records_by_source

    def _split_objects_for_worker(self, objects):
        wi = get_worker_info()
        worker_id = wi.id if wi else 0
        num_workers = wi.num_workers if wi else 1

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        else:
            rank = 0
            world = 1

        gid = rank * num_workers + worker_id
        gnum = world * num_workers
        return objects[gid::gnum]

    def _touch_store_cache(self, zarr_path):
        if zarr_path in self._store_cache_order:
            self._store_cache_order.remove(zarr_path)
        self._store_cache_order.append(zarr_path)

        while len(self._store_cache_order) > self.max_store_cache_size:
            victim = self._store_cache_order.pop(0)
            self._store_cache.pop(victim, None)

    def _get_store(self, zarr_path):
        if zarr_path not in self._store_cache:
            self._store_cache[zarr_path] = zarr.open_group(zarr_path, mode="r")
        self._touch_store_cache(zarr_path)
        return self._store_cache[zarr_path]

    def _touch_object_pc_cache(self, cache_key):
        if cache_key in self._object_pc_cache_order:
            self._object_pc_cache_order.remove(cache_key)
        self._object_pc_cache_order.append(cache_key)

        while len(self._object_pc_cache_order) > self.max_object_pc_cache_size:
            victim = self._object_pc_cache_order.pop(0)
            self._object_pc_cache.pop(victim, None)

    def _get_object_pc(self, object_pc_root, object_name):
        cache_key = (object_pc_root, object_name)
        if cache_key not in self._object_pc_cache:
            pc = torch.load(
                os.path.join(object_pc_root, f"{object_name}.pt"),
                map_location="cpu",
                weights_only=False,
            )
            self._object_pc_cache[cache_key] = pc[:, :3].to(self.dtype)
        self._touch_object_pc_cache(cache_key)
        return self._object_pc_cache[cache_key]

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

    def remove_hand_mount(self, q, hand_name):
        trans, rpy, joint_vals = q[:3], q[3:6], q[6:]
        T_world_base = self.get_se3_from_trans_and_rpy(trans, rpy)

        if hand_name in self.align_meta:
            T_base_palm = self.align_meta[hand_name]
            if isinstance(T_base_palm, torch.Tensor):
                T_base_palm = T_base_palm.detach().cpu().numpy()
        else:
            T_base_palm = np.eye(4, dtype=np.float32)

        T_world_palm = T_world_base @ T_base_palm
        trans_palm = T_world_palm[:3, 3]
        rpy_palm = R.from_matrix(T_world_palm[:3, :3]).as_euler("XYZ")
        return np.concatenate([trans_palm, rpy_palm, joint_vals], axis=0)

    def _matrix_to_pose_(self, rotation_matrix):
        poses = torch.zeros((self.max_links, 6), dtype=torch.float32)
        num_links = min(rotation_matrix.shape[0], self.max_links)
        for link_id in range(num_links):
            T = rotation_matrix[link_id]
            rot = T[:3, :3].numpy()
            trans = T[:3, 3].numpy()
            axis_angle = R.from_matrix(rot).as_rotvec()
            poses[link_id] = torch.from_numpy(np.concatenate((trans, axis_angle), axis=0))
        return poses

    def _sample_object_pc(self, object_pc, target_pose, generator):
        ones = torch.ones((object_pc.shape[0], 1), dtype=self.dtype)
        object_pc_homo = torch.cat([object_pc, ones], dim=1)

        target_object_se3 = self.get_se3_from_trans_and_quat(target_pose)
        target_object_se3 = torch.from_numpy(target_object_se3).to(self.dtype)
        transformed_object_pc = (object_pc_homo @ target_object_se3.T)[:, :3]

        if transformed_object_pc.shape[0] >= self.num_object_points:
            idx = torch.randperm(transformed_object_pc.shape[0], generator=generator)[:self.num_object_points]
            return transformed_object_pc[idx]

        pad = torch.randint(
            0,
            transformed_object_pc.shape[0],
            (self.num_object_points - transformed_object_pc.shape[0],),
            generator=generator,
        )
        return torch.cat([transformed_object_pc, transformed_object_pc[pad]], dim=0)

    def _sample_env_pc(self, env_pc, c_object, generator):
        d2 = ((env_pc - c_object[None, :]) ** 2).sum(dim=1)

        if env_pc.shape[0] >= self.num_env_points:
            idx = torch.topk(d2, k=self.num_env_points, largest=False).indices
            return env_pc[idx]

        pad = torch.randint(
            0,
            env_pc.shape[0],
            (self.num_env_points - env_pc.shape[0],),
            generator=generator,
        )
        return torch.cat([env_pc, env_pc[pad]], dim=0)

    def _choose_random_sample_idx(self, record, store, generator, used_hands=None):
        total_samples = store["data/joint"].shape[0]
        force_hand_name = record["force_hand_name"]
        fallback = None

        if total_samples <= 0:
            return fallback

        for _ in range(self.hand_resample_trials):
            idx = int(torch.randint(0, total_samples, (1,), generator=generator).item())

            if force_hand_name is not None:
                return idx, force_hand_name

            hand_id = int(store["meta/hand_id"][idx])
            if hand_id not in record["hand_id_map"]:
                continue

            hand_name = record["hand_id_map"][hand_id]
            if fallback is None:
                fallback = (idx, hand_name)

            if used_hands is None or hand_name not in used_hands:
                return idx, hand_name

        return fallback

    def _load_one_sample(self, record, generator, used_hands=None):
        store = self._get_store(record["zarr_path"])
        picked = self._choose_random_sample_idx(record, store, generator, used_hands)
        if picked is None:
            return None

        sample_idx, hand_name = picked
        env_pc_np = np.asarray(store["data/pc"][sample_idx], dtype=np.float32)
        joint_np_full = np.asarray(store["data/joint"][sample_idx], dtype=np.float32)
        joint_dim = int(store["data/joint_dim_raw"][sample_idx])
        joint_np = joint_np_full[:joint_dim]

        meta_json = store["meta/meta_json"][sample_idx]
        if isinstance(meta_json, bytes):
            meta_json = meta_json.decode("utf-8")

        try:
            meta = json.loads(str(meta_json))
        except Exception:
            return None
        if not isinstance(meta, dict) or "target" not in meta:
            return None

        target_pose = np.asarray(meta["target"], dtype=np.float32).reshape(-1)
        if target_pose.shape[0] != 7:
            return None

        object_pc = self._get_object_pc(record["object_pc_root"], record["object_name"])
        sampled_object_pc = self._sample_object_pc(object_pc, meta["target"], generator)
        c_object = sampled_object_pc.mean(dim=0)

        target_q_np = self.remove_hand_mount(joint_np, hand_name)
        target_q = torch.from_numpy(target_q_np).to(torch.float32)
        if target_q.shape[0] > self.max_joint_dim:
            return None

        link_se3 = self.hand_model[hand_name].get_link_se3(target_q)
        link_pose = self._matrix_to_pose_(link_se3)

        target_q_pad = torch.cat(
            [
                target_q,
                torch.zeros(self.max_joint_dim - target_q.shape[0], dtype=target_q.dtype),
            ],
            dim=0,
        )

        env_pc = torch.from_numpy(env_pc_np).to(self.dtype)
        sampled_env_pc = self._sample_env_pc(env_pc, c_object, generator)

        return {
            "robot_id": self.hand_name_to_local_id[hand_name],
            "env_pc": sampled_env_pc,
            "target_q": target_q_pad,
            "target_vec": link_pose,
            "object_pc": sampled_object_pc,
            "hand_name": hand_name,
        }

    def _pack_batch(self, samples):
        return {
            "robot_id": torch.tensor([s["robot_id"] for s in samples], dtype=torch.long),
            "env_pc": torch.stack([s["env_pc"] for s in samples], dim=0),
            "target_q": torch.stack([s["target_q"] for s in samples], dim=0),
            "target_vec": torch.stack([s["target_vec"] for s in samples], dim=0),
            "object_pc": torch.stack([s["object_pc"] for s in samples], dim=0),
        }

    def _local_records_by_source_for_worker(self):
        out = {}
        for source_name, data in self.object_records_by_source.items():
            local_records = self._split_objects_for_worker(data["records"])
            if local_records:
                out[source_name] = local_records
        return out

    def _sample_source_name(self, generator, available_source_names):
        probs = np.asarray(
            [
                self.source_probs[self.source_names.index(name)]
                for name in available_source_names
            ],
            dtype=np.float64,
        )
        probs = probs / probs.sum()
        source_idx = int(torch.multinomial(torch.from_numpy(probs), 1, generator=generator).item())
        return available_source_names[source_idx]

    def __iter__(self):
        wi = get_worker_info()
        worker_id = wi.id if wi else 0

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        worker_seed = self.seed + rank * 100000 + worker_id
        generator = torch.Generator()
        generator.manual_seed(worker_seed)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2 ** 32))

        records_by_source = self._local_records_by_source_for_worker()
        available_source_names = list(records_by_source.keys())
        if not available_source_names:
            return iter(())

        while True:
            batch_samples = []
            used_hands = set()

            while len(batch_samples) < self.batch_size:
                source_name = self._sample_source_name(generator, available_source_names)
                records = records_by_source[source_name]
                record_idx = int(torch.randint(0, len(records), (1,), generator=generator).item())
                sample = self._load_one_sample(records[record_idx], generator, used_hands=used_hands)
                if sample is None:
                    continue
                batch_samples.append(sample)
                used_hands.add(sample["hand_name"])

            yield self._pack_batch(batch_samples)


def create_dataloader_zarr(cfg):
    dataset = TROMixedHandZarrDataset(
        sources=cfg.sources,
        align_meta_path=cfg.align_meta_path,
        hand_type=cfg.hand_type,
        is_mirror=cfg.get("is_mirror", False),
        batch_size=cfg.batch_size,
        max_links=cfg.get("max_links", 25),
        max_joint_dim=cfg.get("max_joint_dim", 30),
        num_object_points=cfg.get("num_object_points", 512),
        num_env_points=cfg.get("num_env_points", 2048),
        dtype=torch.float32,
        seed=cfg.get("seed", 0),
        hand_resample_trials=cfg.get("hand_resample_trials", 4),
    )

    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
