from isaacgym import gymapi
from isaacgym import gymtorch

import os
import sys
import time
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.controller import controller


class IsaacValidator:
    def __init__(
        self,
        robot_name,
        bsz_val_data_paths,
        bsz_target_object_names,
        bsz_meta_names,
        joint_orders,
        gpu=0,
        is_filter=False,
        use_gui=False,
        robot_friction=3.0,
        object_friction=3.0,
        table_friction=3.0,
        steps_per_sec=100,
        grasp_step=100,
        debug_interval=0.01,
        table_height=0.0,
        table_size=(0.8, 0.8, 0.04),
        pregrasp_height=0.20,
        lift_height=0.20,
        base_z_offset=1.0,
        approach_steps=50,
        preshape_steps=40,
        close_steps=50,
        squeeze_steps=50,
        lift_steps=50,
        settle_steps=20,
        success_lift_thresh=0.03,
        palm_normal_local=(0.0, 0.0, -1.0),
    ):
        self.gym = gymapi.acquire_gym()

        self.robot_name = robot_name
        self.bsz_target_object_names = bsz_target_object_names
        self.bsz_meta_names = bsz_meta_names
        self.meta_pose = {}
        self.batch_size = 0
        self.env_to_object_name = {}
        self.env_to_meta_name = {}
        self.env_to_pose_index = {}

        for meta_name, object_name, val_data_path in zip(
            bsz_meta_names, bsz_target_object_names, bsz_val_data_paths
        ):
            self.meta_pose[meta_name] = json.load(open(val_data_path))["meta"]
            meta_count = len(self.meta_pose[meta_name])
            for pose_index in range(meta_count):
                self.env_to_object_name[self.batch_size] = object_name
                self.env_to_meta_name[self.batch_size] = meta_name
                self.env_to_pose_index[self.batch_size] = pose_index
                self.batch_size += 1

        self.joint_orders = joint_orders
        self.gpu = gpu
        self.is_filter = is_filter
        self.robot_friction = robot_friction
        self.object_friction = object_friction
        self.table_friction = table_friction
        self.steps_per_sec = steps_per_sec
        self.grasp_step = grasp_step
        self.debug_interval = debug_interval

        self.table_height = table_height
        self.table_size = table_size
        self.pregrasp_height = pregrasp_height
        self.lift_height = lift_height
        self.base_z_offset = base_z_offset
        self.approach_steps = approach_steps
        self.preshape_steps = preshape_steps
        self.close_steps = close_steps
        self.squeeze_steps = squeeze_steps
        self.lift_steps = lift_steps
        self.settle_steps = settle_steps
        self.success_lift_thresh = success_lift_thresh

        self.palm_normal_local = torch.tensor(palm_normal_local, dtype=torch.float32)
        self.top_grasp_cos_thresh = float(np.cos(np.pi / 3.0))

        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self.table_handles = []
        self.object_root_sim_indices = []

        self.robot_asset = None
        self.object_asset = {}
        self.table_asset = None

        self.rigid_body_num = None
        self.object_original_masses = []
        self.urdf2isaac_order = None
        self.isaac2urdf_order = None

        self.pregrasp_q = None
        self.cover_q = None
        self.grasp_q = None
        self.squeeze_q = None
        self.lift_q = None
        self.outer_stage_q = None

        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.dt = 1 / steps_per_sec
        self.sim_params.substeps = 4
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 16
        self.sim_params.physx.num_velocity_iterations = 2

        # Collision mesh in schunk hand leads to unstable performance
        if self.robot_name == "schunk_hand" or self.robot_name == "wuji_hand":
            self.sim_params.physx.contact_offset = 0.015
        else:
            self.sim_params.physx.contact_offset = 0.002

        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        self.sim_params.physx.rest_offset = 0.0

        self.sim = self.gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self._rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._actor_root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        self.viewer = None
        if use_gui:
            self.has_viewer = True
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.camera_props.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
            self.gym.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(0.0, 1.5, 1.0),
                gymapi.Vec3(0.0, 0.0, 0.0),
            )
        else:
            self.has_viewer = False

        self.robot_asset_options = gymapi.AssetOptions()
        self.robot_asset_options.disable_gravity = True
        self.robot_asset_options.fix_base_link = True
        self.robot_asset_options.collapse_fixed_joints = True

        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.override_com = True
        self.object_asset_options.override_inertia = True
        self.object_asset_options.density = 500

    def set_view_all_envs(self):
        if not self.has_viewer:
            return

        num_per_row = int(self.batch_size ** 0.5)
        spacing = 1.0

        cx = (num_per_row - 1) * spacing * 0.5
        cy = (num_per_row - 1) * spacing * 0.5

        cam_pos = gymapi.Vec3(
            cx + 1.6 * num_per_row,
            cy + 1.6 * num_per_row,
            0.7 * num_per_row,
        )
        cam_tgt = gymapi.Vec3(cx, cy, 0.15)

        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_tgt)

    def set_asset(self, robot_path, robot_file, object_paths, object_files):
        self.robot_asset = self.gym.load_asset(
            self.sim, robot_path, robot_file, self.robot_asset_options
        )

        for object_name, object_path, object_file in zip(
            self.bsz_target_object_names, object_paths, object_files
        ):
            self.object_asset[object_name] = self.gym.load_asset(
                self.sim, object_path, object_file, self.object_asset_options
            )

        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(
            self.sim,
            self.table_size[0],
            self.table_size[1],
            self.table_size[2],
            table_asset_options,
        )

        self.rigid_body_num = self.gym.get_asset_rigid_body_count(self.robot_asset) + 1

    def create_envs(self):
        for env_idx in range(self.batch_size):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-1, -1, -1),
                gymapi.Vec3(1, 1, 1),
                int(self.batch_size ** 0.5),
            )
            self.envs.append(env)

            if self.has_viewer:
                x_axis_dir = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
                x_axis_color = np.array([1, 0, 0], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, x_axis_dir, x_axis_color)

                y_axis_dir = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
                y_axis_color = np.array([0, 1, 0], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, y_axis_dir, y_axis_color)

                z_axis_dir = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
                z_axis_color = np.array([0, 0, 1], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, z_axis_dir, z_axis_color)

            table_transform = gymapi.Transform()
            table_transform.p = gymapi.Vec3(
                0.0,
                0.0,
                self.table_height - self.table_size[2] * 0.5,
            )
            table_handle = self.gym.create_actor(
                env,
                self.table_asset,
                table_transform,
                f"table_{env_idx}",
                env_idx,
            )
            self.table_handles.append(table_handle)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env, table_handle)
            for i in range(len(table_shape_props)):
                table_shape_props[i].friction = self.table_friction
            self.gym.set_actor_rigid_shape_properties(env, table_handle, table_shape_props)

            object_name = self.env_to_object_name[env_idx]
            meta_name = self.env_to_meta_name[env_idx]
            pose_index = self.env_to_pose_index[env_idx]
            meta_pose = self.meta_pose[meta_name][pose_index]

            object_transform = gymapi.Transform()
            object_transform.p = gymapi.Vec3(*meta_pose[:3])
            object_transform.r = gymapi.Quat(*meta_pose[3:7])

            object_handle = self.gym.create_actor(
                env,
                self.object_asset[object_name],
                object_transform,
                f"object_{env_idx}",
                env_idx,
            )
            self.object_handles.append(object_handle)

            object_shape_props = self.gym.get_actor_rigid_shape_properties(env, object_handle)
            for i in range(len(object_shape_props)):
                object_shape_props[i].friction = self.object_friction
            self.gym.set_actor_rigid_shape_properties(env, object_handle, object_shape_props)

            robot_init = gymapi.Transform()
            robot_init.p = gymapi.Vec3(0.0, 0.0, self.base_z_offset)

            robot_handle = self.gym.create_actor(
                env,
                self.robot_asset,
                robot_init,
                f"robot_{env_idx}",
                env_idx,
            )
            self.robot_handles.append(robot_handle)

            robot_props = self.gym.get_actor_dof_properties(env, robot_handle)
            robot_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            robot_props["stiffness"].fill(2000)
            robot_props["damping"].fill(80)
            self.gym.set_actor_dof_properties(env, robot_handle, robot_props)

            robot_shape_props = self.gym.get_actor_rigid_shape_properties(env, robot_handle)
            for i in range(len(robot_shape_props)):
                robot_shape_props[i].friction = self.robot_friction
            self.gym.set_actor_rigid_shape_properties(env, robot_handle, robot_shape_props)

        for env_idx in range(len(self.envs)):
            obj_property = self.gym.get_actor_rigid_body_properties(
                self.envs[env_idx], self.object_handles[env_idx]
            )
            object_mass = [obj_property[i].mass for i in range(len(obj_property))]
            self.object_original_masses.append(object_mass)

        self.urdf2isaac_order = np.zeros(len(self.joint_orders), dtype=np.int32)
        self.isaac2urdf_order = np.zeros(len(self.joint_orders), dtype=np.int32)
        for urdf_idx, joint_name in enumerate(self.joint_orders):
            isaac_idx = self.gym.find_actor_dof_index(
                self.envs[0],
                self.robot_handles[0],
                joint_name,
                gymapi.DOMAIN_ACTOR,
            )
            self.urdf2isaac_order[isaac_idx] = urdf_idx
            self.isaac2urdf_order[urdf_idx] = isaac_idx

        for env_idx in range(len(self.envs)):
            sim_idx = self.gym.get_actor_index(
                self.envs[env_idx],
                self.object_handles[env_idx],
                gymapi.DOMAIN_SIM,
            )
            self.object_root_sim_indices.append(sim_idx)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _set_robot_state(self, q_urdf):
        q_urdf = self._to_numpy(q_urdf)
        q_isaac = q_urdf[:, self.urdf2isaac_order]

        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]
            robot_handle = self.robot_handles[env_idx]

            dof_states = self.gym.get_actor_dof_states(
                env, robot_handle, gymapi.STATE_ALL
            ).copy()
            dof_states["pos"] = q_isaac[env_idx]
            dof_states["vel"].fill(0.0)
            self.gym.set_actor_dof_states(env, robot_handle, dof_states, gymapi.STATE_ALL)
            self.gym.set_actor_dof_position_targets(env, robot_handle, q_isaac[env_idx])

    def _set_robot_targets(self, q_urdf):
        q_urdf = self._to_numpy(q_urdf)
        q_isaac = q_urdf[:, self.urdf2isaac_order]

        for env_idx in range(len(self.envs)):
            self.gym.set_actor_dof_position_targets(
                self.envs[env_idx],
                self.robot_handles[env_idx],
                q_isaac[env_idx],
            )

    def _simulate_steps(self, num_steps):
        for _ in range(num_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            if self.has_viewer:
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                t = time.time()
                while time.time() - t < self.debug_interval:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)

    def _move_linear(self, q_start, q_end, steps, sim_steps_per_target=1):
        q_start = self._to_numpy(q_start)
        q_end = self._to_numpy(q_end)

        for step in range(steps):
            alpha = (step + 1) / steps
            q_target = q_start + alpha * (q_end - q_start)
            self._set_robot_targets(q_target)
            self._simulate_steps(sim_steps_per_target)

    def _get_object_positions(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(self._actor_root_states)

        object_indices = torch.tensor(
            self.object_root_sim_indices,
            dtype=torch.long,
            device=root_states.device,
        )
        return root_states[object_indices, :3].clone()

    def _set_object_gravity_enabled(self, enabled: bool):
        disable_flag = getattr(gymapi, "RIGID_BODY_DISABLE_GRAVITY", 1)

        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]
            object_handle = self.object_handles[env_idx]
            body_props = self.gym.get_actor_rigid_body_properties(env, object_handle)

            changed = False
            for i in range(len(body_props)):
                if hasattr(body_props[i], "flags"):
                    if enabled:
                        body_props[i].flags &= ~disable_flag
                    else:
                        body_props[i].flags |= disable_flag
                    changed = True

            if changed:
                self.gym.set_actor_rigid_body_properties(env, object_handle, body_props, True)

    def _reset_environments(self):
        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]
            object_handle = self.object_handles[env_idx]

            meta_name = self.env_to_meta_name[env_idx]
            pose_index = self.env_to_pose_index[env_idx]
            meta_pose = self.meta_pose[meta_name][pose_index]

            rb_states = self.gym.get_actor_rigid_body_states(env, object_handle, gymapi.STATE_ALL)
            rb_states["pose"]["p"].fill((meta_pose[0], meta_pose[1], meta_pose[2]))
            rb_states["pose"]["r"].fill((meta_pose[3], meta_pose[4], meta_pose[5], meta_pose[6]))
            rb_states["vel"]["linear"].fill((0.0, 0.0, 0.0))
            rb_states["vel"]["angular"].fill((0.0, 0.0, 0.0))
            self.gym.set_actor_rigid_body_states(env, object_handle, rb_states, gymapi.STATE_ALL)

        self._set_robot_state(self.pregrasp_q)
        
    def _get_palm_normal_world(self, q_batch: torch.Tensor) -> torch.Tensor:
        rpy = q_batch[:, 3:6].detach().cpu().numpy()
        rot_mats = R.from_euler("XYZ", rpy).as_matrix()
        rot_mats = torch.tensor(rot_mats, dtype=q_batch.dtype, device=q_batch.device)

        local_axis = self.palm_normal_local.to(
            device=q_batch.device,
            dtype=q_batch.dtype,
        )

        world_axis = torch.matmul(rot_mats, local_axis)
        world_axis = world_axis / (torch.norm(world_axis, dim=1, keepdim=True) + 1e-8)
        return world_axis

    def _retreat_along_palm_normal(self, q_batch: torch.Tensor, distance: float) -> torch.Tensor:
        q_out = q_batch.clone()
        palm_normal_world = self._get_palm_normal_world(q_batch)
        q_out[:, :3] -= distance * palm_normal_world
        return q_out

    def _make_lift_q(self, q_batch: torch.Tensor, distance: float) -> torch.Tensor:
        q_out = q_batch.clone()
        q_out[:, 2] += distance
        return q_out

    def set_actor_pose_dof(self, q, q_pd=None, alpha=1.0, ratio_noise_std=0.4, ratio_min=0.5, ratio_max=1.5):
        self.gym.prepare_sim(self.sim)

        if q_pd == None:
            outer_q, q_pd = controller(self.robot_name, q)
        else:
            outer_q, _ = controller(self.robot_name, q)

        outer_q_sim = outer_q.clone()
        inner_q_sim = q_pd.clone()

        outer_q_sim[:, 2] -= self.base_z_offset
        inner_q_sim[:, 2] -= self.base_z_offset

        lift_q = self._make_lift_q(inner_q_sim, self.lift_height)

        # Keep old member names so that other code does not need to change.
        # In the simplified version:
        #   pregrasp_q / cover_q / outer_stage_q = outer_q
        #   grasp_q / squeeze_q = inner_q
        self.pregrasp_q = outer_q_sim
        self.cover_q = outer_q_sim
        self.outer_stage_q = outer_q_sim
        self.grasp_q = inner_q_sim
        self.squeeze_q = inner_q_sim
        self.lift_q = lift_q

        # Start directly at outer_q.
        self._set_robot_state(self.outer_stage_q)

    def run_sim(self):
        self._set_object_gravity_enabled(False)

        self._reset_environments()
        self._simulate_steps(5)
        self._reset_environments()

        start_xy = []
        start_z = []
        for env_idx in range(len(self.envs)):
            meta_name = self.env_to_meta_name[env_idx]
            pose_index = self.env_to_pose_index[env_idx]
            meta_pose = self.meta_pose[meta_name][pose_index]
            start_xy.append([meta_pose[0], meta_pose[1]])
            start_z.append(meta_pose[2])

        start_xy = torch.tensor(start_xy, dtype=torch.float32)
        start_z = torch.tensor(start_z, dtype=torch.float32)

        # 0) Start directly at outer_q.
        self._set_robot_state(self.outer_stage_q)

        # 1) Directly set target to inner_q.
        # No manual interpolation between outer_q and inner_q.
        self._set_robot_targets(self.squeeze_q)

        for _ in range(self.grasp_step):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            if self.has_viewer:
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                t = time.time()
                while time.time() - t < self.debug_interval:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)

        self._set_object_gravity_enabled(True)
        self._simulate_steps(2)

        # 2) Lift up.
        self._move_linear(
            self.squeeze_q,
            self.lift_q,
            self.lift_steps,
            sim_steps_per_target=1,
        )

        self._simulate_steps(10)
        end_pos = self._get_object_positions().cpu()

        lift_amount = end_pos[:, 2] - start_z
        xy_shift = (end_pos[:, :2] - start_xy).norm(dim=-1)
        success = (lift_amount >= self.success_lift_thresh) & (xy_shift <= 0.05)

        success_dict = {meta_name: [] for meta_name in self.bsz_meta_names}
        for env_idx in range(len(self.envs)):
            meta_name = self.env_to_meta_name[env_idx]
            success_dict[meta_name].append(success[env_idx].item())

        return success_dict

    def reset_simulator(self):
        if getattr(self, "has_viewer", False) and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None

        if getattr(self, "envs", None):
            for env in self.envs:
                self.gym.destroy_env(env)
            self.envs = []

        if getattr(self, "sim", None) is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None

        self.sim = self.gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self._rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._actor_root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        if self.has_viewer:
            self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
            self.gym.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(0.0, 1.5, 1.0),
                gymapi.Vec3(0.0, 0.0, 0.0),
            )

        self.robot_handles = []
        self.object_handles = []
        self.table_handles = []
        self.object_root_sim_indices = []
        self.robot_asset = None
        self.object_asset = {}
        self.table_asset = None
        self.object_original_masses = []

    def destroy(self):
        if getattr(self, "has_viewer", False) and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None

        if getattr(self, "envs", None):
            for env in self.envs:
                self.gym.destroy_env(env)
            self.envs = []

        if getattr(self, "sim", None) is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None