"""Batched multi-target IK solver with analytic Jacobian.

Extends ``solve_ik_with_multiple_targets`` to support:

* **Batch dimension** – solve ``B`` IK problems in parallel via ``jax.vmap``.
* **Position-only mode** – when only target positions are given.
* **Full SE3 pose mode** – when both orientation and position are given.

Both modes use a custom analytic Jacobian that concatenates all target
residuals into a single vector so the optimizer can correlate constraints
across targets.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp

import pyroki as pk
from pyroki._residuals._pose_residual_analytic_jac import (
    _get_actuated_joints_applied_to_target,
)


# ── Public API ────────────────────────────────────────────────────────


def solve_ik_with_multiple_targets_batch(
    robot: pk.Robot,
    target_link_names: Sequence[str],
    initial_cfgs: onp.ndarray,
    target_positions: onp.ndarray,
    target_wxyzs: onp.ndarray | None = None,
    pos_weight: float = 10.0,
    ori_weight: float = 5.0,
    limit_weight: float = 10.0,
    max_iterations: int = 50,
) -> onp.ndarray:
    """Solve IK for a batch of multi-target problems.

    Args:
        robot: PyRoKi Robot.
        target_link_names: Sequence of link names to be controlled.
        initial_cfgs: Initial joint configs.  Shape: ``(B, n_actuated_joints)``.
        target_positions: Target positions.  Shape: ``(B, num_targets, 3)``.
        target_wxyzs: Target orientations (wxyz quaternions).
            Shape: ``(B, num_targets, 4)``.  When *None*, position-only IK is used.
        pos_weight: Weight for position residual.
        ori_weight: Weight for orientation residual (ignored in pos-only mode).
        limit_weight: Weight for joint-limit cost.
        max_iterations: Solver iteration budget per sample.

    Returns:
        cfgs: Joint configurations.  Shape: ``(B, n_actuated_joints)``.
    """
    num_targets = len(target_link_names)
    batch_size = initial_cfgs.shape[0]
    assert initial_cfgs.shape == (batch_size, robot.joints.num_actuated_joints)
    assert target_positions.shape == (batch_size, num_targets, 3)

    target_link_indices = jnp.array(
        [robot.links.names.index(n) for n in target_link_names], dtype=jnp.int32
    )

    if target_wxyzs is not None:
        assert target_wxyzs.shape == (batch_size, num_targets, 4)
        target_poses_wxyz_xyz = jnp.concatenate(
            [jnp.asarray(target_wxyzs), jnp.asarray(target_positions)], axis=-1
        )
        cfgs = _solve_batch_pose(
            robot,
            target_link_indices,
            jnp.asarray(initial_cfgs),
            target_poses_wxyz_xyz,
            jnp.float32(pos_weight),
            jnp.float32(ori_weight),
            jnp.float32(limit_weight),
            max_iterations,
        )
    else:
        cfgs = _solve_batch_pos(
            robot,
            target_link_indices,
            jnp.asarray(initial_cfgs),
            jnp.asarray(target_positions),
            jnp.float32(pos_weight),
            jnp.float32(limit_weight),
            max_iterations,
        )

    assert cfgs.shape == (batch_size, robot.joints.num_actuated_joints)
    return onp.array(cfgs)


# ── Batch entry points ────────────────────────────────────────────────


@jdc.jit
def _solve_batch_pos(
    robot: pk.Robot,
    target_link_indices: jax.Array,
    initial_cfgs: jax.Array,
    target_positions: jax.Array,
    pos_weight: jax.Array,
    limit_weight: jax.Array,
    max_iterations: jdc.Static[int] = 50,
) -> jax.Array:
    return jax.vmap(
        lambda init_q, tgt_pos: _solve_single_pos(
            robot, target_link_indices, init_q, tgt_pos,
            pos_weight, limit_weight, max_iterations,
        )
    )(initial_cfgs, target_positions)


@jdc.jit
def _solve_batch_pose(
    robot: pk.Robot,
    target_link_indices: jax.Array,
    initial_cfgs: jax.Array,
    target_poses_wxyz_xyz: jax.Array,
    pos_weight: jax.Array,
    ori_weight: jax.Array,
    limit_weight: jax.Array,
    max_iterations: jdc.Static[int] = 50,
) -> jax.Array:
    return jax.vmap(
        lambda init_q, tgt: _solve_single_pose(
            robot, target_link_indices, init_q, tgt,
            pos_weight, ori_weight, limit_weight, max_iterations,
        )
    )(initial_cfgs, target_poses_wxyz_xyz)


# ── Single-sample solvers ────────────────────────────────────────────


def _solve_single_pos(
    robot: pk.Robot,
    target_link_indices: jax.Array,
    init_q: jax.Array,
    target_positions: jax.Array,
    pos_weight: jax.Array,
    limit_weight: jax.Array,
    max_iterations: int,
) -> jax.Array:
    """Position-only IK for one sample."""
    joint_var = robot.joint_var_cls(0)

    costs = [
        _multi_target_pos_cost(
            robot, joint_var, target_link_indices, target_positions, pos_weight,
        ),
        pk.costs.limit_constraint(robot, joint_var, limit_weight),
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(init_q)]),
            linear_solver="dense_cholesky",
            verbose=False,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
            termination=jaxls.TerminationConfig(
                max_iterations=max_iterations, early_termination=True,
            ),
        )
    )
    return sol[joint_var]


def _solve_single_pose(
    robot: pk.Robot,
    target_link_indices: jax.Array,
    init_q: jax.Array,
    target_poses_wxyz_xyz: jax.Array,
    pos_weight: jax.Array,
    ori_weight: jax.Array,
    limit_weight: jax.Array,
    max_iterations: int,
) -> jax.Array:
    """Full SE3 pose IK for one sample."""
    joint_var = robot.joint_var_cls(0)

    costs = [
        _multi_target_pose_cost(
            robot, joint_var, target_link_indices,
            target_poses_wxyz_xyz, pos_weight, ori_weight,
        ),
        pk.costs.limit_constraint(robot, joint_var, limit_weight),
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(init_q)]),
            linear_solver="dense_cholesky",
            verbose=False,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
            termination=jaxls.TerminationConfig(
                max_iterations=max_iterations, early_termination=True,
            ),
        )
    )
    return sol[joint_var]


# ── Cost factories (residual + analytic Jacobian) ─────────────────────


_PosCostJacCache = tuple[jax.Array, jax.Array, jax.Array]
_PoseCostJacCache = tuple[jax.Array, jax.Array, jax.Array]


def _multi_target_pos_jac(
    vals: jaxls.VarValues,
    jac_cache: _PosCostJacCache,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    target_link_indices: jax.Array,
    target_positions: jax.Array,
    pos_weight: jax.Array | float,
) -> jax.Array:
    """Analytic Jacobian for batched position-only residual.

    Returns shape ``(num_targets * 3, num_actuated_joints)``.
    """
    del vals, joint_var, target_positions

    Ts_world_joint, Ts_world_link, _ = jac_cache
    num_targets = target_link_indices.shape[0]
    n_act = robot.joints.num_actuated_joints

    Ts_world_joint_se3 = jaxlie.SE3(Ts_world_joint)
    T_world_targets = jaxlie.SE3(Ts_world_link[target_link_indices])

    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]
    omega_local = joint_twists[:, 3:]
    vel_local = joint_twists[:, :3]
    omega_world = Ts_world_joint_se3.rotation() @ omega_local
    vel_world = Ts_world_joint_se3.rotation() @ vel_local

    diff = (
        T_world_targets.translation()[:, None, :]
        - Ts_world_joint_se3.translation()[None, :, :]
    )
    jac_dp = (
        jnp.cross(omega_world[None, :, :], diff) + vel_world[None, :, :]
    ).transpose(0, 2, 1)  # (N, 3, n_joints)

    # Determine which actuated joints contribute to each target link.
    pji = jnp.asarray(robot.links.parent_joint_indices, dtype=jnp.int32)
    parent_joint_indices = jnp.where(pji == -1, 0, pji)
    target_joint_indices = parent_joint_indices[target_link_indices]
    joints_applied = jax.vmap(
        lambda idx: _get_actuated_joints_applied_to_target(robot, idx)
    )(target_joint_indices)

    full_jac = jnp.zeros((num_targets * 3, n_act))

    def _scatter_one(carry, i):
        jac_i = jac_dp[i]
        applied_i = joints_applied[i]
        jac_i_act = (
            jnp.zeros((3, n_act))
            .at[:, applied_i]
            .add((applied_i[None, :] != -1) * jac_i)
        )
        return jax.lax.dynamic_update_slice(carry, jac_i_act, (i * 3, 0)), None

    full_jac, _ = jax.lax.scan(_scatter_one, full_jac, jnp.arange(num_targets))
    return pos_weight * full_jac


@jaxls.Cost.factory(jac_custom_with_cache_fn=_multi_target_pos_jac)
def _multi_target_pos_cost(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    target_link_indices: jax.Array,
    target_positions: jax.Array,
    pos_weight: jax.Array | float,
) -> tuple[jax.Array, _PosCostJacCache]:
    """Multi-target position residual with analytic Jacobian cache."""
    joint_cfg = vals[joint_var]
    Ts_world_joint = robot._forward_kinematics_joints(joint_cfg)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)
    actual_pos = jaxlie.SE3(Ts_world_link[target_link_indices]).translation()
    pos_error = actual_pos - target_positions
    return (
        pos_weight * pos_error.flatten(),
        (Ts_world_joint, Ts_world_link, pos_error),
    )


def _multi_target_pose_jac(
    vals: jaxls.VarValues,
    jac_cache: _PoseCostJacCache,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    target_link_indices: jax.Array,
    target_poses_wxyz_xyz: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
) -> jax.Array:
    """Analytic Jacobian for batched SE3 pose residual.

    Returns shape ``(num_targets * 6, num_actuated_joints)``.
    """
    del vals, joint_var, target_poses_wxyz_xyz

    Ts_world_joint, Ts_world_link, pose_errors_wxyz_xyz = jac_cache
    num_targets = target_link_indices.shape[0]
    n_act = robot.joints.num_actuated_joints

    Ts_world_joint_se3 = jaxlie.SE3(Ts_world_joint)
    T_world_targets = jaxlie.SE3(Ts_world_link[target_link_indices])
    pose_errors = jaxlie.SE3(pose_errors_wxyz_xyz)

    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]
    omega_local = joint_twists[:, 3:]
    vel_local = joint_twists[:, :3]
    omega_world = Ts_world_joint_se3.rotation() @ omega_local
    vel_world = Ts_world_joint_se3.rotation() @ vel_local

    diff = (
        T_world_targets.translation()[:, None, :]
        - Ts_world_joint_se3.translation()[None, :, :]
    )
    v_world = jnp.cross(omega_world[None, :, :], diff) + vel_world[None, :, :]

    R_ee_world = T_world_targets.rotation().inverse().as_matrix()
    vel_ee = jnp.einsum("nij,nmj->nmi", R_ee_world, v_world)
    omega_ee = jnp.einsum("nij,mj->nmi", R_ee_world, omega_world)

    spatial_jac = jnp.concatenate([vel_ee, omega_ee], axis=-1).transpose(0, 2, 1)

    jlog_matrices = jax.vmap(lambda e: e.jlog())(pose_errors)
    jac_all = jnp.einsum("nij,njk->nik", jlog_matrices, spatial_jac)

    # Scatter to actuated joints.
    pji = jnp.asarray(robot.links.parent_joint_indices, dtype=jnp.int32)
    parent_joint_indices = jnp.where(pji == -1, 0, pji)
    target_joint_indices = parent_joint_indices[target_link_indices]
    joints_applied = jax.vmap(
        lambda idx: _get_actuated_joints_applied_to_target(robot, idx)
    )(target_joint_indices)

    full_jac = jnp.zeros((num_targets * 6, n_act))

    def _scatter_one(carry, i):
        jac_i = jac_all[i]
        applied_i = joints_applied[i]
        jac_i_act = (
            jnp.zeros((6, n_act))
            .at[:, applied_i]
            .add((applied_i[None, :] != -1) * jac_i)
        )
        return jax.lax.dynamic_update_slice(carry, jac_i_act, (i * 6, 0)), None

    full_jac, _ = jax.lax.scan(_scatter_one, full_jac, jnp.arange(num_targets))

    weights = jnp.tile(
        jnp.array([pos_weight] * 3 + [ori_weight] * 3), num_targets
    )
    return full_jac * weights[:, None]


@jaxls.Cost.factory(jac_custom_with_cache_fn=_multi_target_pose_jac)
def _multi_target_pose_cost(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    target_link_indices: jax.Array,
    target_poses_wxyz_xyz: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
) -> tuple[jax.Array, _PoseCostJacCache]:
    """Multi-target SE3 pose residual with analytic Jacobian cache."""
    joint_cfg = vals[joint_var]
    Ts_world_joint = robot._forward_kinematics_joints(joint_cfg)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

    T_actual = jaxlie.SE3(Ts_world_link[target_link_indices])
    T_target = jaxlie.SE3(target_poses_wxyz_xyz)
    pose_errors = T_target.inverse() @ T_actual

    residual = pose_errors.log()
    weights = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    return (
        (residual * weights).flatten(),
        (Ts_world_joint, Ts_world_link, pose_errors.wxyz_xyz),
    )
