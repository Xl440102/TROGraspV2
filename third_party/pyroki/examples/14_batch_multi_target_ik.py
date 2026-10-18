"""Batched Multi-Target IK

Verify the batched multi-target IK solver via an FK -> IK round-trip:

1. Load a robot and pick several target links.
2. Sample random joint configs (ground truth), run FK to get target poses.
3. Start from noisy initial configs, solve batched IK.
4. Visualise **all B samples in parallel** in viser — each sample shows
   GT / Init / Solved robots overlapping at the same position.
5. Save per-link error plots to PNG.

Colour legend (each sample):
  - Green (semi-transparent) = GT ground truth
  - Red   (semi-transparent) = Initial noisy config
  - Blue  (opaque)           = Solved IK result

Run:  python examples/14_batch_multi_target_ik.py --mode pose -B 8
      python examples/14_batch_multi_target_ik.py --mode pose -B 8 --vis
Open: http://localhost:8080 in your browser (only when --vis is used).
"""

import argparse
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import matplotlib.pyplot as plt
import numpy as np
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks

MAX_VIS_SAMPLES = 3

OUTPUT_DIR = Path(__file__).parent / "output"


def main(batch_size: int = 8, mode: str = "pose", vis: bool = False):
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    target_link_names = ["panda_link4", "panda_link6", "panda_hand"]
    target_link_indices = [robot.links.names.index(n) for n in target_link_names]
    n_act = robot.joints.num_actuated_joints

    print(f"Robot: panda  |  Links: {target_link_names}  |  B={batch_size}  |  mode={mode}")

    # ── 1. Ground-truth configs & FK ─────────────────────────────────

    rng = np.random.default_rng(42)
    lower, upper = np.array(robot.joints.lower_limits), np.array(robot.joints.upper_limits)
    gt_cfgs = rng.uniform(lower, upper, size=(batch_size, n_act)).astype(np.float32)
    fk_all = jax.vmap(robot.forward_kinematics)(jnp.array(gt_cfgs))

    target_positions = np.array(jaxlie.SE3(fk_all[:, target_link_indices, :]).translation())
    target_wxyzs = np.array(fk_all[:, target_link_indices, :4])

    # ── 2. Noisy initial configs ─────────────────────────────────────

    noise = rng.normal(scale=0.3, size=(batch_size, n_act)).astype(np.float32)
    initial_cfgs = np.clip(gt_cfgs + noise, lower, upper)

    # ── 3. Solve batched IK ──────────────────────────────────────────

    use_pose = mode == "pose"
    kw = dict(
        robot=robot, target_link_names=target_link_names,
        initial_cfgs=initial_cfgs, target_positions=target_positions,
        target_wxyzs=target_wxyzs if use_pose else None,
        pos_weight=50.0, ori_weight=10.0, limit_weight=10.0, max_iterations=80,
    )
    _ = pks.solve_ik_with_multiple_targets_batch(**kw)
    n_runs = 20
    t0 = time.time()
    for _ in range(n_runs):
        solved_cfgs = pks.solve_ik_with_multiple_targets_batch(**kw)
    elapsed = (time.time() - t0) / n_runs
    print(f"  solve: {elapsed*1000:.1f} ms (avg {n_runs} runs)")

    # ── 4. Errors ────────────────────────────────────────────────────

    fk_solved = jax.vmap(robot.forward_kinematics)(jnp.array(solved_cfgs))
    fk_init = jax.vmap(robot.forward_kinematics)(jnp.array(initial_cfgs))

    gt_poses = jaxlie.SE3(fk_all[:, target_link_indices, :])
    solved_poses = jaxlie.SE3(fk_solved[:, target_link_indices, :])
    init_poses = jaxlie.SE3(fk_init[:, target_link_indices, :])

    pos_err = np.linalg.norm(
        np.array(solved_poses.translation()) - np.array(gt_poses.translation()), axis=-1,
    ) * 1000
    ori_err = np.degrees(np.linalg.norm(
        np.array((gt_poses.inverse() @ solved_poses).log())[..., 3:], axis=-1,
    ))
    init_pos_err = np.linalg.norm(
        np.array(init_poses.translation()) - np.array(gt_poses.translation()), axis=-1,
    ) * 1000

    # ── 5. Save error plot ───────────────────────────────────────────

    OUTPUT_DIR.mkdir(exist_ok=True)
    _save_error_plot(target_link_names, pos_err, ori_err, init_pos_err,
                     mode, batch_size, elapsed)

    # ── 6. Viser (optional) ─────────────────────────────────────────

    if not vis:
        return

    import viser
    from viser.extras import ViserUrdf

    n_vis = min(batch_size, MAX_VIS_SAMPLES)
    if n_vis < batch_size:
        print(f"  viser: showing first {n_vis} of {batch_size} samples (max {MAX_VIS_SAMPLES})")

    server = viser.ViserServer(host="0.0.0.0", port=8080)

    spacing = 1.5
    grid_w = n_vis * spacing
    server.scene.add_grid("/ground", width=grid_w + 1, height=2, cell_size=0.25)

    robot_handles = []

    for i in range(n_vis):
        ox = i * spacing - (n_vis - 1) * spacing / 2

        base = f"/batch_{i}"
        server.scene.add_frame(base, show_axes=False, position=(ox, 0, 0))
        server.scene.add_label(f"{base}/label", f"#{i}", position=(0, 0, 1.15))

        u_gt = ViserUrdf(server, urdf, root_node_name=f"{base}/gt",
                         mesh_color_override=(0.2, 0.85, 0.3, 0.35))
        u_init = ViserUrdf(server, urdf, root_node_name=f"{base}/init",
                           mesh_color_override=(0.9, 0.25, 0.2, 0.25))
        u_solved = ViserUrdf(server, urdf, root_node_name=f"{base}/solved",
                             mesh_color_override=(0.25, 0.5, 0.95, 0.9))

        tgt_handles = []
        for name in target_link_names:
            h = server.scene.add_frame(
                f"{base}/target_{name}", show_axes=True,
                axes_length=0.10, axes_radius=0.004,
            )
            tgt_handles.append(h)

        robot_handles.append((u_gt, u_init, u_solved, tgt_handles))

        u_gt.update_cfg(gt_cfgs[i])
        u_init.update_cfg(initial_cfgs[i])
        u_solved.update_cfg(solved_cfgs[i])
        for j, h in enumerate(tgt_handles):
            h.position = tuple(target_positions[i, j].tolist())
            h.wxyz = tuple(target_wxyzs[i, j].tolist())

    # ── GUI controls ─────────────────────────────────────────────────

    show_gt = server.gui.add_checkbox("Show GT (green)", initial_value=True)
    show_init = server.gui.add_checkbox("Show Init (red)", initial_value=True)
    show_solved = server.gui.add_checkbox("Show Solved (blue)", initial_value=True)
    show_targets = server.gui.add_checkbox("Show target frames", initial_value=True)

    @show_gt.on_update
    def _(_):
        for i in range(n_vis):
            server.scene.set_visibility(f"/batch_{i}/gt", show_gt.value)

    @show_init.on_update
    def _(_):
        for i in range(n_vis):
            server.scene.set_visibility(f"/batch_{i}/init", show_init.value)

    @show_solved.on_update
    def _(_):
        for i in range(n_vis):
            server.scene.set_visibility(f"/batch_{i}/solved", show_solved.value)

    @show_targets.on_update
    def _(_):
        for i in range(n_vis):
            for name in target_link_names:
                server.scene.set_visibility(f"/batch_{i}/target_{name}", show_targets.value)

    lines = [f"### Batched IK  |  mode={mode}  B={batch_size}  "
             f"(showing {n_vis})\n"]
    lines.append(f"Solve time: **{elapsed*1000:.1f} ms** for {batch_size} samples\n")
    lines.append("| # | Pos err (mm) | Ori err (deg) | Init err (mm) |")
    lines.append("|---|-------------|--------------|--------------|")
    for i in range(n_vis):
        lines.append(
            f"| {i} | {pos_err[i].mean():.3f} | "
            f"{ori_err[i].mean():.3f} | {init_pos_err[i].mean():.1f} |"
        )
    lines.append(f"\n**Avg (all {batch_size}):** "
                 f"pos={pos_err.mean():.3f} mm  ori={ori_err.mean():.3f} deg")
    lines.append(f"\n<span style='color:#33D94D'>**green**</span>=GT  "
                 f"<span style='color:#E64033'>**red**</span>=Init  "
                 f"<span style='color:#4080F2'>**blue**</span>=Solved")
    server.gui.add_markdown("\n".join(lines))

    print(f"\n  Viser: http://localhost:8080")
    print(f"  {n_vis} samples shown. Toggle visibility to compare.")
    print(f"  Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


# ── Error plot ────────────────────────────────────────────────────────


def _save_error_plot(link_names, pos_err, ori_err, init_pos_err, mode, B, elapsed):
    n = len(link_names)
    x = np.arange(n)
    w = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Batched Multi-Target IK  |  mode={mode}  B={B}  "
        f"avg solve={elapsed*1000:.1f}ms", fontsize=13,
    )

    ax = axes[0]
    ax.bar(x - w/2, init_pos_err.mean(axis=0), w, label="Before IK", color="#DD8452", alpha=0.8)
    ax.bar(x + w/2, pos_err.mean(axis=0), w, label="After IK", color="#4C72B0", alpha=0.8)
    ax.errorbar(x + w/2, pos_err.mean(axis=0), yerr=pos_err.std(axis=0),
                fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(link_names, rotation=25, ha="right")
    ax.set_ylabel("Position error (mm)")
    ax.set_title("Position error")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, ori_err.mean(axis=0), color="#55A868", alpha=0.8)
    ax.errorbar(x, ori_err.mean(axis=0), yerr=ori_err.std(axis=0),
                fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(link_names, rotation=25, ha="right")
    ax.set_ylabel("Orientation error (deg)")
    ax.set_title("Orientation error (after IK)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    samples = np.arange(pos_err.shape[0])
    ax.bar(samples - w/2, init_pos_err.mean(axis=1), w,
           label="Before IK", color="#DD8452", alpha=0.8)
    ax.bar(samples + w/2, pos_err.mean(axis=1), w,
           label="After IK", color="#4C72B0", alpha=0.8)
    ax.set_xticks(samples)
    ax.set_xlabel("Batch sample")
    ax.set_ylabel("Avg position error (mm)")
    ax.set_title("Per-sample error (before vs after IK)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / f"14_batch_ik_{mode}_B{B}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched multi-target IK round-trip")
    parser.add_argument("--batch_size", "-B", type=int, default=8)
    parser.add_argument(
        "--mode", choices=["pos", "pose"], default="pose",
        help="pos: position-only IK; pose: position+orientation IK",
    )
    parser.add_argument(
        "--vis", action="store_true",
        help=f"Launch viser 3D viewer (shows up to {MAX_VIS_SAMPLES} samples)",
    )
    args = parser.parse_args()
    main(batch_size=args.batch_size, mode=args.mode, vis=args.vis)
