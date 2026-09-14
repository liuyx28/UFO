"""Kinematic MuJoCo playback of UFO motion pkls (no dynamics / no policy).

Example::

    conda activate ufo
    cd /path/to/UFO
    MUJOCO_GL=egl python -m humanoidverse.tools.preview_ufo_pkl \\
      --pkl humanoidverse/data/motion_lib_astron/full_ufo.pkl \\
      --robot-config configs/robots/astron.yaml \\
      --out-dir humanoidverse/data/motion_lib_astron/preview
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import mediapy as media
import numpy as np

from humanoidverse.mjlab_inference_utils import MujocoQposRenderer
from humanoidverse.utils.robot_spec import load_robot_training_spec, resolve_robot_config_path


def _ensure_ffmpeg() -> None:
    """Prefer system ffmpeg; fall back to imageio-ffmpeg's bundled binary."""
    try:
        media._get_ffmpeg_path()
        return
    except Exception:
        pass
    try:
        import imageio_ffmpeg

        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found. Install system ffmpeg or `pip/conda install imageio-ffmpeg`."
        ) from exc


def _control_to_qpos_order_indices(robot_training: Any) -> tuple[np.ndarray, list[str]]:
    control_joint_names = list(robot_training.robot.control_joint_names)
    qpos_joint_names = sorted(control_joint_names, key=lambda joint: robot_training.robot.joint_qpos_addr[joint])
    index_by_control_joint = {joint: idx for idx, joint in enumerate(control_joint_names)}
    return np.asarray([index_by_control_joint[joint] for joint in qpos_joint_names], dtype=np.int64), qpos_joint_names


def _motion_qpos(record: dict[str, Any], dof_qpos_order_indices: np.ndarray) -> tuple[np.ndarray, float]:
    root_pos = np.asarray(record["root_trans_offset"], dtype=np.float64)
    if "root_rot" in record:
        root_rot_xyzw = np.asarray(record["root_rot"], dtype=np.float64)
    else:
        raise KeyError("Motion record missing root_rot")
    dof = np.asarray(record.get("dof_pos", record.get("dof")), dtype=np.float64)
    if dof.ndim != 2:
        raise ValueError(f"dof must be [T, ndof], got {dof.shape}")
    root_quat_wxyz = np.concatenate([root_rot_xyzw[..., 3:4], root_rot_xyzw[..., :3]], axis=-1)
    dof_qpos = dof[:, dof_qpos_order_indices]
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_qpos], axis=-1)
    fps = float(record.get("fps", 30.0))
    return qpos, fps


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--motion-list", type=int, nargs="*", default=None, help="Indices into sorted motion keys; default = all")
    parser.add_argument("--max-motions", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    args = parser.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    _ensure_ffmpeg()

    pkl_path = args.pkl.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    robot_config = resolve_robot_config_path(args.robot_config)
    robot_training = load_robot_training_spec(robot_config)
    xml_path = Path(robot_training.robot.xml_path).expanduser().resolve()
    control_joints = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joints)
    dof_qpos_order_indices, qpos_joint_names = _control_to_qpos_order_indices(robot_training)

    motions = joblib.load(pkl_path)
    if not isinstance(motions, dict):
        raise TypeError(f"Expected dict pkl, got {type(motions)}")
    keys = sorted(motions.keys())
    indices = list(range(len(keys))) if args.motion_list is None else list(args.motion_list)
    if args.max_motions is not None:
        indices = indices[: args.max_motions]

    print(f"[preview] pkl={pkl_path}")
    print(f"[preview] robot={robot_config} xml={xml_path}")
    print(f"[preview] motions={len(keys)} selected={len(indices)} control_dof={num_dof}")
    if qpos_joint_names != control_joints:
        print(f"[preview] qpos joint order differs from control order: {qpos_joint_names}")

    renderer = MujocoQposRenderer(
        xml_path,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        expected_qpos_size=7 + num_dof,
    )

    meta_items: list[dict[str, Any]] = []
    try:
        for rank, idx in enumerate(indices):
            if idx < 0 or idx >= len(keys):
                raise IndexError(f"motion index {idx} out of range [0, {len(keys)})")
            key = keys[idx]
            record = motions[key]
            qpos, fps = _motion_qpos(record, dof_qpos_order_indices)
            n_frames = qpos.shape[0] if args.max_steps is None else min(qpos.shape[0], int(args.max_steps))
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:80]
            out_path = out_dir / f"{rank:02d}_{idx}_{safe}.mp4"
            print(f"[preview] [{rank+1}/{len(indices)}] idx={idx} key={key} frames={n_frames} fps={fps} -> {out_path.name}")
            frames = [renderer.render_qpos(qpos[t]) for t in range(n_frames)]
            media.write_video(str(out_path), frames, fps=fps)
            meta_items.append(
                {
                    "index": int(idx),
                    "key": key,
                    "file": out_path.name,
                    "frames": int(n_frames),
                    "fps": float(fps),
                    "seconds": float(n_frames) / float(fps),
                    "bytes": int(out_path.stat().st_size),
                }
            )
    finally:
        renderer.close()

    meta_path = out_dir / "_preview_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "source_pkl": str(pkl_path),
                "robot_config": str(robot_config),
                "xml_path": str(xml_path),
                "items": meta_items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[preview] wrote {len(meta_items)} videos under {out_dir}")
    print(f"[preview] meta={meta_path}")


if __name__ == "__main__":
    main()
