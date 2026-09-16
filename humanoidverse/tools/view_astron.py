"""Load only the Astron robot into a MuJoCo viewer.

Example::

    conda activate ufo
    cd /home/ubt/rhys/UFO
    MUJOCO_GL=glfw python -m humanoidverse.tools.view_astron
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

UFO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = UFO_ROOT / "humanoidverse/data/robots/astron/walker_astron_v1.xml"

# Standing pose from humanoidverse/config/robot/astron/astron_auto.yaml
DEFAULT_JOINT_ANGLES = {
    "L_hip_pitch_joint": -0.1,
    "L_hip_roll_joint": 0.015,
    "L_hip_yaw_joint": 0.0,
    "L_knee_pitch_joint": 0.2,
    "L_ankle_pitch_joint": -0.1,
    "L_ankle_roll_joint": 0.0,
    "R_hip_pitch_joint": -0.1,
    "R_hip_roll_joint": -0.015,
    "R_hip_yaw_joint": 0.0,
    "R_knee_pitch_joint": 0.2,
    "R_ankle_pitch_joint": -0.1,
    "R_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_roll_joint": 0.0,
    "L_shoulder_pitch_joint": 0.0,
    "L_shoulder_roll_joint": 0.087,
    "L_shoulder_yaw_joint": 0.0,
    "L_elbow_pitch_joint": 0.0,
    "L_elbow_yaw_joint": 0.0,
    "L_wrist_pitch_joint": 0.0,
    "L_wrist_roll_joint": 0.0,
    "R_shoulder_pitch_joint": 0.0,
    "R_shoulder_roll_joint": -0.087,
    "R_shoulder_yaw_joint": 0.0,
    "R_elbow_pitch_joint": 0.0,
    "R_elbow_yaw_joint": 0.0,
    "R_wrist_pitch_joint": 0.0,
    "R_wrist_roll_joint": 0.0,
}
DEFAULT_ROOT_POS = np.array([0.0, 0.0, 0.9], dtype=np.float64)
DEFAULT_ROOT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return name or f"<{int(obj_id)}>"


def apply_default_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            addr = int(model.jnt_qposadr[joint_id])
            data.qpos[addr : addr + 3] = DEFAULT_ROOT_POS
            data.qpos[addr + 3 : addr + 7] = DEFAULT_ROOT_QUAT_WXYZ
            break
    for joint_name, angle in DEFAULT_JOINT_ANGLES.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(angle)
    mujoco.mj_forward(model, data)


def print_model_summary(model: mujoco.MjModel) -> None:
    colliding = []
    visual_only = []
    for geom_id in range(model.ngeom):
        name = _mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body = _mj_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        item = f"{name} (body={body}, contype={contype}, conaffinity={conaffinity})"
        if contype or conaffinity:
            colliding.append(item)
        else:
            visual_only.append(item)
    print(f"nq={model.nq}  nv={model.nv}  nbody={model.nbody}  njnt={model.njnt}  ngeom={model.ngeom}")
    print(f"collision geoms: {len(colliding)}  visual-only geoms: {len(visual_only)}")
    print("MuJoCo XML self-collision is enabled when colliding geoms have overlapping contype/conaffinity.")
    print("Note: Hydra `robot.asset.self_collisions` is an Isaac Gym filter flag, not this MJCF.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="Astron MJCF path (default: humanoidverse/data/robots/astron/walker_astron_v1.xml)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run physics (robot will fall). Default holds the standing pose.",
    )
    args = parser.parse_args(argv)

    xml_path = args.xml.expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"Astron XML not found: {xml_path}")

    print(f"Loading {xml_path}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    apply_default_pose(model, data)
    print_model_summary(model)

    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        print("[INFO] Viewer opened. Close the window or Ctrl+C to exit.", flush=True)
        while viewer.is_running():
            if args.simulate:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.", flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
