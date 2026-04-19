"""Domino stacking with web UI — sim, real, or both.

Self-contained ROS1 node that:
  1. Serves the domino-placement web UI on http://localhost:5000 (visualized.html).
  2. Receives N domino (x, y, yaw) targets from the UI.
  3. For each target: picks a block from the shelf and places it standing at
     the UI pose, using on-the-fly IK against a MuJoCo planning model.
  4. After the last placement, knocks the first domino to start the chain.

Execution backend is selectable via CLI flags:
  --sim / --no-sim      (default: --sim)    visualize + torque-track in MuJoCo
  --real / --no-real    (default: --no-real) drive the real arm via frankapy

Default mode is sim-only. Pass --real to enable the arm; pass --no-sim if you
only want the real arm. Both flags on runs sim first then real per segment
(useful as a visual preview before committing to the physical motion).

This module is intentionally independent of dominoes.pickup — the IK solver,
planning-model builder, and side-pick motion plan are inlined below.
"""
import argparse
import os
import sys
import threading
import time
import types
import json as _json
import webbrowser
import xml.etree.ElementTree as ET
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import mujoco as mj
from mujoco import viewer as _mjviewer


WAYPOINTS = np.array(
    [[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8, 0.1]],
    dtype=float,
)
HOME_QPOS = WAYPOINTS[0][:7].copy()

# Pickup order copied from integration_simulation.py (24 blocks). Picks the
# top block of each stack first to avoid disturbing lower blocks, then works
# down. Truncated to len(BLOCK_ORDER) if the UI sends more dominoes.
BLOCK_ORDER = [
    "RBottomFar3",
    "RBottomClose3",
    "LBottomClose3",
    "LBottomFar3",
    "RMiddleFar3", "RMiddleFar2", "RMiddleFar1",
    "RMiddleClose3", "RMiddleClose2", "RMiddleClose1",
    "RTopFar3", "RTopFar2", "RTopFar1",
    "RTopClose3", "RTopClose2", "RTopClose1",
    "RBottomFar2", "RBottomFar1",
    "RBottomClose2", "RBottomClose1",
    "LBottomClose2", "LBottomClose1",
    "LBottomFar2", "LBottomFar1",
]

END_OF_TABLE = 0.55 + 0.135 + 0.05

GRIPPER_OPEN   = 0.08
GRIPPER_CLOSED = 0.0
OPEN_THRESHOLD = GRIPPER_OPEN - 1e-4

MOVE_DURATION       = 3.0
CLAMP_MOVE_DURATION = 0.1
POST_CLAMP_WAIT     = 0.5
HOME_TOL            = 1e-2

# Sim-side constants (used only when --sim is active).
KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)
SIM_GRIPPER_OPEN    = 0.04
SIM_GRIPPER_CLOSED  = 0.015
SIM_CLAMP_DURATION  = 1.0

PLACED_GRASP_Z = 0.235
JOINT7_LIMIT   = 2.8973

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HTML_PATH  = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'config', 'visualized.html'))

_ui_positions = []
_start_event  = threading.Event()
_stop_event   = threading.Event()


# Stdlib shims — the module intentionally does NOT import rospy at the top so
# `python3 integration_real.py` works in sim mode on a machine without ROS
# installed. Under --real, main() imports rospy lazily for ROS logging;
# these shims are a no-op fallback everywhere else.
def _is_shutdown():
    return False


def _log(msg, *args):
    print(msg % args if args else msg)


def _logwarn(msg, *args):
    print("WARN: " + (msg % args if args else msg))


def _logerr(msg, *args):
    print("ERROR: " + (msg % args if args else msg))


class _UIHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args): pass

    def do_GET(self):
        if self.path == '/':
            with open(_HTML_PATH, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        resp = b'{"ok":true}'
        if self.path == '/start':
            n = int(self.headers.get('Content-Length', 0))
            global _ui_positions
            _ui_positions = _json.loads(self.rfile.read(n))
            _stop_event.clear()
            _start_event.set()
        elif self.path == '/stop':
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            _stop_event.set()
        else:
            resp = b'{"ok":false}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(resp))
        self.end_headers()
        self.wfile.write(resp)


# ---- MuJoCo planning model (self-contained; no dependency on pickup.py) ----

def _get_model_directory():
    """Locate the franka_emika_panda folder. Prefer the catkin-installed share
    path (rospkg) so this works from an install tree; fall back to the path
    relative to this source file for local dev."""
    try:
        import rospkg
        share_dir = Path(rospkg.RosPack().get_path("dominoes"))
        candidate = share_dir / "franka_emika_panda"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    return Path(__file__).resolve().parents[1] / "franka_emika_panda"


def _add_free_block_to_model(tree, name, pos, density, size, rgba, free):
    worldbody = tree.getroot().find("worldbody")
    body = ET.SubElement(
        worldbody, "body",
        {"name": name, "pos": f"{pos[0]} {pos[1]} {pos[2]}"},
    )
    ET.SubElement(
        body, "geom",
        {
            "type": "box",
            "density": f"{density}",
            "size": f"{size[0]} {size[1]} {size[2]}",
            "rgba": f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
        },
    )
    if free:
        ET.SubElement(body, "freejoint")


def _add_static_scene(tree, end_of_table):
    """Static shelves + 24 free blocks (three shelves on the right, one on the
    left with near/far piles of 3 blocks each). Geometry mirrors pickup.py."""
    shelves = [
        ["TablePlane", [end_of_table - 0.275, 0.0, -0.005], [0.275, 0.504, 0.0051]],
        ["LShelfDistal", [end_of_table - 0.09 - 0.0225, 0.504 - 0.045 - 0.0225, 0.315], [0.0225, 0.0225, 0.315]],
        ["LShelfProximal", [end_of_table - 0.55 - 0.0225, 0.504 - 0.045 - 0.0225, 0.3825 - 0.135], [0.0225, 0.0225, 0.3825]],
        ["LShelfBack", [end_of_table - 0.55 - 0.0225 - 0.09, 0.504 - 0.045 - 0.0225, 0.3825 - 0.135], [0.0225, 0.0225, 0.3825]],
        ["LShelfMid", [end_of_table - 0.32, 0.504 - 0.045 - 0.0225, 0.315], [0.0225, 0.0225, 0.315]],
        ["LShelfArch", [end_of_table - 0.275 - 0.135 + 0.0225, 0.504 - 0.045 - 0.0225, 0.63 + 0.0225], [0.315, 0.0225, 0.0225]],
        ["LShelfBottom", [end_of_table - 0.275 - 0.135 + 0.0225, 0.504 - 0.09 - 0.135 / 2.0, 0.1375 + 0.005], [0.2525, 0.135 / 2.0, 0.005]],
        ["LShelfBottomSupp1", [end_of_table - 0.55 - 0.0225 - 0.09 + 0.045, 0.504 - 0.225 / 2.0, 0.1375 - 0.0225], [0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSupp2", [end_of_table - 0.32 - 0.045, 0.504 - 0.225 / 2.0, 0.1375 - 0.0225], [0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSupp3", [end_of_table - 0.09 - 0.0225 - 0.045, 0.504 - 0.225 / 2.0, 0.1375 - 0.0225], [0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSuppB", [end_of_table - 0.275 - 0.135 + 0.0225, 0.504 - 0.0225, 0.1375 + 0.0225], [0.315, 0.0225, 0.0225]],
        ["RShelfDistal", [end_of_table - 0.09 - 0.0225, -0.504 + 0.045 + 0.0225, 0.315], [0.0225, 0.0225, 0.315]],
        ["RShelfProximal", [end_of_table - 0.55 - 0.0225, -0.504 + 0.045 + 0.0225, 0.3825 - 0.135], [0.0225, 0.0225, 0.3825]],
        ["RShelfBack", [end_of_table - 0.55 - 0.0225 - 0.09, -0.504 + 0.045 + 0.0225, 0.3825 - 0.135], [0.0225, 0.0225, 0.3825]],
        ["RShelfMid", [end_of_table - 0.32, -0.504 + 0.045 + 0.0225, 0.315], [0.0225, 0.0225, 0.315]],
        ["RShelfArch", [end_of_table - 0.275 - 0.135 + 0.0225, -0.504 + 0.045 + 0.0225, 0.63 + 0.0225], [0.315, 0.0225, 0.0225]],
        ["RShelfBottom", [end_of_table - 0.275 - 0.135 + 0.0225, -0.504 + 0.09 + 0.135 / 2.0, 0.1375 + 0.005], [0.2525, 0.135 / 2.0, 0.005]],
        ["RShelfBottomSupp1", [end_of_table - 0.55 - 0.0225 - 0.09 + 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225], [0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSupp2", [end_of_table - 0.32 - 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225], [0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSupp3", [end_of_table - 0.09 - 0.0225 - 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225], [0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSuppB", [end_of_table - 0.275 - 0.135 + 0.0225, -0.504 + 0.0225, 0.1375 + 0.0225], [0.315, 0.0225, 0.0225]],
        ["RShelfMiddle", [end_of_table - 0.275 - 0.135 + 0.0225, -0.504 + 0.09 + 0.135 / 2.0, 0.1375 + 0.005 + 0.2], [0.2525, 0.135 / 2.0, 0.005]],
        ["RShelfMiddleSupp1", [end_of_table - 0.55 - 0.0225 - 0.09 + 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225 + 0.2], [0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSupp2", [end_of_table - 0.32 - 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225 + 0.2], [0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSupp3", [end_of_table - 0.09 - 0.0225 - 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225 + 0.2], [0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSuppB", [end_of_table - 0.275 - 0.135 + 0.0225, -0.504 + 0.0225, 0.1375 + 0.0225 + 0.2], [0.315, 0.0225, 0.0225]],
        ["RShelfTop", [end_of_table - 0.275 - 0.135 + 0.0225, -0.504 + 0.09 + 0.135 / 2.0, 0.1375 + 0.005 + 0.4], [0.2525, 0.135 / 2.0, 0.005]],
        ["RShelfTopSupp1", [end_of_table - 0.55 - 0.0225 - 0.09 + 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225 + 0.4], [0.0225, 0.1125, 0.0225]],
        ["RShelfTopSupp2", [end_of_table - 0.32 - 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225 + 0.4], [0.0225, 0.1125, 0.0225]],
        ["RShelfTopSupp3", [end_of_table - 0.09 - 0.0225 - 0.045, -0.504 + 0.225 / 2.0, 0.1375 - 0.0225 + 0.4], [0.0225, 0.1125, 0.0225]],
        ["RShelfTopSuppB", [end_of_table - 0.275 - 0.135 + 0.0225, -0.504 + 0.0225, 0.1375 + 0.0225 + 0.4], [0.315, 0.0225, 0.0225]],
    ]
    for name, pos, size in shelves:
        _add_free_block_to_model(tree, name, pos, 20, size, [0.2, 0.2, 0.9, 1.0], False)

    box_size = [0.025, 0.075, 0.015]
    box_rgba = [0.0, 0.9, 0.2, 1.0]

    left_shelf_y      = 0.504 - 0.09 - 0.135 / 2.0
    left_shelf_top_z  = 0.1375 + 0.005 + 0.005
    left_far_x        = end_of_table - 0.135 - 0.15
    left_close_x      = (end_of_table - 0.135 - 2 * 0.2525) + 0.15

    right_shelf_y         = -0.504 + 0.09 + 0.135 / 2.0
    right_far_x           = end_of_table - 0.135 - 0.15
    right_close_x         = (end_of_table - 0.135 - 2 * 0.2525) + 0.15
    rshelf_bottom_top_z   = 0.1375 + 0.005 + 0.005
    rshelf_middle_top_z   = 0.1375 + 0.005 + 0.2 + 0.005
    rshelf_top_top_z      = 0.1375 + 0.005 + 0.4 + 0.005

    _add_free_block_to_model(tree, "LBottomFar1",   [left_far_x, left_shelf_y, left_shelf_top_z + 0.015], 20, box_size, box_rgba, True)
    _add_free_block_to_model(tree, "LBottomFar2",   [left_far_x, left_shelf_y, left_shelf_top_z + 0.045], 20, box_size, box_rgba, True)
    _add_free_block_to_model(tree, "LBottomFar3",   [left_far_x, left_shelf_y, left_shelf_top_z + 0.075], 20, box_size, box_rgba, True)
    _add_free_block_to_model(tree, "LBottomClose1", [left_close_x, left_shelf_y, left_shelf_top_z + 0.015], 20, box_size, box_rgba, True)
    _add_free_block_to_model(tree, "LBottomClose2", [left_close_x, left_shelf_y, left_shelf_top_z + 0.045], 20, box_size, box_rgba, True)
    _add_free_block_to_model(tree, "LBottomClose3", [left_close_x, left_shelf_y, left_shelf_top_z + 0.075], 20, box_size, box_rgba, True)

    for shelf_prefix, shelf_top_z in (
        ("RBottom", rshelf_bottom_top_z),
        ("RMiddle", rshelf_middle_top_z),
        ("RTop",    rshelf_top_top_z),
    ):
        _add_free_block_to_model(tree, f"{shelf_prefix}Far1",   [right_far_x,   right_shelf_y, shelf_top_z + 0.015], 20, box_size, box_rgba, True)
        _add_free_block_to_model(tree, f"{shelf_prefix}Far2",   [right_far_x,   right_shelf_y, shelf_top_z + 0.045], 20, box_size, box_rgba, True)
        _add_free_block_to_model(tree, f"{shelf_prefix}Far3",   [right_far_x,   right_shelf_y, shelf_top_z + 0.075], 20, box_size, box_rgba, True)
        _add_free_block_to_model(tree, f"{shelf_prefix}Close1", [right_close_x, right_shelf_y, shelf_top_z + 0.015], 20, box_size, box_rgba, True)
        _add_free_block_to_model(tree, f"{shelf_prefix}Close2", [right_close_x, right_shelf_y, shelf_top_z + 0.045], 20, box_size, box_rgba, True)
        _add_free_block_to_model(tree, f"{shelf_prefix}Close3", [right_close_x, right_shelf_y, shelf_top_z + 0.075], 20, box_size, box_rgba, True)


def _build_planning_model():
    """Parse the root MuJoCo XML, inject shelves + free blocks, write the
    generated XML next to the root (so mesh paths still resolve), then load
    it into a MuJoCo model for IK. Returns (model, data, arm_idx)."""
    model_dir = _get_model_directory()
    root_xml = model_dir / "panda_torque_table.xml"
    generated_xml = model_dir / "panda_torque_table_final.xml"

    tree = ET.parse(root_xml)
    _add_static_scene(tree, END_OF_TABLE)
    tree.write(generated_xml, encoding="utf-8", xml_declaration=True)

    model = mj.MjModel.from_xml_path(str(generated_xml))
    data = mj.MjData(model)
    arm_idx = [0, 1, 2, 3, 4, 5, 6]

    data.qpos[arm_idx] = WAYPOINTS[0][:7]
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)
    return model, data, arm_idx


# ---- IK solver (damped least squares) ----

def calculate_ik_6d(model, data, target_pos, target_direction,
                    body_name="hand", max_iters=500, tol=1e-3, step_size=0.2):
    """Drives the hand body toward target_pos with its z-axis aligned to
    target_direction. Seeds from the current data.qpos and restores it on
    return; caller is responsible for setting the desired seed first."""
    target_dir = np.array(target_direction, dtype=float)
    target_dir /= np.linalg.norm(target_dir)

    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")

    dof_indices = [0, 1, 2, 3, 4, 5, 6]
    jnt_lo = model.jnt_range[dof_indices, 0]
    jnt_hi = model.jnt_range[dof_indices, 1]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    qpos0 = data.qpos.copy()

    for _ in range(max_iters):
        mj.mj_forward(model, data)
        err_pos = target_pos - data.xpos[body_id]
        current_dir = data.xmat[body_id].reshape(3, 3)[:, 2]
        err_rot = np.cross(current_dir, target_dir)
        err = np.hstack((err_pos, err_rot))

        if np.linalg.norm(err) < tol:
            break

        mj.mj_jacBody(model, data, jacp, jacr, body_id)
        jacobian = np.vstack((jacp[:, dof_indices], jacr[:, dof_indices]))
        jacobian_pinv = jacobian.T @ np.linalg.inv(
            jacobian @ jacobian.T + 1e-4 * np.eye(6)
        )
        data.qpos[dof_indices] += step_size * (jacobian_pinv @ err)
        data.qpos[dof_indices] = np.clip(data.qpos[dof_indices], jnt_lo, jnt_hi)

    solved_qpos = data.qpos[dof_indices].copy()
    data.qpos[:] = qpos0
    mj.mj_forward(model, data)
    return solved_qpos


# ---- pickup motion plan (real-arm-verified offsets from pickup.py) ----

def _plan_side_pick(model, data, arm_idx, home_qpos, block_name, block_id):
    """Return (pregrasp_q, grasp_q, lift_q, pullout_q) for one shelf block.

    Left-side blocks are approached from +Y, right-side from -Y. After each
    IK solve, joint 7 is rotated by -π/2 to orient the gripper perpendicular
    to the domino's long axis (matches pickup.py)."""
    data.qpos[arm_idx] = home_qpos
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)

    block_pos = data.xpos[block_id].copy()

    if block_name.startswith("L"):
        side_dir = np.array([0.0, 1.0, 0.0])
        pregrasp_xyz = block_pos + np.array([0.0, -0.20, 0.0])
        grasp_xyz    = block_pos + np.array([0.0, -0.12, 0.0])
        pullout_xyz  = block_pos + np.array([0.0, -0.40, 0.0])
    else:
        side_dir = np.array([0.0, -1.0, 0.0])
        pregrasp_xyz = block_pos + np.array([0.0, 0.20, 0.0])
        grasp_xyz    = block_pos + np.array([0.0, 0.12, 0.0])
        pullout_xyz  = block_pos + np.array([0.0, 0.40, 0.0])
    lift_xyz = grasp_xyz + np.array([0.0, 0.0, 0.02])

    pregrasp_q = calculate_ik_6d(model, data, pregrasp_xyz, side_dir)
    grasp_q    = calculate_ik_6d(model, data, grasp_xyz,    side_dir)
    lift_q     = calculate_ik_6d(model, data, lift_xyz,     side_dir)
    pullout_q  = calculate_ik_6d(model, data, pullout_xyz,  side_dir)

    for q in (pregrasp_q, grasp_q, lift_q, pullout_q):
        q[6] -= np.pi / 2

    return pregrasp_q, grasp_q, lift_q, pullout_q


# ---- placement & knock IK (no torque control, no viewer) ----

def _is_at_home(joints):
    return all(abs(a - b) < HOME_TOL for a, b in zip(joints, HOME_QPOS))


def _ik_chain(model, data, arm_idx, seed_q, targets):
    """Solve IK for a sequence of (xyz, direction) targets, progressively
    seeding each solve from the previous solution. Snapshots data.qpos/qvel
    on entry and restores them on exit so planning never corrupts the live
    sim state (the executor reads q_start from data.qpos)."""
    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()
    try:
        data.qpos[arm_idx] = np.asarray(seed_q, dtype=float)
        data.qvel[arm_idx] = 0.0
        mj.mj_forward(model, data)
        solutions = []
        for xyz, direction in targets:
            q = calculate_ik_6d(model, data, xyz, direction)
            solutions.append(q)
            data.qpos[arm_idx] = q
            mj.mj_forward(model, data)
        return solutions
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        mj.mj_forward(model, data)


def _normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _choose_joint7_for_yaw(raw_place_j7, target_yaw, current_j7):
    raw_j7 = raw_place_j7 + target_yaw
    cand_a = _normalize_angle(raw_j7)
    cand_b = _normalize_angle(raw_j7 + np.pi)

    def score(c):
        if abs(c) > JOINT7_LIMIT:
            return float('inf')
        return abs(c - current_j7)

    return min([cand_a, cand_b], key=score)


def plan_pickup_waypoints(model, data, arm_idx, block_name, block_id):
    """Wrap _plan_side_pick's four joint targets as (joints, gripper, duration)
    tuples for the executor, including the open->closed clamp step."""
    pregrasp_q, grasp_q, lift_q, pullout_q = _plan_side_pick(
        model, data, arm_idx, HOME_QPOS, block_name, block_id
    )
    return [
        (HOME_QPOS,   GRIPPER_OPEN,   MOVE_DURATION),
        (pregrasp_q,  GRIPPER_OPEN,   MOVE_DURATION),
        (grasp_q,     GRIPPER_OPEN,   MOVE_DURATION),
        (grasp_q,     GRIPPER_CLOSED, CLAMP_MOVE_DURATION),
        (lift_q,      GRIPPER_CLOSED, MOVE_DURATION),
        (pullout_q,   GRIPPER_CLOSED, MOVE_DURATION),
        (HOME_QPOS,   GRIPPER_CLOSED, MOVE_DURATION),
    ]


def plan_place_domino_standing(model, data, arm_idx, current_q, target_xy, target_yaw):
    """Port of integration_simulation.py:172-210 minus the torque-control loop.
    Returns waypoints for transit -> preplace -> place -> release -> preplace -> home."""
    down_dir = np.array([0.0, 0.0, -1.0])
    transit_xyz  = np.array([target_xy[0], target_xy[1], 0.45])
    preplace_xyz = np.array([target_xy[0], target_xy[1], PLACED_GRASP_Z + 0.10])
    place_xyz    = np.array([target_xy[0], target_xy[1], PLACED_GRASP_Z])

    transit_q, preplace_q, place_q = _ik_chain(
        model, data, arm_idx, current_q,
        [(transit_xyz, down_dir), (preplace_xyz, down_dir), (place_xyz, down_dir)],
    )

    final_j7 = _choose_joint7_for_yaw(place_q[6], target_yaw, current_q[6])
    transit_q[6]  = final_j7
    preplace_q[6] = final_j7
    place_q[6]    = final_j7

    return [
        (transit_q,  GRIPPER_CLOSED, MOVE_DURATION),
        (preplace_q, GRIPPER_CLOSED, MOVE_DURATION),
        (place_q,    GRIPPER_CLOSED, MOVE_DURATION),
        (place_q,    GRIPPER_OPEN,   CLAMP_MOVE_DURATION),
        (preplace_q, GRIPPER_OPEN,   MOVE_DURATION),
        (HOME_QPOS,  GRIPPER_OPEN,   MOVE_DURATION),
    ]


def plan_knock_first_domino(model, data, arm_idx, current_q, first_xy, first_yaw, next_xy):
    """Knock the first domino toward the second one. The strike direction is
    taken directly from (next_xy - first_xy), ignoring first_yaw. The arm
    approaches 2 cm in front of the first block along that direction, then
    slides in a straight line to the midpoint between the two blocks,
    dragging the closed gripper across the first block's top to tip it.
    Strike-plane z sits 2.5 cm above the first block's center (knock_z).
    Fallback: if next_xy is None (single-domino UI), derive direction from
    first_yaw so the knock still plays."""
    down_dir = np.array([0.0, 0.0, -1.0])
    first_xy_np = np.asarray(first_xy, dtype=float)

    if next_xy is not None:
        delta = np.asarray(next_xy, dtype=float) - first_xy_np
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            direction = np.array([-np.cos(first_yaw), np.sin(first_yaw)])
            mid_xy = first_xy_np + direction * 0.04
        else:
            direction = delta / dist
            mid_xy = (first_xy_np + np.asarray(next_xy, dtype=float)) / 2.0
    else:
        direction = np.array([-np.cos(first_yaw), np.sin(first_yaw)])
        mid_xy = first_xy_np + direction * 0.04

    # 2.5 cm above the first block's center (center of a standing domino sits
    # at PLACED_GRASP_Z in the planning model, since that's also where the
    # gripper held it during placement).
    knock_z = PLACED_GRASP_Z -0.025

    transit_xyz  = np.array([first_xy_np[0], first_xy_np[1], 0.45])
    preplace_xyz = np.array([first_xy_np[0]- direction[0]* 0.03, first_xy_np[1] - direction[1]* 0.03, knock_z + 0.10])
    prep_xyz     = np.array([first_xy_np[0] - direction[0] * 0.03,
                             first_xy_np[1] - direction[1] * 0.03,
                             knock_z])
    strike_xyz   = np.array([mid_xy[0], mid_xy[1], knock_z])

    transit_q, preplace_q, prep_q, strike_q = _ik_chain(
        model, data, arm_idx, current_q,
        [(transit_xyz, down_dir), (preplace_xyz, down_dir),
         (prep_xyz, down_dir), (strike_xyz, down_dir)],
    )

    # Match the wrist rotation used to place the first domino so the gripper
    # orientation over the block is identical. Mirrors the IK + selection
    # done inside plan_place_domino_standing for the place pose.
    place_xyz_for_j7 = np.array([first_xy_np[0], first_xy_np[1], PLACED_GRASP_Z])
    (place_q_for_j7,) = _ik_chain(
        model, data, arm_idx, current_q,
        [(place_xyz_for_j7, down_dir)],
    )
    placement_j7 = _choose_joint7_for_yaw(place_q_for_j7[6], first_yaw, current_q[6])
    transit_q[6]  = placement_j7
    preplace_q[6] = placement_j7
    prep_q[6]     = placement_j7
    strike_q[6]   = placement_j7

    return [
        (transit_q,  GRIPPER_CLOSED, MOVE_DURATION),
        (preplace_q, GRIPPER_CLOSED, MOVE_DURATION),
        (prep_q,     GRIPPER_CLOSED, MOVE_DURATION),
        (strike_q,   GRIPPER_CLOSED, MOVE_DURATION),
        (preplace_q, GRIPPER_CLOSED, MOVE_DURATION),
        (HOME_QPOS,  GRIPPER_CLOSED, MOVE_DURATION),
    ]


def _sim_run_segment(ctx, q_start, q_goal, gripper_pos, duration):
    """Torque-track from q_start to q_goal over `duration` seconds, holding the
    gripper at `gripper_pos`. Ported verbatim from integration_simulation.py,
    with _stop_event guard so the UI Stop button can interrupt mid-segment."""
    try:
        from dominoes import RobotUtil as rt
    except ImportError:
        import RobotUtil as rt

    n_steps = max(1, int(duration / ctx.dt))
    t = 0.0
    for _ in range(n_steps):
        if _stop_event.is_set():
            return
        q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, duration)
        q  = ctx.data.qpos[ctx.arm_idx].copy()
        qd = ctx.data.qvel[ctx.arm_idx].copy()
        tau = KP * (q_des - q) + KD * (qd_des - qd)
        ctx.data.ctrl[ctx.arm_idx]     = tau + ctx.data.qfrc_bias[:7]
        ctx.data.ctrl[ctx.gripper_idx] = gripper_pos
        mj.mj_step(ctx.model, ctx.data)
        ctx.viewer.sync()
        t += ctx.dt


def execute_waypoints(waypoints, sim_ctx=None, fa=None, prev_gripper_state="open"):
    """Drive the selected backends through (joints, gripper_width, duration)
    waypoints. Sim runs first (preview), then real (execute). Either side is
    skipped when its context is None. Gripper state machine for the real arm
    matches runner.py:91-107."""
    for joints, gripper_width, duration in waypoints:
        if _is_shutdown() or _stop_event.is_set():
            return prev_gripper_state

        joint_goal = list(joints)

        if sim_ctx is not None:
            q_start  = sim_ctx.data.qpos[sim_ctx.arm_idx].copy()
            sim_grip = SIM_GRIPPER_OPEN if gripper_width >= OPEN_THRESHOLD else SIM_GRIPPER_CLOSED
            sim_dur  = SIM_CLAMP_DURATION if duration < 0.2 else duration
            _sim_run_segment(sim_ctx, q_start, np.asarray(joint_goal, dtype=float),
                             sim_grip, sim_dur)
            if _stop_event.is_set():
                return prev_gripper_state

        if fa is not None:
            fa.goto_joints(joint_goal, duration=duration)

            desired = "open" if gripper_width >= OPEN_THRESHOLD else "closed"
            if desired != prev_gripper_state:
                if desired == "open":
                    fa.open_gripper()
                else:
                    fa.close_gripper(grasp=True)
                    time.sleep(POST_CLAMP_WAIT)
                prev_gripper_state = desired

            if _is_at_home(joint_goal) and gripper_width >= OPEN_THRESHOLD:
                fa.open_gripper()
                prev_gripper_state = "open"

    return prev_gripper_state


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="integration_real",
        description="Domino stacking with web UI (sim, real, or both).",
    )
    parser.add_argument('--sim',    dest='sim',  action='store_true')
    parser.add_argument('--no-sim', dest='sim',  action='store_false')
    parser.set_defaults(sim=True)
    parser.add_argument('--real',    dest='real', action='store_true')
    parser.add_argument('--no-real', dest='real', action='store_false')
    parser.set_defaults(real=False)
    return parser.parse_args(argv)


def _reset_sim_to_home(sim_ctx):
    sim_ctx.data.qpos[sim_ctx.arm_idx] = HOME_QPOS
    sim_ctx.data.qvel[sim_ctx.arm_idx] = 0.0
    mj.mj_forward(sim_ctx.model, sim_ctx.data)
    sim_ctx.viewer.sync()


def main():
    args = _parse_args(sys.argv[1:])
    if not (args.sim or args.real):
        _logerr("Both --sim and --real are disabled; nothing to run.")
        return

    _log("Mode: sim=%s real=%s", args.sim, args.real)

    model, data, arm_idx = _build_planning_model()

    block_ids = {}
    for name in BLOCK_ORDER:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in planning model.")
        block_ids[name] = bid

    sim_ctx = None
    if args.sim:
        data.qpos[arm_idx] = HOME_QPOS
        data.qvel[arm_idx] = 0.0
        mj.mj_forward(model, data)
        v = _mjviewer.launch_passive(model, data)
        v.cam.distance = 2.5
        v.cam.azimuth += 90
        v.sync()
        sim_ctx = types.SimpleNamespace(
            model=model, data=data, viewer=v,
            arm_idx=arm_idx, gripper_idx=7, dt=model.opt.timestep,
        )

    fa = None
    if args.real:
        import rospy
        rospy.init_node("integration_real", anonymous=False)
        from frankapy import FrankaArm
        fa = FrankaArm()
        fa.reset_joints(duration=MOVE_DURATION)
        fa.open_gripper()
    prev_gripper = "open"

    server = HTTPServer(('localhost', 5000), _UIHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _log("UI serving at http://localhost:5000 (source: %s)", _HTML_PATH)
    try:
        webbrowser.open('http://localhost:5000')
    except Exception:
        pass

    try:
        while not _is_shutdown():
            _log("Waiting for positions from UI...")
            while not _start_event.wait(timeout=0.5):
                if _is_shutdown():
                    return
            _start_event.clear()
            _stop_event.clear()

            domino_plan = [(np.array([p['x'], p['y']]), p['yaw']) for p in _ui_positions]
            if not domino_plan:
                continue

            n = min(len(domino_plan), len(BLOCK_ORDER))
            if len(domino_plan) > len(BLOCK_ORDER):
                _logwarn("UI sent %d dominoes; truncating to %d (shelf capacity).",
                              len(domino_plan), len(BLOCK_ORDER))

            _log("=== Starting %d domino(es) ===", n)

            for i in range(n):
                if _stop_event.is_set():
                    break
                block_name = BLOCK_ORDER[i]
                target_xy, target_yaw = domino_plan[i]

                _log("[%d/%d] Pick %s from shelf", i + 1, n, block_name)
                pickup_wps = plan_pickup_waypoints(model, data, arm_idx,
                                                   block_name, block_ids[block_name])
                prev_gripper = execute_waypoints(
                    pickup_wps, sim_ctx=sim_ctx, fa=fa,
                    prev_gripper_state=prev_gripper,
                )
                if sim_ctx is None:
                    data.qpos[arm_idx] = HOME_QPOS
                    mj.mj_forward(model, data)

                if _stop_event.is_set():
                    break

                _log("        Place at xy=(%.3f, %.3f) yaw=%.3f",
                              target_xy[0], target_xy[1], target_yaw)
                place_wps = plan_place_domino_standing(
                    model, data, arm_idx, HOME_QPOS, target_xy, target_yaw
                )
                prev_gripper = execute_waypoints(
                    place_wps, sim_ctx=sim_ctx, fa=fa,
                    prev_gripper_state=prev_gripper,
                )
                if sim_ctx is None:
                    data.qpos[arm_idx] = HOME_QPOS
                    mj.mj_forward(model, data)

            if _stop_event.is_set():
                _logwarn("Stopped — returning home.")
                if fa is not None:
                    fa.reset_joints(duration=MOVE_DURATION)
                    fa.open_gripper()
                    prev_gripper = "open"
                if sim_ctx is not None:
                    _reset_sim_to_home(sim_ctx)
                continue

            _log("=== Placements done, knocking first domino ===")
            first_xy, first_yaw = domino_plan[0]
            next_xy = domino_plan[1][0] if n >= 2 else None
            knock_wps = plan_knock_first_domino(
                model, data, arm_idx, HOME_QPOS, first_xy, first_yaw, next_xy
            )
            prev_gripper = execute_waypoints(
                knock_wps, sim_ctx=sim_ctx, fa=fa,
                prev_gripper_state=prev_gripper,
            )
            _log("=== Chain reaction complete ===")

    finally:
        server.shutdown()
        if sim_ctx is not None:
            try:
                sim_ctx.viewer.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
