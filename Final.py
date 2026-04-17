import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET
import time
import RobotUtil as rt
import sys
import threading
import json as _json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
MODEL_XML      = "franka_emika_panda/panda_torque_table_final.xml"

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)

SEGMENT_DURATION = 2.0
HOLD_DURATION    = 1.0
GRIPPER_OPEN     = 0.04
GRIPPER_CLOSED   = 0.015

HOME_QPOS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
EndofTable = 0.55 + 0.135 + 0.05

_ui_positions = []
_start_event  = threading.Event()
_stop_event   = threading.Event()


class _UIHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args): pass

    def do_GET(self):
        if self.path == '/':
            html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualized.html')
            with open(html_path, 'rb') as f:
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


def calculate_ik_6d(model, data, target_pos, target_direction=None, target_quat=None,
                    body_name="hand", max_iters=500, tol=1e-3, step_size=0.2, seed_q=None):
    if target_direction is not None:
        target_dir = np.array(target_direction, dtype=float)
        target_dir /= np.linalg.norm(target_dir)

    body_id     = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    dof_indices = [0, 1, 2, 3, 4, 5, 6]
    nv          = model.nv
    jnt_lo      = model.jnt_range[dof_indices, 0]
    jnt_hi      = model.jnt_range[dof_indices, 1]

    jacp  = np.zeros((3, nv))
    jacr  = np.zeros((3, nv))
    qpos0 = data.qpos.copy()

    if seed_q is not None:
        data.qpos[dof_indices] = seed_q.copy()
    else:
        data.qpos[dof_indices] = HOME_QPOS.copy()

    for _ in range(max_iters):
        mj.mj_forward(model, data)
        err_pos = target_pos - data.xpos[body_id]

        if target_direction is not None:
            current_dir = data.xmat[body_id].reshape(3, 3)[:, 2]
            err_rot     = np.cross(current_dir, target_dir)
        else:
            current_quat = data.xquat[body_id]
            err_rot = np.zeros(3)
            mj.mju_subQuat(err_rot, target_quat, current_quat)

        err = np.hstack((err_pos, err_rot))
        if np.linalg.norm(err) < tol:
            break

        mj.mj_jacBody(model, data, jacp, jacr, body_id)
        J = np.vstack((jacp[:, dof_indices], jacr[:, dof_indices]))

        J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        data.qpos[dof_indices] += step_size * (J_pinv @ err)
        data.qpos[dof_indices]  = np.clip(data.qpos[dof_indices], jnt_lo, jnt_hi)

    solved = data.qpos[dof_indices].copy()
    data.qpos[:] = qpos0
    mj.mj_forward(model, data)
    return solved


def run_segment(q_start, q_goal, gripper_pos, n_steps, duration):
    t = 0.0
    for _ in range(n_steps):
        if _stop_event.is_set():
            return

        q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, duration)
        q   = data.qpos[arm_idx].copy()
        qd  = data.qvel[arm_idx].copy()
        tau = KP * (q_des - q) + KD * (qd_des - qd)
        data.ctrl[arm_idx]     = tau + data.qfrc_bias[:7]
        data.ctrl[gripper_idx] = gripper_pos
        mj.mj_step(model, data)
        v.sync()
        t += dt


def pick_from_shelf_and_hold(block_name):
    block_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, block_name)
    block_pos = data.xpos[block_id].copy()

    left_side_dir = np.array([0.0, 1.0, 0.0])
    right_side_dir = np.array([0.0, -1.0, 0.0])

    if block_name.startswith("L"):
        side_dir = left_side_dir
        pregrasp_xyz = block_pos + np.array([0.0, -0.20, 0.0])
        grasp_xyz    = block_pos + np.array([0.0, -0.15, 0.0])
        pullout_xyz  = block_pos + np.array([0.0, -0.40, 0.0])
    else:
        side_dir = right_side_dir
        pregrasp_xyz = block_pos + np.array([0.0, 0.20, 0.0])
        grasp_xyz    = block_pos + np.array([0.0, 0.15, 0.0])
        pullout_xyz  = block_pos + np.array([0.0, 0.40, 0.0])

    pregrasp_q = calculate_ik_6d(model, data, pregrasp_xyz, target_direction=side_dir, seed_q=HOME_QPOS)
    grasp_q    = calculate_ik_6d(model, data, grasp_xyz,    target_direction=side_dir, seed_q=pregrasp_q)
    pullout_q  = calculate_ik_6d(model, data, pullout_xyz,  target_direction=side_dir, seed_q=grasp_q)

    pregrasp_q[6] -= np.pi / 2
    grasp_q[6]    -= np.pi / 2
    pullout_q[6]  -= np.pi / 2

    run_segment(HOME_QPOS, pregrasp_q, GRIPPER_OPEN, segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(pregrasp_q, grasp_q,   GRIPPER_OPEN, segment_steps + hold_steps, SEGMENT_DURATION)

    run_segment(grasp_q, grasp_q, GRIPPER_CLOSED, hold_steps * 2, HOLD_DURATION * 2)

    run_segment(grasp_q, pullout_q, GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    run_segment(pullout_q, HOME_QPOS, GRIPPER_CLOSED, segment_steps + hold_steps, SEGMENT_DURATION)


def place_domino_standing(target_xy, target_yaw):
    down_dir = np.array([0.0, 0.0, -1.0])
    placed_grasp_z = 0.235

    transit_xyz  = np.array([target_xy[0], target_xy[1], 0.45])
    preplace_xyz = np.array([target_xy[0], target_xy[1], placed_grasp_z + 0.10])
    place_xyz    = np.array([target_xy[0], target_xy[1], placed_grasp_z])

    transit_q  = calculate_ik_6d(model, data, transit_xyz,  target_direction=down_dir, seed_q=HOME_QPOS)
    preplace_q = calculate_ik_6d(model, data, preplace_xyz, target_direction=down_dir, seed_q=transit_q)
    place_q    = calculate_ik_6d(model, data, place_xyz,    target_direction=down_dir, seed_q=preplace_q)

    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    JOINT7_LIMIT = 2.8973
    current_j7 = HOME_QPOS[6]
    raw_j7 = place_q[6] + target_yaw
    cand_a = normalize_angle(raw_j7)
    cand_b = normalize_angle(raw_j7 + np.pi)

    def score(c):
        if abs(c) > JOINT7_LIMIT: return float('inf')
        return abs(c - current_j7)

    final_joint7 = min([cand_a, cand_b], key=score)

    transit_q[6]  = final_joint7
    preplace_q[6] = final_joint7
    place_q[6]    = final_joint7

    run_segment(HOME_QPOS,  transit_q,  GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    run_segment(transit_q,  preplace_q, GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    run_segment(preplace_q, place_q,    GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)

    run_segment(place_q, place_q, GRIPPER_OPEN, hold_steps*2, HOLD_DURATION*2)

    run_segment(place_q, preplace_q, GRIPPER_OPEN, segment_steps, SEGMENT_DURATION)
    run_segment(preplace_q, HOME_QPOS, GRIPPER_OPEN, segment_steps, SEGMENT_DURATION)


def knock_first_domino_with_arm(first_xy, first_yaw, next_xy=None):
    down_dir = np.array([0.0, 0.0, -1.0])

    if next_xy is not None:
        delta = np.asarray(next_xy, dtype=float) - np.asarray(first_xy, dtype=float)
        n = np.linalg.norm(delta)
        push_vec = (np.array([delta[0] / n, delta[1] / n, 0.0]) if n > 1e-6
                    else np.array([-np.cos(first_yaw), np.sin(first_yaw), 0.0]))
    else:
        push_vec = np.array([-np.cos(first_yaw), np.sin(first_yaw), 0.0])

    placed_grasp_z = 0.235

    transit_xyz  = np.array([first_xy[0], first_xy[1], 0.45])
    preplace_xyz = np.array([first_xy[0], first_xy[1], placed_grasp_z + 0.10])
    prep_xyz     = np.array([first_xy[0] - push_vec[0] * 0.01,
                             first_xy[1] - push_vec[1] * 0.01,
                             placed_grasp_z])
    strike_xyz   = np.array([first_xy[0], first_xy[1], placed_grasp_z])

    current_q  = data.qpos[arm_idx].copy()
    transit_q  = calculate_ik_6d(model, data, transit_xyz,  target_direction=down_dir, seed_q=current_q)
    preplace_q = calculate_ik_6d(model, data, preplace_xyz, target_direction=down_dir, seed_q=transit_q)
    prep_q     = calculate_ik_6d(model, data, prep_xyz,     target_direction=down_dir, seed_q=preplace_q)
    strike_q   = calculate_ik_6d(model, data, strike_xyz,   target_direction=down_dir, seed_q=prep_q)

    print("Moving arm above first domino...")
    run_segment(current_q,  transit_q,  GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    run_segment(transit_q,  preplace_q, GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    run_segment(preplace_q, prep_q,     GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    print("Gentle nudge (1 cm)!")
    run_segment(prep_q,     strike_q,   GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    run_segment(strike_q,   strike_q,   GRIPPER_CLOSED, hold_steps,    HOLD_DURATION)
    run_segment(strike_q,   preplace_q, GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    run_segment(preplace_q, HOME_QPOS,  GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)


if __name__ == "__main__":
    np.random.seed(13)

    modelTree = ET.parse(ROOT_MODEL_XML)

    BLOCKS=[
        ["TablePlane",[EndofTable-0.275,0.,-0.005],[0.275, 0.504, 0.0051]],
        ["LShelfDistal",[EndofTable-0.09-0.0225, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["LShelfProximal",[EndofTable-0.55-0.0225, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["LShelfBack",[EndofTable-0.55-0.0225-0.09, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["LShelfMid",[EndofTable-0.32, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["LShelfArch",[EndofTable-0.275-0.135+0.0225, 0.504-0.045-0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
        ["LShelfBottom",[EndofTable-0.275-0.135+0.0225, 0.504-0.09-0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
        ["LShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSupp2",[EndofTable-0.32-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, 0.504-0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfDistal",[EndofTable-0.09-0.0225, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["RShelfProximal",[EndofTable-0.55-0.0225, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["RShelfBack",[EndofTable-0.55-0.0225-0.09, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["RShelfMid",[EndofTable-0.32, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["RShelfArch",[EndofTable-0.275-0.135+0.0225, -0.504+0.045+0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfBottom",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
        ["RShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfMiddle",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.2],[0.2525, 0.135/2., 0.005]],
        ["RShelfMiddleSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.2],[0.315, 0.0225, 0.0225]],
        ["RShelfTop",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.4],[0.2525, 0.135/2., 0.005]],
        ["RShelfTopSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.4],[0.315, 0.0225, 0.0225]],
    ]

    for i in range(len(BLOCKS)):
        rt.add_free_block_to_model(tree=modelTree, name=BLOCKS[i][0], pos=BLOCKS[i][1], density=20, size=BLOCKS[i][2], rgba=[0.2, 0.2, 0.9, 1], free=False)

    box_size = [0.025, 0.075, 0.015]
    box_rgba = [0.0, 0.9, 0.2, 1.0]

    left_shelf_y = 0.504 - 0.09 - 0.135 / 2.0
    left_shelf_top_z = 0.1375 + 0.005 + 0.005
    left_pile1_x = EndofTable - 0.135 - 0.15
    left_pile2_x = (EndofTable - 0.135 - 2 * 0.2525) + 0.15

    rt.add_free_block_to_model(tree=modelTree, name="LBottomFar1",   pos=[left_pile1_x, left_shelf_y, left_shelf_top_z + 0.015], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="LBottomFar2",   pos=[left_pile1_x, left_shelf_y, left_shelf_top_z + 0.045], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="LBottomFar3",   pos=[left_pile1_x, left_shelf_y, left_shelf_top_z + 0.075], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="LBottomClose1", pos=[left_pile2_x, left_shelf_y, left_shelf_top_z + 0.015], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="LBottomClose2", pos=[left_pile2_x, left_shelf_y, left_shelf_top_z + 0.045], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="LBottomClose3", pos=[left_pile2_x, left_shelf_y, left_shelf_top_z + 0.075], density=20, size=box_size, rgba=box_rgba, free=True)

    right_far_x = EndofTable - 0.135 - 0.15
    right_close_x = (EndofTable - 0.135 - 2 * 0.2525) + 0.15
    right_shelf_y = -0.504 + 0.09 + 0.135 / 2.0
    rshelf_bottom_top_z = 0.1375 + 0.005 + 0.005
    rshelf_middle_top_z = 0.1375 + 0.005 + 0.2 + 0.005
    rshelf_top_top_z = 0.1375 + 0.005 + 0.4 + 0.005

    rt.add_free_block_to_model(tree=modelTree, name="RBottomFar1",   pos=[right_far_x,   right_shelf_y, rshelf_bottom_top_z + 0.015], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RBottomFar2",   pos=[right_far_x,   right_shelf_y, rshelf_bottom_top_z + 0.045], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RBottomFar3",   pos=[right_far_x,   right_shelf_y, rshelf_bottom_top_z + 0.075], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RBottomClose1", pos=[right_close_x, right_shelf_y, rshelf_bottom_top_z + 0.015], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RBottomClose2", pos=[right_close_x, right_shelf_y, rshelf_bottom_top_z + 0.045], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RBottomClose3", pos=[right_close_x, right_shelf_y, rshelf_bottom_top_z + 0.075], density=20, size=box_size, rgba=box_rgba, free=True)

    rt.add_free_block_to_model(tree=modelTree, name="RMiddleFar1",   pos=[right_far_x,   right_shelf_y, rshelf_middle_top_z + 0.015], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RMiddleFar2",   pos=[right_far_x,   right_shelf_y, rshelf_middle_top_z + 0.045], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RMiddleFar3",   pos=[right_far_x,   right_shelf_y, rshelf_middle_top_z + 0.075], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RMiddleClose1", pos=[right_close_x, right_shelf_y, rshelf_middle_top_z + 0.015], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RMiddleClose2", pos=[right_close_x, right_shelf_y, rshelf_middle_top_z + 0.045], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RMiddleClose3", pos=[right_close_x, right_shelf_y, rshelf_middle_top_z + 0.075], density=20, size=box_size, rgba=box_rgba, free=True)

    rt.add_free_block_to_model(tree=modelTree, name="RTopFar1",      pos=[right_far_x,   right_shelf_y, rshelf_top_top_z + 0.015], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RTopFar2",      pos=[right_far_x,   right_shelf_y, rshelf_top_top_z + 0.045], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RTopFar3",      pos=[right_far_x,   right_shelf_y, rshelf_top_top_z + 0.075], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RTopClose1",    pos=[right_close_x, right_shelf_y, rshelf_top_top_z + 0.015], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RTopClose2",    pos=[right_close_x, right_shelf_y, rshelf_top_top_z + 0.045], density=20, size=box_size, rgba=box_rgba, free=True)
    rt.add_free_block_to_model(tree=modelTree, name="RTopClose3",    pos=[right_close_x, right_shelf_y, rshelf_top_top_z + 0.075], density=20, size=box_size, rgba=box_rgba, free=True)

    modelTree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)

    server = HTTPServer(('localhost', 5000), _UIHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Open http://localhost:5000 — arrange dominoes, then click 'Start Placing'")
    try: webbrowser.open('http://localhost:5000')
    except Exception: pass

    model = mj.MjModel.from_xml_path(MODEL_XML)
    data  = mj.MjData(model)

    arm_idx     = list(range(7))
    gripper_idx = 7
    dt            = model.opt.timestep
    segment_steps = max(1, int(SEGMENT_DURATION / dt))
    hold_steps    = max(1, int(HOLD_DURATION    / dt))

    data.qpos[arm_idx] = HOME_QPOS
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)

    v = viewer.launch_passive(model, data)
    v.cam.distance = 2.5
    v.cam.azimuth += 90
    v.sync()

    block_order = [
        "RBottomFar3",
        "RBottomClose3",
        "RMiddleFar3", "RMiddleFar2", "RMiddleFar1",
        "RMiddleClose3", "RMiddleClose2", "RMiddleClose1",
        "RTopFar3", "RTopFar2", "RTopFar1",
        "RTopClose3", "RTopClose2", "RTopClose1",
        "LBottomClose3",
        "LBottomFar3",
        "RBottomFar2", "RBottomFar1",
        "RBottomClose2", "RBottomClose1",
        "LBottomClose2", "LBottomClose1",
        "LBottomFar2", "LBottomFar1",
    ]

    try:
        while v.is_running():
            print("\nWaiting for positions from UI...")
            _start_event.wait()
            if not v.is_running(): break
            _start_event.clear()
            _stop_event.clear()

            domino_plan = [(np.array([p['x'], p['y']]), p['yaw']) for p in _ui_positions]
            n = len(domino_plan)
            if n == 0: continue

            print(f"\n=== Starting Domino Assembly ({n} dominoes) ===")

            for i, (target_xy, target_yaw) in enumerate(domino_plan):
                if _stop_event.is_set() or i >= len(block_order):
                    break

                block_name = block_order[i]
                print(f"[{i+1}/{n}] Picking {block_name} from shelf...")

                pick_from_shelf_and_hold(block_name)

                if _stop_event.is_set():
                    break

                print(f"        Placing at UI=({_ui_positions[i].get('ui_x', 0):.2f}, {_ui_positions[i].get('ui_y', 0):.2f})")
                place_domino_standing(target_xy, target_yaw)

            if _stop_event.is_set():
                print("\nStopped — returning home...")
                run_segment(data.qpos[arm_idx], HOME_QPOS, GRIPPER_OPEN, segment_steps, SEGMENT_DURATION)
            else:
                print("\n=== Assembly Complete! Knocking first domino... ===")
                first_pos, first_yaw = domino_plan[0]
                next_pos = domino_plan[1][0] if len(domino_plan) >= 2 else None
                knock_first_domino_with_arm(first_pos, first_yaw, next_xy=next_pos)
                print("=== Chain reaction complete ===")

    finally:
        v.close()
