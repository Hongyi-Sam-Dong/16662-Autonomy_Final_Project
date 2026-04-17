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
MODEL_XML      = "franka_emika_panda/panda_torque_table_dominoes.xml"

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)

SEGMENT_DURATION = 2.0
HOLD_DURATION    = 1.0
GRIPPER_OPEN     = 0.06
GRIPPER_CLOSED   = 0.024

EndofTable = 0.55 + 0.135 + 0.05

DOMINO_HALF_SIZE = [0.025, 0.075, 0.015]

HOME_QPOS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])

_carry = {"idx": None, "local_pos": None, "local_quat": None}

HIDDEN_POS = [2.0, 2.0, 0.0]

_ui_positions = []
_start_event  = threading.Event()
_stop_event   = threading.Event()

_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualized.html')


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


def set_block_pose(body_name, pos, quat=[1, 0, 0, 0]):
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    jnt_adr = model.body_jntadr[body_id]
    q_adr   = model.jnt_qposadr[jnt_adr]
    v_adr   = model.jnt_dofadr[jnt_adr]
    data.qpos[q_adr:q_adr+3] = pos
    data.qpos[q_adr+3:q_adr+7] = quat
    data.qvel[v_adr:v_adr+6]   = 0.0


def run_segment(q_start, q_goal, gripper_pos, n_steps, duration):
    t = 0.0
    _hand_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")
    for _ in range(n_steps):
        if _stop_event.is_set():
            return
        if _carry["idx"] is not None:
            h_pos  = data.xpos[_hand_id].copy()
            h_mat  = data.xmat[_hand_id].reshape(3, 3)
            h_quat = data.xquat[_hand_id].copy()
            world_pos = h_pos + h_mat @ _carry["local_pos"]
            world_quat = np.zeros(4)
            mj.mju_mulQuat(world_quat, h_quat, _carry["local_quat"])
            set_block_pose(f"Domino_{_carry['idx']}", world_pos, world_quat)

        q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, duration)
        q   = data.qpos[arm_idx].copy()
        qd  = data.qvel[arm_idx].copy()
        tau = KP * (q_des - q) + KD * (qd_des - qd)
        data.ctrl[arm_idx]     = tau + data.qfrc_bias[:7]
        data.ctrl[gripper_idx] = gripper_pos
        mj.mj_step(model, data)
        v.sync()
        t += dt


def spawn_and_grasp(block_idx):
    receive_qpos = HOME_QPOS.copy()

    run_segment(data.qpos[arm_idx], receive_qpos, GRIPPER_OPEN, segment_steps, SEGMENT_DURATION)
    run_segment(receive_qpos, receive_qpos, GRIPPER_OPEN, hold_steps, HOLD_DURATION)

    hand_id   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")
    hand_pos  = data.xpos[hand_id].copy()
    hand_mat  = data.xmat[hand_id].reshape(3, 3)
    hand_quat = data.xquat[hand_id].copy()

    block_center = hand_pos + hand_mat @ np.array([0.0, 0.0, 0.155])
    spawn_quat   = [0.5, 0.5, 0.5, 0.5]

    set_block_pose(f"Domino_{block_idx}", block_center, spawn_quat)
    mj.mj_forward(model, data)

    _carry["local_pos"]  = np.array([0.0, 0.0, 0.155])
    hand_quat_inv = np.array([hand_quat[0], -hand_quat[1], -hand_quat[2], -hand_quat[3]])
    local_quat = np.zeros(4)
    mj.mju_mulQuat(local_quat, hand_quat_inv, np.array(spawn_quat))
    _carry["local_quat"] = local_quat
    _carry["idx"]        = block_idx

    run_segment(receive_qpos, receive_qpos, GRIPPER_CLOSED, hold_steps*2, HOLD_DURATION*2)

    return receive_qpos


def place_domino_standing(lift_q, target_xy, target_yaw):
    down_dir = np.array([0.0, 0.0, -1.0])
    placed_grasp_z = 0.235

    # 定義路徑點
    transit_xyz  = np.array([target_xy[0], target_xy[1], 0.45])
    preplace_xyz = np.array([target_xy[0], target_xy[1], placed_grasp_z + 0.10]) # 稍微降低預放高度縮短時間
    place_xyz    = np.array([target_xy[0], target_xy[1], placed_grasp_z])

    # 1. 解算 IK
    transit_q  = calculate_ik_6d(model, data, transit_xyz,  target_direction=down_dir, seed_q=lift_q)
    preplace_q = calculate_ik_6d(model, data, preplace_xyz, target_direction=down_dir, seed_q=transit_q)
    place_q    = calculate_ik_6d(model, data, place_xyz,    target_direction=down_dir, seed_q=preplace_q)

    # 2. 角度正規化 + 方向選擇
    # Dominoes are symmetric mod π (3×5 cm footprint looks identical when flipped 180°).
    # So yaw and yaw+π are visually equivalent — pick whichever requires LESS joint7 motion
    # AND is within Panda's joint7 limit (±2.8973 rad ≈ ±166°).
    # This avoids huge sweeps (e.g. 170° → -190° long way) that knock over placed dominoes.
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    JOINT7_LIMIT = 2.8973
    current_j7 = lift_q[6]  # starting joint7 angle
    raw_j7 = place_q[6] + target_yaw
    cand_a = normalize_angle(raw_j7)
    cand_b = normalize_angle(raw_j7 + np.pi)   # 180°-flipped equivalent (same domino pose)

    def score(c):
        if abs(c) > JOINT7_LIMIT:
            return float('inf')                # reject: outside joint limit
        return abs(c - current_j7)             # prefer smallest motion

    final_joint7 = min([cand_a, cand_b], key=score)
    print(f"    joint7: current={np.degrees(current_j7):+.1f}° "
          f"candidates=[{np.degrees(cand_a):+.1f}°, {np.degrees(cand_b):+.1f}°] "
          f"→ chose {np.degrees(final_joint7):+.1f}° "
          f"(Δ={np.degrees(final_joint7 - current_j7):+.1f}°)")

    transit_q[6]  = final_joint7
    preplace_q[6] = final_joint7
    place_q[6]    = final_joint7

    # 3. 執行動作
    # 前往目標上方
    run_segment(lift_q,     transit_q,  GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    # 垂直下降
    run_segment(transit_q,  preplace_q, GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)
    run_segment(preplace_q, place_q,    GRIPPER_CLOSED, segment_steps, SEGMENT_DURATION)

    # 鬆開夾爪
    run_segment(place_q,    place_q,    GRIPPER_OPEN,   hold_steps*2,  HOLD_DURATION*2)
    _carry["idx"] = None
    
    # 垂直撤離 (避免掃倒剛放好的骨牌)
    run_segment(place_q,    preplace_q, GRIPPER_OPEN,   segment_steps, SEGMENT_DURATION)
    run_segment(preplace_q, HOME_QPOS,  GRIPPER_OPEN,   segment_steps, SEGMENT_DURATION)

def knock_first_domino_with_arm(first_xy, first_yaw, next_xy=None):
    """
    Tina, 這是使用機械臂實體推倒第一顆骨牌的邏輯：
    1. 閉合夾爪移到骨牌後方
    2. 向骨牌方向移動推倒它

    兩個關鍵修正：
    (a) push_vec 改用 domino[0]→domino[1] 的實際連線（最穩妥，不依賴 yaw 換算）。
        原本的 (-sin, cos) 推導出來的方向差 90°，所以手臂會沿骨牌長軸滑過去而打不到面。
    (b) prep/strike 的 z 不能是 0（那樣 hand 會嘗試觸到地板導致 IK 失敗或撞桌）。
        骨牌中心 z≈0.08、頂端 z≈0.155；hand 比指尖高 ~0.155m，所以 hand z=0.24
        對應指尖 z≈0.085（骨牌中段），剛好合適推倒。
    """
    down_dir = np.array([0.0, 0.0, -1.0])

    # 1. 計算推動方向
    if next_xy is not None:
        delta = np.asarray(next_xy, dtype=float) - np.asarray(first_xy, dtype=float)
        n = np.linalg.norm(delta)
        push_vec = (np.array([delta[0] / n, delta[1] / n, 0.0]) if n > 1e-6
                    else np.array([-np.cos(first_yaw), np.sin(first_yaw), 0.0]))
    else:
        # fallback: sim 座標下的骨牌 fall direction（UI→SIM 軸交換後推出來的）
        push_vec = np.array([-np.cos(first_yaw), np.sin(first_yaw), 0.0])

    print(f"[knock] push_vec = ({push_vec[0]:+.3f}, {push_vec[1]:+.3f})")

    # 2. 關鍵點：hand z=0.24 → 指尖 ~0.085（骨牌中段）
    strike_z = 0.24
    prep_xyz   = np.array([first_xy[0], first_xy[1], strike_z + 0.02]) - push_vec * 0.12
    strike_xyz = np.array([first_xy[0], first_xy[1], strike_z])        + push_vec * 0.118

    # 3. 解算 IK（不再強設 joint7，保留 IK 自己收斂的解即可——推一下而已，夾爪朝向無關緊要）
    current_q = data.qpos[arm_idx].copy()
    prep_q   = calculate_ik_6d(model, data, prep_xyz,   target_direction=down_dir, seed_q=current_q)
    strike_q = calculate_ik_6d(model, data, strike_xyz, target_direction=down_dir, seed_q=prep_q)

    # 4. 執行動作
    print("Moving arm to strike position...")
    run_segment(current_q, prep_q,   GRIPPER_CLOSED, segment_steps,     SEGMENT_DURATION)

    print("Striking!!!")
    run_segment(prep_q,    strike_q, GRIPPER_CLOSED, segment_steps // 3, SEGMENT_DURATION / 3)

    # 停在擊倒位置，讓 chain 有時間倒下（0.5 秒）
    run_segment(strike_q,  strike_q, GRIPPER_CLOSED, hold_steps,        HOLD_DURATION)

    # 5. 撤離並回到 Home
    run_segment(strike_q,  HOME_QPOS, GRIPPER_CLOSED, segment_steps,    SEGMENT_DURATION)


MAX_DOMINOES = 60


if __name__ == "__main__":
    server = HTTPServer(('localhost', 5000), _UIHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Open http://localhost:5000 — arrange dominoes, then click 'Start Placing'")
    try:
        webbrowser.open('http://localhost:5000')
    except Exception:
        pass

    np.random.seed(42)
    modelTree = ET.parse(ROOT_MODEL_XML)
    for i in range(MAX_DOMINOES):
        rt.add_free_block_to_model(
            tree=modelTree, name=f"Domino_{i}",
            pos=[HIDDEN_POS[0], HIDDEN_POS[1] + i*0.1, HIDDEN_POS[2]],
            density=20, size=DOMINO_HALF_SIZE, rgba=[0.1, 0.8, 0.8, 1], free=True
        )
    modelTree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)

    model = mj.MjModel.from_xml_path(MODEL_XML)
    data  = mj.MjData(model)

    arm_idx     = list(range(7))
    gripper_idx = 7
    dt            = model.opt.timestep
    segment_steps = max(1, int(SEGMENT_DURATION / dt))
    hold_steps    = max(1, int(HOLD_DURATION    / dt))

    data.qpos[arm_idx] = HOME_QPOS
    data.qvel[arm_idx] = 0.0
    for i in range(MAX_DOMINOES):
        set_block_pose(f"Domino_{i}", [HIDDEN_POS[0], HIDDEN_POS[1] + i*0.1, HIDDEN_POS[2]])
    mj.mj_forward(model, data)

    v = viewer.launch_passive(model, data)
    v.cam.distance = 2.5
    v.cam.azimuth += 90
    v.sync()

    try:
        while v.is_running():
            print("\nWaiting for positions from UI...")
            _start_event.wait()
            if not v.is_running():
                break
            _start_event.clear()
            _stop_event.clear()

            domino_plan = [(np.array([p['x'], p['y']]), p['yaw']) for p in _ui_positions]
            n = len(domino_plan)
            if n == 0:
                continue

            print(f"\n=== Domino plan ({n} dominoes) ===")
            print(f"  {'#':>3}  {'UI_x':>8}  {'UI_y':>8}  {'UI_yaw':>8}  |  "
                  f"{'SIM_x':>8}  {'SIM_y':>8}  {'SIM_yaw':>8}")
            for i, p in enumerate(_ui_positions):
                ui_x   = p.get('ui_x',   float('nan'))
                ui_y   = p.get('ui_y',   float('nan'))
                ui_yaw = p.get('ui_yaw', float('nan'))
                print(f"  {i+1:>3}  {ui_x:>8.2f}  {ui_y:>8.2f}  {np.degrees(ui_yaw):>7.1f}°  |  "
                      f"{p['x']:>8.4f}  {p['y']:>8.4f}  {np.degrees(p['yaw']):>7.1f}°")

            for i in range(MAX_DOMINOES):
                set_block_pose(f"Domino_{i}", [HIDDEN_POS[0], HIDDEN_POS[1] + i*0.1, HIDDEN_POS[2]])
            mj.mj_forward(model, data)

            print(f"\n=== Starting Domino Assembly ({n} dominoes) ===")
            for i, (target_xy, target_yaw) in enumerate(domino_plan):
                if _stop_event.is_set():
                    break
                ui_p = _ui_positions[i]
                print(f"Placing {i+1}/{n} | "
                      f"UI=({ui_p.get('ui_x', 0):.2f},{ui_p.get('ui_y', 0):.2f}) "
                      f"SIM=({target_xy[0]:.4f},{target_xy[1]:.4f}) "
                      f"yaw={np.degrees(target_yaw):.1f}°")
                receive_q = spawn_and_grasp(i)
                if _stop_event.is_set():
                    _carry["idx"] = None
                    break
                place_domino_standing(receive_q, target_xy, target_yaw)

            if _stop_event.is_set():
                _carry["idx"] = None
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
