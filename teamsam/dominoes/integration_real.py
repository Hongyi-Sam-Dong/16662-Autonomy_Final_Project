"""Real-robot domino stacking with web UI.

Self-contained ROS1 node that:
  1. Serves the domino-placement web UI on http://localhost:5000 (visualized.html).
  2. Receives N domino (x, y, yaw) targets from the UI.
  3. For each target: picks a block from the shelf (reusing pickup.py's verified
     plan_side_pick) and places it standing at the UI pose (on-the-fly IK).
  4. After the last placement, knocks the first domino to start the chain.

Execution goes directly to the arm via frankapy — no /get_next_joint_target
service, no MuJoCo viewer, no torque control. MuJoCo is used for IK only.
"""
import os
import threading
import json as _json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import mujoco as mj
import rospy
from frankapy import FrankaArm

from dominoes.pickup import (
    BLOCK_ORDER,
    WAYPOINTS,
    build_planning_model,
    plan_side_pick,
    calculate_ik_6d,
)


HOME_QPOS = WAYPOINTS[0][:7].copy()

GRIPPER_OPEN   = 0.08
GRIPPER_CLOSED = 0.0
OPEN_THRESHOLD = GRIPPER_OPEN - 1e-4

MOVE_DURATION       = 3.0
CLAMP_MOVE_DURATION = 0.1
POST_CLAMP_WAIT     = 0.5
HOME_TOL            = 1e-2

PLACED_GRASP_Z = 0.235
JOINT7_LIMIT   = 2.8973

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HTML_PATH  = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'config', 'visualized.html'))

_ui_positions = []
_start_event  = threading.Event()
_stop_event   = threading.Event()


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


def _is_at_home(joints):
    return all(abs(a - b) < HOME_TOL for a, b in zip(joints, HOME_QPOS))


def _ik_chain(model, data, arm_idx, seed_q, targets):
    """Solve IK for a sequence of (xyz, direction) targets, progressively
    seeding each solve from the previous solution. Returns list of q vectors."""
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
    """Reuse pickup.py's proven pregrasp/grasp/lift/pullout plan for one block
    and wrap it as (joints, gripper, duration) waypoints for the executor."""
    pregrasp_q, grasp_q, lift_q, pullout_q = plan_side_pick(
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
    """Port of integration_simulation.py:213-252 minus the torque-control loop.
    Gripper stays closed throughout; no hold waypoint (frankapy goto_joints is
    blocking, so the arm naturally settles at strike_q before retracting)."""
    down_dir = np.array([0.0, 0.0, -1.0])
    push_vec = np.array([-np.cos(first_yaw), np.sin(first_yaw), 0.0])
    if next_xy is not None:
        delta = np.asarray(next_xy, dtype=float) - np.asarray(first_xy, dtype=float)
        if push_vec[0] * delta[0] + push_vec[1] * delta[1] < 0:
            push_vec = -push_vec

    transit_xyz  = np.array([first_xy[0], first_xy[1], 0.45])
    preplace_xyz = np.array([first_xy[0], first_xy[1], PLACED_GRASP_Z + 0.10])
    prep_xyz     = np.array([first_xy[0] - push_vec[0] * 0.02,
                             first_xy[1] - push_vec[1] * 0.02,
                             PLACED_GRASP_Z])
    if next_xy is not None:
        strike_xyz = np.array([(first_xy[0] + next_xy[0]) / 2.0,
                               (first_xy[1] + next_xy[1]) / 2.0,
                               PLACED_GRASP_Z])
    else:
        strike_xyz = np.array([first_xy[0] + push_vec[0] * 0.04,
                               first_xy[1] + push_vec[1] * 0.04,
                               PLACED_GRASP_Z + 0.01])

    transit_q, preplace_q, prep_q, strike_q = _ik_chain(
        model, data, arm_idx, current_q,
        [(transit_xyz, down_dir), (preplace_xyz, down_dir),
         (prep_xyz, down_dir), (strike_xyz, down_dir)],
    )

    return [
        (transit_q,  GRIPPER_CLOSED, MOVE_DURATION),
        (preplace_q, GRIPPER_CLOSED, MOVE_DURATION),
        (prep_q,     GRIPPER_CLOSED, MOVE_DURATION),
        (strike_q,   GRIPPER_CLOSED, MOVE_DURATION),
        (preplace_q, GRIPPER_CLOSED, MOVE_DURATION),
        (HOME_QPOS,  GRIPPER_CLOSED, MOVE_DURATION),
    ]


def execute_waypoints(fa, waypoints, prev_gripper_state):
    """Drive the real arm through (joints, gripper_width, duration) waypoints.
    Gripper state machine matches runner.py:91-107."""
    for joints, gripper_width, duration in waypoints:
        if rospy.is_shutdown() or _stop_event.is_set():
            return prev_gripper_state

        joint_goal = list(joints)
        fa.goto_joints(joint_goal, duration=duration)

        desired = "open" if gripper_width >= OPEN_THRESHOLD else "closed"
        if desired != prev_gripper_state:
            if desired == "open":
                fa.open_gripper()
            else:
                fa.close_gripper(grasp=True)
                rospy.sleep(POST_CLAMP_WAIT)
            prev_gripper_state = desired

        if _is_at_home(joint_goal) and gripper_width >= OPEN_THRESHOLD:
            fa.open_gripper()
            prev_gripper_state = "open"

    return prev_gripper_state


def main():
    rospy.init_node("integration_real", anonymous=False)

    model, data, arm_idx = build_planning_model()

    block_ids = {}
    for name in BLOCK_ORDER:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in planning model.")
        block_ids[name] = bid

    server = HTTPServer(('localhost', 5000), _UIHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rospy.loginfo("UI serving at http://localhost:5000 (source: %s)", _HTML_PATH)
    try:
        webbrowser.open('http://localhost:5000')
    except Exception:
        pass

    fa = FrankaArm()
    fa.reset_joints(duration=MOVE_DURATION)
    fa.open_gripper()
    prev_gripper = "open"

    while not rospy.is_shutdown():
        rospy.loginfo("Waiting for positions from UI...")
        while not _start_event.wait(timeout=0.5):
            if rospy.is_shutdown():
                server.shutdown()
                return
        _start_event.clear()
        _stop_event.clear()

        domino_plan = [(np.array([p['x'], p['y']]), p['yaw']) for p in _ui_positions]
        if not domino_plan:
            continue

        n = min(len(domino_plan), len(BLOCK_ORDER))
        if len(domino_plan) > len(BLOCK_ORDER):
            rospy.logwarn("UI sent %d dominoes; truncating to %d (shelf capacity).",
                          len(domino_plan), len(BLOCK_ORDER))

        rospy.loginfo("=== Starting %d domino(es) ===", n)

        for i in range(n):
            if _stop_event.is_set():
                break
            block_name = BLOCK_ORDER[i]
            target_xy, target_yaw = domino_plan[i]

            rospy.loginfo("[%d/%d] Pick %s from shelf", i + 1, n, block_name)
            pickup_wps = plan_pickup_waypoints(model, data, arm_idx,
                                               block_name, block_ids[block_name])
            prev_gripper = execute_waypoints(fa, pickup_wps, prev_gripper)
            data.qpos[arm_idx] = HOME_QPOS
            mj.mj_forward(model, data)

            if _stop_event.is_set():
                break

            rospy.loginfo("        Place at xy=(%.3f, %.3f) yaw=%.3f",
                          target_xy[0], target_xy[1], target_yaw)
            place_wps = plan_place_domino_standing(
                model, data, arm_idx, HOME_QPOS, target_xy, target_yaw
            )
            prev_gripper = execute_waypoints(fa, place_wps, prev_gripper)
            data.qpos[arm_idx] = HOME_QPOS
            mj.mj_forward(model, data)

        if _stop_event.is_set():
            rospy.logwarn("Stopped — returning home.")
            fa.reset_joints(duration=MOVE_DURATION)
            fa.open_gripper()
            prev_gripper = "open"
            continue

        rospy.loginfo("=== Placements done, knocking first domino ===")
        first_xy, first_yaw = domino_plan[0]
        next_xy = domino_plan[1][0] if n >= 2 else None
        knock_wps = plan_knock_first_domino(
            model, data, arm_idx, HOME_QPOS, first_xy, first_yaw, next_xy
        )
        prev_gripper = execute_waypoints(fa, knock_wps, prev_gripper)
        rospy.loginfo("=== Chain reaction complete ===")

    server.shutdown()


if __name__ == "__main__":
    main()
