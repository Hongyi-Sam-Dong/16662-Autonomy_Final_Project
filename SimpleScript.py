import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET
import time
import RobotUtil as rt
import sys

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml" 
MODEL_XML = "franka_emika_panda/panda_torque_table_shelves.xml" 

# 7-DoF Panda joint waypoints in radians (examples)
WAYPOINTS = np.array([
    [0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
    [0.0, 0.65,  0.0, -2.0,  0.0,  2.65,  0.8, 0.04],
    [0.0, 0.65,  0.0, -2.0,  0.0,  2.65,  0.8, 0.0125],
    [1.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.0125],
    [1.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
    [0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
], dtype=float)

SEGMENT_DURATION = 2.0
HOLD_DURATION = 1.0

# Controller gains (per-joint)
KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)

def calculate_ik_6d(model, data, target_pos, target_quat=None, target_direction=None,
                    body_name="hand", max_iters=500, tol=1e-3, step_size=0.2):
    if target_quat is None and target_direction is None:
        raise ValueError("Provide either target_quat or target_direction.")

    # Normalize direction if provided
    if target_direction is not None:
        target_dir = np.array(target_direction, dtype=float)
        target_dir /= np.linalg.norm(target_dir)

    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    dof_indices = [0, 1, 2, 3, 4, 5, 6]
    nv = model.nv

    # Read joint limits from the model for clamping
    jnt_lo = model.jnt_range[dof_indices, 0]
    jnt_hi = model.jnt_range[dof_indices, 1]

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    # Save initial state
    qpos0 = data.qpos.copy()

    for _ in range(max_iters):
        mj.mj_forward(model, data)

        # 1. Positional Error (3D)
        current_pos = data.xpos[body_id]
        err_pos = target_pos - current_pos

        # 2. Rotational Error
        if target_direction is not None:
            xmat = data.xmat[body_id].reshape(3, 3)
            current_dir = xmat[:, 2]             
            err_rot = np.cross(current_dir, target_dir)
        else:
            # Full orientation constraint via quaternion
            current_quat = data.xquat[body_id]
            err_rot = np.zeros(3)
            mj.mju_subQuat(err_rot, target_quat, current_quat)

        # Combine into a single 6D error vector [X, Y, Z, Rx, Ry, Rz]
        err = np.hstack((err_pos, err_rot))

        if np.linalg.norm(err) < tol:
            break

        # Get both position and rotation Jacobians
        mj.mj_jacBody(model, data, jacp, jacr, body_id)

        # Stack them into a 6x7 matrix
        J = np.vstack((jacp[:, dof_indices], jacr[:, dof_indices]))

        # Damped Least Squares (Pseudo-inverse) for 6D
        lambda_sq = 1e-4
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))

        # Calculate and apply the change in joint angles
        delta_q = J_pinv @ err
        data.qpos[dof_indices] += step_size * delta_q

        # Clamp to joint limits to prevent divergence
        data.qpos[dof_indices] = np.clip(data.qpos[dof_indices], jnt_lo, jnt_hi)
        
    solved_qpos = data.qpos[dof_indices].copy()
    data.qpos[:] = qpos0
    mj.mj_forward(model, data)
    
    return solved_qpos

def run_segment(q_start, q_goal, gripper_pos, n_steps, duration):
    t = 0.0
    for _ in range(n_steps):
        q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, duration)
        q  = data.qpos[arm_idx].copy()
        qd = data.qvel[arm_idx].copy()
        tau = KP * (q_des - q) + KD * (qd_des - qd)
        data.ctrl[arm_idx]     = tau + data.qfrc_bias[:7]
        data.ctrl[gripper_idx] = gripper_pos
        mj.mj_step(model, data)
        v.sync()
        t += dt

def pick_from_table(pregrasp_q, grasp_q, lift_q):
    run_segment(home_qpos,   pregrasp_q, 0.04, segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(pregrasp_q,  grasp_q,    0.04, segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(grasp_q,     grasp_q,    0.0,  hold_steps * 5,             HOLD_DURATION)
    run_segment(grasp_q,     lift_q,     0.0,  segment_steps + hold_steps, SEGMENT_DURATION)

def place_left_and_regrasp(lift_q):
    down_dir = np.array([0.0, 0.0, -1.0])
    side_dir = np.array([0.0,  1.0,  0.0])

    shelf_top_z  = 0.1375 + 0.005 + 0.005
    place_x      = EndofTable - 0.275 - 0.135 + 0.0225
    place_y      = 0.504 - 0.09 - 0.135 / 2.0 - 0.135 / 2.0 + 0.03
    place_hand_z = shelf_top_z + 0.05 + 0.08

    preplace_xyz = np.array([place_x, place_y, place_hand_z + 0.20])
    place_xyz    = np.array([place_x, place_y, place_hand_z])

    data.qpos[arm_idx] = lift_q
    preplace_q = calculate_ik_6d(model, data, preplace_xyz, target_direction=down_dir)
    place_q    = calculate_ik_6d(model, data, place_xyz,    target_direction=down_dir)

    run_segment(lift_q,     preplace_q, 0.0,  segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(preplace_q, place_q,    0.0,  segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(place_q,    place_q,    0.04, hold_steps * 5,             HOLD_DURATION)
    run_segment(place_q,    preplace_q, 0.04, segment_steps + hold_steps, SEGMENT_DURATION)

    block_center   = np.array([place_x, place_y, shelf_top_z + 0.02])
    preregrasp_xyz = block_center + np.array([0.0, -0.20, 0.0])
    regrasp_xyz    = block_center + np.array([0.0, -0.1,  0.0])
    relift_xyz     = block_center + np.array([0.0, -0.1,  0.30])

    data.qpos[arm_idx] = preplace_q
    preregrasp_q = calculate_ik_6d(model, data, preregrasp_xyz, target_direction=side_dir)
    regrasp_q    = calculate_ik_6d(model, data, regrasp_xyz,    target_direction=side_dir)
    relift_q     = calculate_ik_6d(model, data, relift_xyz,     target_direction=side_dir)
    preregrasp_q[6] = preregrasp_q[6] - np.pi/2
    regrasp_q[6]    = regrasp_q[6]    - np.pi/2
    relift_q[6]     = relift_q[6]     - np.pi/2

    run_segment(preplace_q,   preregrasp_q, 0.04,  segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(preregrasp_q, regrasp_q,    0.04,  segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(regrasp_q,    regrasp_q,    0.015, hold_steps * 5,             HOLD_DURATION)
    run_segment(regrasp_q,    relift_q,     0.015, segment_steps + hold_steps, SEGMENT_DURATION)

    return relift_q

def place_on_right_shelf(relift_q, shelf_name, z_offset=0.0):
    right_side_dir = np.array([0.0, -1.0, 0.0])

    run_segment(relift_q, home_qpos, 0.015, segment_steps + hold_steps, SEGMENT_DURATION)

    rsm      = next(b for b in BLOCKS if b[0] == shelf_name)
    rsm_pos  = rsm[1]
    rsm_size = rsm[2]

    right_shelf_x       = rsm_pos[0]
    right_shelf_top_z   = rsm_pos[2] + rsm_size[2]
    right_shelf_front_y = rsm_pos[1] + rsm_size[1]
    right_block_y = right_shelf_front_y - 0.02
    right_block_z = right_shelf_top_z + 0.02 + 0.005 + z_offset

    rplace_xyz     = np.array([right_shelf_x, right_block_y + 0.08, right_block_z])
    pre_rplace_xyz = np.array([right_shelf_x, right_block_y + 0.22, right_block_z])

    data.qpos[arm_idx] = home_qpos
    pre_rplace_q = calculate_ik_6d(model, data, pre_rplace_xyz, target_direction=right_side_dir)
    rplace_q     = calculate_ik_6d(model, data, rplace_xyz,     target_direction=right_side_dir)
    pre_rplace_q[6] = pre_rplace_q[6] - np.pi/2
    rplace_q[6]     = rplace_q[6]     - np.pi/2

    run_segment(home_qpos,    pre_rplace_q, 0.015, segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(pre_rplace_q, rplace_q,     0.015, segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(rplace_q,     rplace_q,     0.04,  hold_steps * 5,             HOLD_DURATION)
    run_segment(rplace_q,     pre_rplace_q, 0.04,  segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(pre_rplace_q, home_qpos,    0.04,  segment_steps + hold_steps, SEGMENT_DURATION)
    run_segment(home_qpos,    home_qpos,    0.04,  hold_steps * 3,             HOLD_DURATION)

if __name__ == "__main__":
    np.random.seed(13)
    EndofTable=0.55+0.135+0.05
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
        rt.add_free_block_to_model(tree=modelTree, name=BLOCKS[i][0], pos=BLOCKS[i][1], density= 20 , size=BLOCKS[i][2] , rgba=[0.2, 0.2, 0.9, 1],free=False)

    #Add free blocks to manipulate
    rt.add_free_block_to_model(tree=modelTree, name="Block",  pos=[EndofTable-0.145,  0.0,  0.05], density=20, size=[0.02, 0.02, 0.02], rgba=[0.0, 0.9, 0.2, 1], free=True)
    rt.add_free_block_to_model(tree=modelTree, name="Block2", pos=[EndofTable-0.145,  0.1,  0.05], density=20, size=[0.02, 0.02, 0.02], rgba=[0.9, 0.2, 0.2, 1], free=True)
    rt.add_free_block_to_model(tree=modelTree, name="Block3", pos=[EndofTable-0.145, -0.1,  0.05], density=20, size=[0.02, 0.02, 0.02], rgba=[0.9, 0.9, 0.0, 1], free=True)

    modelTree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)
    
    # ###### EXECUTE PLAN ######

    # #Load the model
    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)
    arm_idx = [0,1,2,3,4,5,6]
    gripper_idx = 7

    # Initialize arm at first waypoint
    data.qpos[arm_idx] = WAYPOINTS[0][arm_idx]
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)

    # Map step duration to steps
    dt = model.opt.timestep
    segment_steps = max(1, int(SEGMENT_DURATION / dt))
    hold_steps =int(HOLD_DURATION/dt)

    #Launch Viewer
    v = viewer.launch_passive(model, data)
    v.cam.distance=3.0 
    v.cam.azimuth += 90                     
    
    # Grip info
    block_id  = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "Block")
    block2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "Block2")
    block3_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "Block3")
    block_pos = data.xpos[block_id]
    block_pos2 = data.xpos[block2_id]
    block_pos3 = data.xpos[block3_id]
    hand_id   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")
    # print block positions
    print(f"Block 1 position: {block_pos}")
    print(f"Block 2 position: {block_pos2}")
    print(f"Block 3 position: {block_pos3}")

    # for first block
    down_dir = np.array([0.0, 0.0, -1.0])   
    mj.mj_forward(model, data)
    block_pos = data.xpos[block_id].copy()
    pregrasp_xyz = block_pos + np.array([0.0, 0.0, 0.13])
    grasp_xyz    = block_pos + np.array([0.0, 0.0, 0.08])
    lift_xyz     = block_pos + np.array([0.0, 0.0, 0.35])

    home_qpos = WAYPOINTS[0][:7].copy()
    data.qpos[arm_idx] = home_qpos

    pregrasp_q = calculate_ik_6d(model, data, pregrasp_xyz, target_direction=down_dir)
    grasp_q    = calculate_ik_6d(model, data, grasp_xyz,    target_direction=down_dir)
    lift_q     = calculate_ik_6d(model, data, lift_xyz,     target_direction=down_dir)

    # for second block
    mj.mj_forward(model, data)
    block_pos2 = data.xpos[block2_id].copy()
    pregrasp_xyz2 = block_pos2 + np.array([0.0, 0.0, 0.13])
    grasp_xyz2    = block_pos2 + np.array([0.0, 0.0, 0.08])
    lift_xyz2     = block_pos2 + np.array([0.0, 0.0, 0.35])

    pregrasp_q2 = calculate_ik_6d(model, data, pregrasp_xyz2, target_direction=down_dir)
    grasp_q2    = calculate_ik_6d(model, data, grasp_xyz2,    target_direction=down_dir)
    lift_q2     = calculate_ik_6d(model, data, lift_xyz2,     target_direction=down_dir)

    # for third block
    mj.mj_forward(model, data)
    block_pos3 = data.xpos[block3_id].copy()
    pregrasp_xyz3 = block_pos3 + np.array([0.0, 0.0, 0.13])
    grasp_xyz3    = block_pos3 + np.array([0.0, 0.0, 0.08])
    lift_xyz3     = block_pos3 + np.array([0.0, 0.0, 0.35])

    pregrasp_q3 = calculate_ik_6d(model, data, pregrasp_xyz3, target_direction=down_dir)
    grasp_q3    = calculate_ik_6d(model, data, grasp_xyz3,    target_direction=down_dir)
    lift_q3     = calculate_ik_6d(model, data, lift_xyz3,     target_direction=down_dir)

    try:
        data.qpos[arm_idx] = home_qpos
        data.qvel[arm_idx] = 0.0
        mj.mj_forward(model, data)

        print("start first block pick up")
        pick_from_table(pregrasp_q, grasp_q, lift_q)
        relift_q = place_left_and_regrasp(lift_q)
        place_on_right_shelf(relift_q, "RShelfMiddle")
        print("Task complete — Block1 placed on RShelfMiddle")

        print("start second block placement")
        pick_from_table(pregrasp_q2, grasp_q2, lift_q2)
        relift_q2 = place_left_and_regrasp(lift_q2)
        place_on_right_shelf(relift_q2, "RShelfBottom", z_offset=0.005)
        print("Task complete — Block2 placed on RShelfBottom")

        print("start third block placement")
        pick_from_table(pregrasp_q3, grasp_q3, lift_q3)
        relift_q3 = place_left_and_regrasp(lift_q3)
        place_on_right_shelf(relift_q3, "RShelfTop")
        print("Task complete — all three blocks placed on right shelf")

    finally:
        v.close()
