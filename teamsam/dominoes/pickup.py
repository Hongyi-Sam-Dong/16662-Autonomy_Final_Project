from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import rospkg
import rospy

from dominoes.srv import GetNextJointTarget, GetNextJointTargetResponse


WAYPOINTS = np.array(
    [
        [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8, 0.1],
    ],
    dtype=float,
)

BLOCK_ORDER = [
    "RBottomFar3",
    "RBottomFar2",
    "RBottomFar1",
    "RBottomClose3",
    "RBottomClose2",
    "RBottomClose1",
    "RMiddleFar3",
    "RMiddleFar2",
    "RMiddleFar1",
    "RMiddleClose3",
    "RMiddleClose2",
    "RMiddleClose1",
    "RTopFar3",
    "RTopFar2",
    "RTopFar1",
    "RTopClose3",
    "RTopClose2",
    "RTopClose1",
    "LBottomClose3",
    "LBottomClose2",
    "LBottomClose1",
    "LBottomFar3",
    "LBottomFar2",
    "LBottomFar1",
]


def add_free_block_to_model(tree, name, pos, density, size, rgba, free):
    worldbody = tree.getroot().find("worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": name, "pos": f"{pos[0]} {pos[1]} {pos[2]}"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "type": "box",
            "density": f"{density}",
            "size": f"{size[0]} {size[1]} {size[2]}",
            "rgba": f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
        },
    )
    if free:
        ET.SubElement(body, "freejoint")


def calculate_ik_6d(
    model,
    data,
    target_pos,
    target_direction,
    body_name="hand",
    max_iters=500,
    tol=1e-3,
    step_size=0.2,
):
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

        current_pos = data.xpos[body_id]
        err_pos = target_pos - current_pos

        xmat = data.xmat[body_id].reshape(3, 3)
        current_dir = xmat[:, 2]
        err_rot = np.cross(current_dir, target_dir)
        err = np.hstack((err_pos, err_rot))

        if np.linalg.norm(err) < tol:
            break

        mj.mj_jacBody(model, data, jacp, jacr, body_id)
        jacobian = np.vstack((jacp[:, dof_indices], jacr[:, dof_indices]))
        lambda_sq = 1e-4
        jacobian_pinv = jacobian.T @ np.linalg.inv(
            jacobian @ jacobian.T + lambda_sq * np.eye(6)
        )

        data.qpos[dof_indices] += step_size * (jacobian_pinv @ err)
        data.qpos[dof_indices] = np.clip(data.qpos[dof_indices], jnt_lo, jnt_hi)

    solved_qpos = data.qpos[dof_indices].copy()
    data.qpos[:] = qpos0
    mj.mj_forward(model, data)
    return solved_qpos


def pack_target(q_arm, gripper):
    return np.concatenate([q_arm, [gripper]]).astype(float).tolist()


def get_model_directory():
    try:
        share_dir = Path(rospkg.RosPack().get_path("dominoes"))
        model_dir = share_dir / "franka_emika_panda"
        if model_dir.exists():
            return model_dir
    except rospkg.ResourceNotFound:
        pass

    return Path(__file__).resolve().parents[2] / "franka_emika_panda"


def add_static_scene(tree, end_of_table):
    blocks = [
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

    for name, pos, size in blocks:
        add_free_block_to_model(
            tree=tree,
            name=name,
            pos=pos,
            density=20,
            size=size,
            rgba=[0.2, 0.2, 0.9, 1.0],
            free=False,
        )

    box_size = [0.025, 0.075, 0.015]
    box_rgba = [0.0, 0.9, 0.2, 1.0]

    left_shelf_y = 0.504 - 0.09 - 0.135 / 2.0
    left_shelf_top_z = 0.1375 + 0.005 + 0.005
    left_far_x = end_of_table - 0.135 - 0.15
    left_close_x = (end_of_table - 0.135 - 2 * 0.2525) + 0.15

    add_free_block_to_model(tree, "LBottomFar1", [left_far_x, left_shelf_y, left_shelf_top_z + 0.015], 20, box_size, box_rgba, True)
    add_free_block_to_model(tree, "LBottomFar2", [left_far_x, left_shelf_y, left_shelf_top_z + 0.045], 20, box_size, box_rgba, True)
    add_free_block_to_model(tree, "LBottomFar3", [left_far_x, left_shelf_y, left_shelf_top_z + 0.075], 20, box_size, box_rgba, True)
    add_free_block_to_model(tree, "LBottomClose1", [left_close_x, left_shelf_y, left_shelf_top_z + 0.015], 20, box_size, box_rgba, True)
    add_free_block_to_model(tree, "LBottomClose2", [left_close_x, left_shelf_y, left_shelf_top_z + 0.045], 20, box_size, box_rgba, True)
    add_free_block_to_model(tree, "LBottomClose3", [left_close_x, left_shelf_y, left_shelf_top_z + 0.075], 20, box_size, box_rgba, True)

    right_shelf_y = -0.504 + 0.09 + 0.135 / 2.0
    right_far_x = end_of_table - 0.135 - 0.15
    right_close_x = (end_of_table - 0.135 - 2 * 0.2525) + 0.15
    rshelf_bottom_top_z = 0.1375 + 0.005 + 0.005
    rshelf_middle_top_z = 0.1375 + 0.005 + 0.2 + 0.005
    rshelf_top_top_z = 0.1375 + 0.005 + 0.4 + 0.005

    for shelf_prefix, shelf_top_z in [
        ("RBottom", rshelf_bottom_top_z),
        ("RMiddle", rshelf_middle_top_z),
        ("RTop", rshelf_top_top_z),
    ]:
        add_free_block_to_model(tree, f"{shelf_prefix}Far1", [right_far_x, right_shelf_y, shelf_top_z + 0.015], 20, box_size, box_rgba, True)
        add_free_block_to_model(tree, f"{shelf_prefix}Far2", [right_far_x, right_shelf_y, shelf_top_z + 0.045], 20, box_size, box_rgba, True)
        add_free_block_to_model(tree, f"{shelf_prefix}Far3", [right_far_x, right_shelf_y, shelf_top_z + 0.075], 20, box_size, box_rgba, True)
        add_free_block_to_model(tree, f"{shelf_prefix}Close1", [right_close_x, right_shelf_y, shelf_top_z + 0.015], 20, box_size, box_rgba, True)
        add_free_block_to_model(tree, f"{shelf_prefix}Close2", [right_close_x, right_shelf_y, shelf_top_z + 0.045], 20, box_size, box_rgba, True)
        add_free_block_to_model(tree, f"{shelf_prefix}Close3", [right_close_x, right_shelf_y, shelf_top_z + 0.075], 20, box_size, box_rgba, True)


def build_planning_model():
    model_dir = get_model_directory()
    root_model_xml = model_dir / "panda_torque_table.xml"
    generated_model_xml = model_dir / "panda_torque_table_shelves.xml"

    end_of_table = 0.55 + 0.135 + 0.05
    model_tree = ET.parse(root_model_xml)
    add_static_scene(model_tree, end_of_table)
    model_tree.write(generated_model_xml, encoding="utf-8", xml_declaration=True)

    model = mj.MjModel.from_xml_path(str(generated_model_xml))
    data = mj.MjData(model)
    arm_idx = [0, 1, 2, 3, 4, 5, 6]

    data.qpos[arm_idx] = WAYPOINTS[0][:7]
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)
    return model, data, arm_idx


def plan_side_pick(model, data, arm_idx, home_qpos, block_name, block_id):
    left_side_dir = np.array([0.0, 1.0, 0.0])
    right_side_dir = np.array([0.0, -1.0, 0.0])

    data.qpos[arm_idx] = home_qpos
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)

    block_pos = data.xpos[block_id].copy()

    if block_name.startswith("L"):
        side_dir = left_side_dir
        pregrasp_xyz = block_pos + np.array([0.0, -0.20, 0.0])
        grasp_xyz = block_pos + np.array([0.0, -0.1, 0.0])
        lift_xyz = grasp_xyz + np.array([0.0, 0.0, 0.02])
        pullout_xyz = block_pos + np.array([0.0, -0.40, 0.0])
    else:
        side_dir = right_side_dir
        pregrasp_xyz = block_pos + np.array([0.0, 0.20, 0.0])
        grasp_xyz = block_pos + np.array([0.0, 0.1, 0.0])
        lift_xyz = grasp_xyz + np.array([0.0, 0.0, 0.02])
        pullout_xyz = block_pos + np.array([0.0, 0.40, 0.0])

    pregrasp_q = calculate_ik_6d(model, data, pregrasp_xyz, side_dir)
    grasp_q = calculate_ik_6d(model, data, grasp_xyz, side_dir)
    lift_q = calculate_ik_6d(model, data, lift_xyz, side_dir)
    pullout_q = calculate_ik_6d(model, data, pullout_xyz, side_dir)

    for qpos in (pregrasp_q, grasp_q, lift_q, pullout_q):
        qpos[6] -= np.pi / 2

    return pregrasp_q, grasp_q, lift_q, pullout_q


def build_joint_sequence():
    model, data, arm_idx = build_planning_model()
    home_qpos = WAYPOINTS[0][:7].copy()

    block_ids = {}
    for block_name in BLOCK_ORDER:
        block_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, block_name)
        if block_id < 0:
            raise ValueError(f"Body '{block_name}' not found in planning model.")
        block_ids[block_name] = block_id

    sequence = []
    for block_name in BLOCK_ORDER:
        pregrasp_q, grasp_q, lift_q, pullout_q = plan_side_pick(
            model,
            data,
            arm_idx,
            home_qpos,
            block_name,
            block_ids[block_name],
        )

        sequence.extend(
            [
                {
                    "label": f"{block_name}/home_open",
                    "joints": pack_target(home_qpos, 0.1),
                },
                {
                    "label": f"{block_name}/pregrasp_open",
                    "joints": pack_target(pregrasp_q, 0.1),
                },
                {
                    "label": f"{block_name}/grasp_open",
                    "joints": pack_target(grasp_q, 0.1),
                },
                {
                    "label": f"{block_name}/grasp_close",
                    "joints": pack_target(grasp_q, 0.0),
                },
                {
                    "label": f"{block_name}/lift_close",
                    "joints": pack_target(lift_q, 0.0),
                },
                {
                    "label": f"{block_name}/pullout_close",
                    "joints": pack_target(pullout_q, 0.0),
                },
                {
                    "label": f"{block_name}/home_close",
                    "joints": pack_target(home_qpos, 0.0),
                },
                {
                    "label": f"{block_name}/home_open_release",
                    "joints": pack_target(home_qpos, 0.1),
                },
            ]
        )

    return sequence


class PickupNode:
    def __init__(self):
        self.sequence = build_joint_sequence()
        self.cursor = 0
        self.service = rospy.Service(
            "get_next_joint_target",
            GetNextJointTarget,
            self.handle_get_next_joint_target,
        )
        rospy.loginfo("Precomputed %d joint targets.", len(self.sequence))

    def handle_get_next_joint_target(self, request):
        response = GetNextJointTargetResponse()
        if not request.next:
            response.joints = []
            return response

        if self.cursor >= len(self.sequence):
            response.joints = []
            return response

        item = self.sequence[self.cursor]
        response.joints = item["joints"]

        self.cursor += 1
        return response


def main():
    rospy.init_node("pickup")
    PickupNode()
    rospy.spin()


if __name__ == "__main__":
    main()
