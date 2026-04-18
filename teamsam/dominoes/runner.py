"""Client for the dominoes /get_next_joint_target service.

Calls the service in a loop; each response gives 8 floats:
joints[0:7] = arm joint targets, joints[7] = gripper width
(0.04 = open, 0.0 = closed). An empty response ends the sequence.
"""
import rospy
from frankapy import FrankaArm

from dominoes.srv import GetNextJointTarget, GetNextJointTargetRequest


SERVICE_NAME = "/get_next_joint_target"

# Gripper width clamp (metres). MIN is a minimum closed width so the fingers
# don't fully close (e.g. keeps a grip on thin objects); MAX is the physical
# max opening of the Franka gripper.
GRIPPER_MIN = 0.044
GRIPPER_MAX = 0.08
# Any requested width at/above this is treated as "fully open" and routed
# through fa.open_gripper(), which reliably releases a held object. Plain
# goto_gripper() can leave the fingers clamped after a grasp.
OPEN_THRESHOLD = GRIPPER_MAX - 1e-4
# Force (N) to apply when grasping.
GRASP_FORCE = 10.0

# Home joint pose, matches WAYPOINTS[0][:7] in the server's pickup.py. When the
# arm lands at home AND the requested gripper width is "open", we call
# fa.open_gripper() regardless of whether the upstream flag already routed it
# that way — belt-and-suspenders so a held block is always dropped at home.
HOME_QPOS = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]
HOME_TOL = 1e-2


def is_at_home(joints):
    return all(abs(a - b) < HOME_TOL for a, b in zip(joints, HOME_QPOS))


def main():
    fa = FrankaArm()

    rospy.loginfo("waiting for service %s ...", SERVICE_NAME)
    rospy.wait_for_service(SERVICE_NAME)
    get_next = rospy.ServiceProxy(SERVICE_NAME, GetNextJointTarget)
    rospy.loginfo("service ready")

    fa.reset_joints()
    fa.open_gripper()

    step = 0
    while not rospy.is_shutdown():
        try:
            resp = get_next(GetNextJointTargetRequest(next=True))
        except rospy.ServiceException as e:
            rospy.logerr("service call failed: %s", e)
            return

        if len(resp.joints) == 0:
            rospy.loginfo("sequence complete after %d steps", step)
            break

        if len(resp.joints) != 8:
            rospy.logwarn("expected 8 values, got %d; skipping", len(resp.joints))
            continue

        joint_goal = list(resp.joints[:7])
        raw_width = float(resp.joints[7])
        gripper_width = max(GRIPPER_MIN, min(GRIPPER_MAX, raw_width))
        if gripper_width != raw_width:
            rospy.logwarn("clamped gripper %.3f -> %.3f (limit %.2f)",
                          raw_width, gripper_width, GRIPPER_MAX)
        rospy.loginfo("step %d: joints=%s gripper=%.3f",
                      step, joint_goal, gripper_width)

        fa.goto_joints(joint_goal)
        if gripper_width >= OPEN_THRESHOLD:
            fa.open_gripper()
        else:
            fa.goto_gripper(gripper_width, grasp=True, force=GRASP_FORCE)

        if is_at_home(joint_goal) and gripper_width >= OPEN_THRESHOLD:
            rospy.loginfo("at home with open target -> force-releasing gripper")
            fa.open_gripper()

        step += 1
