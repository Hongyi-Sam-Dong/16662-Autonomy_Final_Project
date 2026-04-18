#!/usr/bin/env python3
"""Drive the Franka arm through the dominoes pickup sequence.

Calls the /get_next_joint_target service (advertised by
`roslaunch dominoes pickup.launch`) repeatedly; each response gives
8 floats: joints[0:7] are arm joint targets, joints[7] is gripper width
(0.04 = open, 0.0 = closed). An empty response means the sequence is done.
"""
import rospy
from frankapy import FrankaArm
from dominoes.srv import GetNextJointTarget, GetNextJointTargetRequest


SERVICE_NAME = "/get_next_joint_target"


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
            rospy.logwarn("expected 8 values (7 joints + gripper), got %d; skipping",
                          len(resp.joints))
            continue

        joint_goal = list(resp.joints[:7])
        gripper_width = float(resp.joints[7])

        rospy.loginfo("step %d: joints=%s gripper=%.3f",
                      step, joint_goal, gripper_width)

        fa.goto_joints(joint_goal)
        fa.goto_gripper(gripper_width)

        step += 1


if __name__ == "__main__":
    main()
