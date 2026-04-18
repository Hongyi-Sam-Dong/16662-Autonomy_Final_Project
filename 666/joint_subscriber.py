#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from frankapy import FrankaArm


def on_target(msg, fa):
    if len(msg.position) != 7:
        rospy.logwarn("need 7 joint values, got %d", len(msg.position))
        return
    rospy.loginfo("moving to %s", list(msg.position))
    fa.goto_joints(list(msg.position))


if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()
    rospy.Subscriber(
        "/target_joints",
        JointState,
        on_target,
        callback_args=fa,
        queue_size=1,
    )
    rospy.loginfo("listening on /target_joints for 7-DoF joint goals")
    rospy.spin()
