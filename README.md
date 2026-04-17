# Dominoes Pickup Node

This workspace contains a ROS2 service node that precomputes a list of robot joint targets for the domino pickup task and serves them one at a time on request.

The node does not run the MuJoCo viewer, physics stepping, or torque control loop. Instead, at startup it:

- builds the shelf and block scene from the MuJoCo model
- runs inverse kinematics to compute arm joint targets for each pickup stage
- stores the targets as joint arrays in memory
- waits for a ROS2 service request and returns the next target in the list

Each returned target is a full joint vector:

- `7` Panda arm joints
- `1` gripper value

## Service

Service name:

```bash
/get_next_joint_target
```

Service type:

```bash
dominoes/srv/GetNextJointTarget
```

Service definition:

```srv
bool next
---
float64[] joints
```

Behavior:

- send `next: true` to get the next precomputed joint target
- send `next: false` to get an empty list and leave the cursor unchanged
- when the sequence is exhausted, the node returns an empty `joints` list

Note:

- there is currently no reset service
- to restart the sequence from the beginning, restart the node

## Build

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select dominoes --cmake-clean-cache
```

The workspace may print noisy package-discovery warnings from `mujoco_env`. The `dominoes` package build should still succeed.

## Run

In terminal 1:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run dominoes pickup
```

If the node starts correctly, it will precompute the sequence and then wait for service calls.

## Test The Service

In terminal 2:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Check that the service exists:

```bash
ros2 service list | grep get_next_joint_target
ros2 service type /get_next_joint_target
```

Request the next joint target:

```bash
ros2 service call /get_next_joint_target dominoes/srv/GetNextJointTarget "{next: true}"
```

Call it again to advance to the next target:

```bash
ros2 service call /get_next_joint_target dominoes/srv/GetNextJointTarget "{next: true}"
```

Test the no-op branch:

```bash
ros2 service call /get_next_joint_target dominoes/srv/GetNextJointTarget "{next: false}"
```

Expected result:

- `next: true` returns a `joints` array
- `next: false` returns `joints: []`
- after the last target, the node returns `joints: []`

## What The Joint Sequence Contains

For each block in the configured block order, the node precomputes waypoint-style targets such as:

- home, gripper open
- pregrasp, gripper open
- grasp, gripper open
- grasp, gripper closed
- lift, gripper closed
- pullout, gripper closed
- home, gripper closed
- home, gripper open

These are discrete targets, not a time-parameterized trajectory. The client is responsible for deciding how to execute or interpolate between them.
