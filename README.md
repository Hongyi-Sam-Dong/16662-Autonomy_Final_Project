# Dominoes Pickup — Run Guide

Hi, I'm Sam. Food on me!!! For all RI !!!. Taipan!! Let's GO!!!

ROS1 (Noetic) package `dominoes` that plans and executes domino pickups on a
Franka Panda. A server node advertises `/get_next_joint_target`; a client node
calls it in a loop and drives the arm through `frankapy`.

- Server entry point: `teamsam/scripts/pickup` → `dominoes.pickup:main`
- Client entry point: `teamsam/scripts/pickup_client` → `dominoes.runner:main`
- Launch file: `teamsam/launch/pickup.launch`

The folder is named `teamsam/` but `package.xml` declares the ROS package name
`dominoes` — that's the name you use with `roslaunch` / `rosrun`.

---

## 1. Start the docker container

On the host:

```bash
cd /home/student/16662_RobotAutonomy
./run_docker.sh
```

This drops you into a bash shell inside the container at `/home/ros_ws`.
`run_docker.sh` uses `--rm`, so anything installed inside the container is
**lost on exit** (see §5 for a permanent fix).

For extra terminals inside the same running container, on the host run:

```bash
./terminal_docker.sh
```

---

## 2. One-time setup (per container session)

Inside the container:

```bash
# 2a. Install mujoco (used by pickup.py for planning).
pip3 install mujoco
python3 -c "import mujoco; print(mujoco.__version__)"   # sanity check

# 2b. Build the workspace so the `dominoes` Python package is on sys.path
#     via catkin_python_setup().
cd /home/ros_ws
catkin_make
source devel/setup.bash

# 2c. Verify ROS sees the package and Python can import it.
rospack find dominoes
python3 -c "import dominoes, dominoes.pickup; print(dominoes.__file__)"
```

If `rospack` can't find `dominoes`, re-source `devel/setup.bash` in that shell.
If the Python import fails, re-run `catkin_make`.

---

## 3. Start FrankaPy (required for the client)

The client (`pickup_client`) calls `FrankaArm()`, which needs the frankapy
control bridge running. In its own terminal inside the container:

```bash
bash /home/ros_ws/src/git_packages/frankapy/bash_scripts/start_control_pc.sh -i iam-<robot-name>
```

Replace `iam-<robot-name>` with your lab's control PC hostname. Leave this
running.

---

## 4. Run the pickup

Use two terminals inside the container. **Both** need
`source /home/ros_ws/devel/setup.bash` (usually auto-sourced by `~/.bashrc`).

**Terminal A — server:**

```bash
roslaunch dominoes pickup.launch
```

Keep this running. It brings up the `pickup` node and advertises
`/get_next_joint_target`.

**Terminal B — client:**

```bash
rosrun dominoes pickup_client
```04

The client resets the arm, opens the gripper, then repeatedly calls the
service. Each response gives 8 floats: `joints[0:7]` are arm joint targets,
`joints[7]` is a gripper width (`0.08` ≈ fully open, `0.0` = closed). An empty
response ends the sequence.

Stop either side with `Ctrl-C`.

---

## 5. Make mujoco permanent (optional04)

`run_docker.sh` uses `--rm`, so `pip3 install mujoco` must be re-run every
time. The `Dockerfile` already contains `RUN pip3 install mujoco`; just
rebuild the image once on the host:

```bash
cd /home/student/16662_RobotAutonomy
docker build -t frankapy_docker .
./run_docker.sh
```

After this, step 2a is no longer needed.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mujoco'` | `pip3 install mujoco` (§2a) |
| `ModuleNotFoundError: No module named 'dominoes'` | `cd /home/ros_ws && catkin_make && source devel/setup.bash` |
| `[rospack] Error: package 'dominoes' not found` | `source /home/ros_ws/devel/setup.bash` in that shell |
| `rosrun` says executable not found | `chmod +x teamsam/scripts/pickup teamsam/scripts/pickup_client`, then `catkin_make` |
| Client hangs on `waiting for service /get_next_joint_target` | Server (terminal A) isn't running or crashed |
| Client errors from `FrankaArm()` | FrankaPy control PC (§3) isn't running |
| Stale build state after edits | `cd /home/ros_ws && rm -rf build devel && catkin_make` |

---

## Repo layout

```
Team_Sam_16662-Autonomy_Final_Project/
├── 666/                   # scratch / experiments (joint_subscriber.py, test.py, ...)
└── teamsam/               # ROS package (package.xml name: "dominoes")
    ├── CMakeLists.txt
    ├── package.xml
    ├── setup.py           # catkin_python_setup → installs `dominoes` py pkg
    ├── srv/
    │   └── GetNextJointTarget.srv
    ├── launch/
    │   ├── pickup.launch
    │   └── run_all.launch
    ├── scripts/
    │   ├── pickup         # server entry point
    │   └── pickup_client  # client entry point
    ├── dominoes/          # Python package
    │   ├── __init__.py
    │   ├── pickup.py      # server impl (uses mujoco)
    │   └── runner.py      # client impl (uses frankapy)
    ├── config/
    └── franka_emika_panda/
```
