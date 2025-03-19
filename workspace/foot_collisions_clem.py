import time
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, points_viz, point_viz, robot_frame_viz, frame_viz,get_viewer
import placo
from placo_utils.tf import tf

# Whether to debug (Meshcat viewer)
debug = False
# How many rotations of the foot are used
n_directions = 1024

# Loading the robot
robot = placo.RobotWrapper("../models/sigmaban/")
viz = robot_viz(robot)

# Placing the left foot in world origin
robot.set_joint("left_knee", 0.1)
robot.set_joint("right_knee", 0.1)
robot.update_kinematics()
robot.set_T_world_frame("left_foot", np.eye(4))
robot.update_kinematics()

solver = placo.KinematicsSolver(robot)

T_world_right = robot.get_T_world_frame("right_foot")

# Creating the viewer
viz = robot_viz(robot)

# Trunk
T_world_trunk = robot.get_T_world_frame("trunk")
T_world_trunk[2, 3] = 0.35
trunk_task = solver.add_frame_task("trunk", T_world_trunk)
trunk_task.configure("trunk_task", "soft", 1e3, 1e3)

# Fix right foot on the floor
right_foot_task = solver.add_frame_task("right_foot", T_world_right)
right_foot_task.configure("right_foot", "soft", 1e3, 1e3)

#Add frame task for left foot
T_world_left = np.eye(4)
# T_world_left[1, 3] = 0.16
left_foot_task = solver.add_frame_task("left_foot", T_world_left)
left_foot_task.configure("left_foot", "soft", 1.0, 1.0)

# Regularization task
posture_regularization_task = solver.add_joints_task()
posture_regularization_task.set_joints({dof: 0.0 for dof in robot.joint_names()})
posture_regularization_task.configure("reg", "soft", 1e-5)

# Initializing robot position before enabling constraints
for _ in range(32):
    solver.solve(True)
    robot.update_kinematics()

solver.enable_joint_limits(True)
solver.enable_velocity_limits(True)

t = 0
dt = 0.01
solver.dt = dt
robot.update_kinematics()

def find_highest_distance(direction, max_distance=1, max_angle=np.deg2rad(45)):
    robot.reset()
    robot.update_kinematics()

    for distance in np.linspace(0, max_distance, 100):
        T_world_target = T_world_left

        # Position of the target
        T_world_target[0, 3] = direction[0] * distance
        T_world_target[1, 3] = direction[1] * distance

        # Orientation of the target
        T_world_target[:3, :3] = tf.rotation_matrix(direction[2]*distance, [0, 0, 1])[:3, :3]

        left_foot_task.T_world_frame = T_world_target

        for _ in range(32):
            solver.solve(True)
            robot.update_kinematics()

        if debug:
            robot_frame_viz(robot, "left_foot")
            viz.display(robot.state.q)

        collisions = robot.self_collisions(False)
        points_viz(
            "collisions",
            [c.get_contact(0) for c in collisions],
            radius=0.003,
            color=0xFF0000,
        )
        frame_viz("target", T_world_target, opacity=0.25)

        position_error = np.linalg.norm(left_foot_task.T_world_frame[:3, 3] - robot.get_T_world_frame("left_foot")[:3, 3])
        orientation_error = np.linalg.norm(left_foot_task.T_world_frame[:3, :3] - robot.get_T_world_frame("left_foot")[:3, :3])

        if position_error > 2e-3 or orientation_error > 2e-2:
            if debug:
                input(
                    f"Target unreachable for direction {direction}, press [ENTER] to continue"
                )
            return distance

        if collisions:
            if debug:
                input(
                    f"Collision detected for direction {direction}, press [ENTER] to continue"
                )
            return distance

        if debug:
            time.sleep(0.01)

    return max_distance

import tqdm

A = []
b = []
directions = placo.directions_3d(n_directions)
for direction in tqdm.tqdm(directions):
    dist = find_highest_distance(direction)
    A.append(direction)
    b.append(dist - 1e-2)

from polytope import Polytope

polytope = Polytope(A, b)
polytope.save("workspace.pkl")

polytope.show(show_points=True)
polytope.simplify()
polytope.show(show_points=True)
polytope.save("workspace.pkl")