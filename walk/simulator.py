import os
import time
from typing import Optional

import meshcat.transformations as tf
import mujoco
import mujoco.viewer
import numpy as np
from bam.mujoco import load_config, MujocoController
from bam.model import load_model

class Simulator:
    def __init__(self, model_dir: Optional[str] = None, 
                 use_bam: Optional[bool] = False, 
                 config: Optional[str] = None,
                 kp: Optional[float] = 32.0):
        # If model_dir is not provided, use the current directory
        if model_dir is None:
            if use_bam:
                model_dir = "../models/sigmaban_2025/mujoco_bam"
            else:
                model_dir = "../models/sigmaban_2025/mujoco"
        self.model_dir = model_dir

        # Load the model and data
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(f"{model_dir}/scene.xml")
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        # Retrieve the degrees of freedom id/name pairs
        joints = len(self.model.jnt_pos)
        self.dofs = [[k, self.model.jnt(k).name] for k in range(1, joints)]
        self.dofs_to_index = {dof: k for k, dof in self.dofs}

        self.viewer = None
        self.t: float = 0.0
        self.dt: float = self.model.opt.timestep
        self.frame: int = 0
        self.data.ctrl[:] = 0

        # BAM parameters
        self.use_bam: bool = use_bam
        self.dofs_to_model: dict = {}
        self.dofs_to_ctrl_index: dict = {}
        self.controllers: dict = {}

        # Loading BAM configuration
        if use_bam and config is not None:
            controllers, dofs_to_model, dofs_to_ctrl_id = load_config(config, kp, self.model, self.data)
            self.controllers = controllers
            self.dofs_to_model = dofs_to_model
            self.dofs_to_ctrl_index = dofs_to_ctrl_id

    def add_actuator_model(self, name: str, model_path: str, dofs: list, kp: float = 32) -> None:
        for i, dof in enumerate(dofs):
            self.dofs_to_model[dof] = name
            self.dofs_to_ctrl_index[dof] = i

        act_model = load_model(model_path)
        act_model.actuator.kp = kp
        self.controllers[name] = {"controller": MujocoController(act_model, dofs, self.model, self.data), 
                                  "dofs_state": [self.get_q(dof) for dof in dofs]}
        
    def set_floor_friction(self, friction: float) -> None:
        self.model.geom("floor").friction[0] = friction
        self.model.geom("floor").priority = 1

    def self_collisions(self) -> float:
        forcetorque = np.zeros(6)
        contacts = self.data.contact
        selector = (contacts.geom[:, 0] != 0) * (contacts.geom[:, 1] != 0)
        forces = 0.0
        for id in np.argwhere(selector):
            mujoco.mj_contactForce(self.model, self.data, id, forcetorque)
            forces += np.linalg.norm(forcetorque[:3])

        return forces

    def centroidal_force(self) -> float:
        return np.linalg.norm(self.data.qfrc_constraint[3:])

    def dof_names(self) -> list:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

    def reset_render(self) -> None:
        self.t = 0
        self.frame = 0
        self.viewer_start = time.time()

    def get_range(self, name: str) -> np.ndarray:
        return self.model.joint(name).range

    def get_q(self, name: str) -> float:
        """
        Gets the position of a given joint.

        Args:
            name (str): joint name

        Returns:
            float: joint position
        """
        addr = self.model.jnt_qposadr[self.dofs_to_index[name]]
        return self.data.qpos[addr]

    def get_qdot(self, name: str) -> float:
        """
        Gets the velocity of a given joint.

        Args:
            name (str): joint name

        Returns:
            float: joint velocity
        """
        addr = self.model.jnt_dofadr[self.dofs_to_index[name]]
        return self.data.qvel[addr]

    def set_q(self, name: str, value: float) -> None:
        """
        Sets a value of a given joint.

        Args:
            name (str): joint name
            value (float): target value
        """
        addr = self.model.jnt_qposadr[self.dofs_to_index[name]]
        self.data.qpos[addr] = value

    def get_control(self, name: str) -> None:
        """
        Gets the control for a given actuator

        Args:
            name (str): actuator name
        """
        actuator_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return self.data.ctrl[actuator_idx]

    def get_actuator_index(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def set_control(self, name: str, value: float, reset: bool = False) -> None:
        """
        Sets the control for a given actuator.
        If the actuator is a position actuator, the value is the desired position.
        If the actuator is a motor actuator, the value is the desired torque.

        Args:
            name (str): actuator name
            value (float): target value
        """
        if self.use_bam:
            self.controllers[self.dofs_to_model[name]]["dofs_state"][self.dofs_to_ctrl_index[name]] = value

        else:
            actuator_idx = self.get_actuator_index(name)
            self.data.ctrl[actuator_idx] = value

        if reset:
            self.set_q(name, value)

    def get_gyro(self) -> np.ndarray:
        """
        Gets the gyroscope data.

        Returns:
            np.ndarray: gyroscope data
        """
        return self.data.sensor("gyro").data

    def get_T_world_body(self, body_name: str) -> np.ndarray:
        """
        Gets the transformation from world to body frame.

        Args:
            body_name (str): body name
        """
        T = np.eye(4)
        body = self.data.body(body_name)
        T[:3, :3] = body.xmat.reshape(3, 3)
        T[:3, 3] = body.xpos
        return T

    def get_T_world_site(self, site_name: str) -> np.ndarray:
        """
        Gets the transformation from world to site frame.

        Args:
            site_name (str): site name
        """
        T = np.eye(4)
        site = self.data.site(site_name)
        T[:3, :3] = site.xmat.reshape(3, 3)
        T[:3, 3] = site.xpos
        return T

    def get_T_world_fbase(self) -> np.ndarray:
        """
        Gets the transformation from world to floating base frame.
        """
        data = self.data.joint("torso_2023_freejoint").qpos
        quat = data[3:]
        pos = data[:3]

        T = tf.quaternion_matrix(quat)
        T[:3, 3] = pos
        return T

    def set_T_world_fbase(self, T: np.ndarray) -> None:
        """
        Updates the floating base so that a body transformation match the target one

        Args:
            T (np.ndarray): target transformation
        """
        joint = self.data.joint("torso_2023_freejoint")

        quat = tf.quaternion_from_matrix(T)
        pos = T[:3, 3]

        joint.qpos[:] = [*pos, *quat]
        self.reset_velocity()

    def reset_velocity(self) -> None:
        """
        Resets the velocity of all the joints
        """
        self.data.qvel[:] = 0

    def set_T_world_body(self, body_name: str, T_world_bodyTarget: np.ndarray) -> None:
        """
        Updates the floating base so that a body transformation match the target one

        Args:
            body_name (str): body name
        """
        T_world_fbase = self.get_T_world_fbase()
        T_world_body = self.get_T_world_body(body_name)
        T_body_fbase = np.linalg.inv(T_world_body) @ T_world_fbase

        self.set_T_world_fbase(T_world_bodyTarget @ T_body_fbase)

    def set_T_world_site(self, site_name: str, T_world_siteTarget: np.ndarray) -> None:
        """
        Updates the floating base so that a site transformation match the target one

        Args:
            site_name (str): site name
        """
        T_world_fbase = self.get_T_world_fbase()
        T_world_site = self.get_T_world_site(site_name)
        T_site_fbase = np.linalg.inv(T_world_site) @ T_world_fbase

        self.set_T_world_fbase(T_world_siteTarget @ T_site_fbase)

    def get_pressure_sensors(self) -> dict:
        left_pressures = [
            -self.data.sensor("left_foot_cleat_front_right").data[2],
            -self.data.sensor("left_foot_cleat_front_left").data[2],
            -self.data.sensor("left_foot_cleat_back_right").data[2],
            -self.data.sensor("left_foot_cleat_back_left").data[2],
        ]
        right_pressures = [
            -self.data.sensor("right_foot_cleat_front_right").data[2],
            -self.data.sensor("right_foot_cleat_front_left").data[2],
            -self.data.sensor("right_foot_cleat_back_right").data[2],
            -self.data.sensor("right_foot_cleat_back_left").data[2],
        ]

        return {"left": left_pressures, "right": right_pressures}

    def step(self) -> None:
        if self.use_bam:
            for ctrl in self.controllers.values():
                ctrl["controller"].update(ctrl["dofs_state"])

        self.t = self.frame * self.dt
        mujoco.mj_step(self.model, self.data)
        self.frame += 1



    def set_gravity(self, gravity: np.ndarray) -> None:
        """
        Sets the gravity vector.

        Args:
            gravity (np.ndarray): gravity vector
        """
        self.model.opt.gravity[:] = gravity

    def render(self, realtime: bool = True):
        """
        Renders the visualization of the simulation.

        Args:
            realtime (bool, optional): if True, render will sleep to ensure real time viewing. Defaults to True.
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.reset_render()

        if realtime:
            current_ts = self.viewer_start + self.frame * self.dt
            to_sleep = current_ts - time.time()
            if to_sleep > 0:
                time.sleep(to_sleep)

        self.viewer.sync()


if __name__ == "__main__":
    sim = Simulator()
    sim.step()
    sim.set_T_world_site("left_foot", np.eye(4))

    sim.step()
    start = time.time()
    while True:
        sim.render(True)
        sim.set_control("head_yaw", np.sin(sim.t))
        sim.set_control("head_pitch", np.sin(sim.t))
        sim.step()

        elapsed = time.time() - start
        frames = sim.frame
        print(f"Elapsed: {elapsed:.2f}, Frames: {frames}, FPS: {frames / elapsed:.2f}")