# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
VBot flat terrain navigation environment.
Ported from AnymalC navigation (anymal_c_np.py) to VBot robot model.
Non-navigation parts aligned with walk_np.py (PD control, reward structure, etc.)
"""

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import VBotEnvCfg


@registry.env("vbot_navigation_flat", "np")
class VBotNavFlatEnv(NpEnv):
    """
    VBot flat terrain navigation environment.
    The robot navigates to a target position and heading on flat ground.
    Observation: 54 dims (walk_np 48 dims + 6 navigation dims)
    Action: 12 dims (PD controlled joint targets)
    """

    _cfg: VBotEnvCfg

    def __init__(self, cfg: VBotEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()

        # Navigation visualization bodies
        self._target_marker_body = self._model.get_body("target_marker")
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # Observation: linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12)
        #            + last_actions(12) + commands(3) + position_error(2) + heading_error(1)
        #            + distance(1) + reached_flag(1) + stop_ready_flag(1) = 54
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        # Find DOF indices for target marker and arrows
        self._find_target_marker_dof_indices()
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()

        self._init_buffer()

    # ===================== Spaces =====================

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # ===================== State accessors (walk_np style) =====================

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)

    # ===================== Initialization helpers =====================

    def _init_buffer(self):
        cfg = self._cfg

        self.kps = np.ones(self._num_action, dtype=np.float32) * cfg.control_config.stiffness
        self.kds = np.ones(self._num_action, dtype=np.float32) * cfg.control_config.damping
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32,
        )

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.hip_indices = []
        self.calf_indices = []
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
            if "hip" in self._model.actuator_names[i]:
                self.hip_indices.append(i)
            if "calf" in self._model.actuator_names[i]:
                self.calf_indices.append(i)

        self._init_dof_pos[-self._num_action :] = self.default_angles

    def _find_target_marker_dof_indices(self):
        """
        DOF layout:
          0-2:  target_marker (slide x, slide y, hinge yaw)
          3-5:  base position (x, y, z)
          6-9:  base quaternion (qx, qy, qz, qw)
          10+:  joint angles (12)
        """
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10

    def _find_arrow_dof_indices(self):
        """
        After joints (DOF 10-21):
          22-28: robot_heading_arrow freejoint (3 pos + 4 quat)
          29-35: desired_heading_arrow freejoint (3 pos + 4 quat)
        """
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36

        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start : self._robot_arrow_dof_end] = [
                0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0,
            ]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start : self._desired_arrow_dof_end] = [
                0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0,
            ]

    def _init_contact_geometry(self):
        cfg = self._cfg
        self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)
        self._init_termination_contact()
        self._init_foot_contact()
        # Aliases matching walk_np naming
        self.termination_check = self.termination_contact
        self.num_check = self.num_termination_check
        self.foot_check = self.foot_contact_check
        self.foot_check_num = self.num_foot_check

    def _init_termination_contact(self):
        cfg = self._cfg
        base_indices = []
        for base_name in cfg.asset.terminate_after_contacts_on:
            try:
                idx = self._model.get_geom_index(base_name)
                if idx is not None:
                    base_indices.append(idx)
            except Exception:
                pass

        if base_indices:
            self.termination_contact = np.array(
                [[idx, self.ground_index] for idx in base_indices], dtype=np.uint32
            )
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0

    def _init_foot_contact(self):
        cfg = self._cfg
        foot_indices = []
        for foot_name in cfg.asset.foot_names:
            try:
                idx = self._model.get_geom_index(foot_name)
                if idx is not None:
                    foot_indices.append(idx)
            except Exception:
                pass

        if foot_indices:
            self.foot_contact_check = np.array(
                [[idx, self.ground_index] for idx in foot_indices], dtype=np.uint32
            )
            self.num_foot_check = self.foot_contact_check.shape[0]
        else:
            self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
            self.num_foot_check = 0

    # ===================== Action (walk_np style PD control) =====================

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state

    def _compute_torques(self, actions, data):
        # PD controller (same as walk_np)
        actions_scaled = actions * self._cfg.control_config.action_scale
        torques = self.kps * (
            actions_scaled + self.default_angles - self.get_dof_pos(data)
        ) - self.kds * self.get_dof_vel(data)
        return torques

    # ===================== State update (walk_np structure) =====================

    def update_state(self, state: NpEnvState):
        state = self.update_observation(state)
        state = self.update_terminated(state)
        state = self.update_reward(state)
        state.info["steps"] = state.info.get("steps", np.zeros(self._num_envs, dtype=np.int32)) + 1
        return state

    # ===================== Navigation state =====================

    def _update_nav_state(self, data: mtx.SceneData, info: dict):
        """Compute navigation-specific state and store in info."""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]

        pose_commands = info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        # Position error and distance
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold

        # Heading error
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold

        reached_all = np.logical_and(reached_position, reached_heading)

        # Velocity commands derived from navigation errors
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)

        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_diff) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        gyro = self.get_gyro(data)
        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)

        # Store navigation state in info
        info["velocity_commands"] = velocity_commands
        info["reached_all"] = reached_all
        info["position_error"] = position_error
        info["distance_to_target"] = distance_to_target
        info["heading_diff"] = heading_diff
        info["stop_ready"] = stop_ready
        info["desired_vel_xy"] = desired_vel_xy

    # ===================== Observation (walk_np style + nav extensions) =====================

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """Build observation: walk_np base (48 dims) + navigation extensions (6 dims)."""
        linear_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        local_gravity = Quaternion.rotate_inverse(base_quat, self.gravity_vec)
        diff = self.get_dof_pos(data) - self.default_angles

        noisy_linvel = linear_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = diff * self._cfg.normalization.dof_pos
        noisy_joint_vel = self.get_dof_vel(data) * self._cfg.normalization.dof_vel
        command = info["velocity_commands"] * self.commands_scale
        last_actions = info["current_actions"]

        # Navigation extensions
        position_error_normalized = info["position_error"] / 5.0
        heading_error_normalized = info["heading_diff"] / np.pi
        distance_normalized = np.clip(info["distance_to_target"] / 5.0, 0, 1)
        reached_flag = info["reached_all"].astype(np.float32)
        stop_ready_flag = info["stop_ready"].astype(np.float32)

        obs = np.hstack(
            [
                noisy_linvel,                                    # 3
                noisy_gyro,                                      # 3
                local_gravity,                                   # 3
                noisy_joint_angle,                               # 12
                noisy_joint_vel,                                 # 12
                last_actions,                                    # 12
                command,                                         # 3
                position_error_normalized,                       # 2
                heading_error_normalized[:, np.newaxis],         # 1
                distance_normalized[:, np.newaxis],              # 1
                reached_flag[:, np.newaxis],                     # 1
                stop_ready_flag[:, np.newaxis],                  # 1
            ]
        )
        return obs

    def update_observation(self, state: NpEnvState):
        data = state.data

        # Compute navigation state
        self._update_nav_state(data, state.info)

        # Build observation
        obs = self._get_obs(data, state.info)
        assert obs.shape == (data.shape[0], 54)

        # Foot contacts and air time (walk_np style)
        cquerys = self._model.get_contact_query(data)
        foot_contact = cquerys.is_colliding(self.foot_check)
        state.info["contacts"] = foot_contact.reshape((self._num_envs, self.foot_check_num))
        state.info["feet_air_time"] = self.update_feet_air_time(state.info)

        # Update navigation visualization
        self._update_target_marker(data, state.info["pose_commands"])
        root_pos = self._body.get_pose(data)[:, :3]
        base_lin_vel_xy = self.get_local_linvel(data)[:, :2]
        self._update_heading_arrows(data, root_pos, state.info["desired_vel_xy"], base_lin_vel_xy)

        return state.replace(obs=obs)

    # ===================== Termination (walk_np style + nav timeout) =====================

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_check)
        termination_check = termination_check.reshape((self._num_envs, self.num_check))
        terminated = termination_check.any(axis=1)

        # Navigation-specific: timeout
        if self._cfg.max_episode_steps:
            timeout = state.info["steps"] >= self._cfg.max_episode_steps
            terminated = np.logical_or(terminated, timeout)

        # DOF velocity overflow / NaN / Inf protection
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        terminated = np.logical_or(terminated, vel_overflow)
        terminated = np.logical_or(terminated, vel_extreme)

        # Side-flip termination: tilt angle > 75°
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        local_gravity = Quaternion.rotate_inverse(root_quat, self.gravity_vec)
        gxy = np.linalg.norm(local_gravity[:, :2], axis=1)
        gz = local_gravity[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        terminated = np.logical_or(terminated, side_flip_mask)

        return state.replace(terminated=terminated)

    # ===================== Feet air time (walk_np) =====================

    def update_feet_air_time(self, info: dict):
        feet_air_time = info["feet_air_time"]
        feet_air_time += self._cfg.ctrl_dt
        feet_air_time *= ~info["contacts"]
        return feet_air_time

    # ===================== Reward (walk_np dict-based structure) =====================

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        terminated = state.terminated

        reward_dict = self._get_reward(data, state.info)

        rewards = {k: v * self._cfg.reward_config.scales[k] for k, v in reward_dict.items()}
        rwd = sum(rewards.values())
        # Allow negative rewards so that penalties (lin_vel_z, ang_vel_xy, etc.) are effective
        rwd = np.clip(rwd, -100.0, 10000.0)
        if "termination" in self._cfg.reward_config.scales:
            termination = self._reward_termination(terminated) * self._cfg.reward_config.scales["termination"]
            rwd += termination

        # Keep termination penalty instead of zeroing reward on termination
        # so the agent learns to avoid falling

        return state.replace(reward=rwd)

    def _get_reward(
        self,
        data: mtx.SceneData,
        info: dict,
    ) -> dict[str, np.ndarray]:
        commands = info["velocity_commands"]
        return {
            # Walk-style locomotion rewards
            "lin_vel_z": self._reward_lin_vel_z(data),
            "ang_vel_xy": self._reward_ang_vel_xy(data),
            "orientation": self._reward_orientation(data),
            "torques": self._reward_torques(data),
            "dof_vel": self._reward_dof_vel(data),
            "dof_acc": self._reward_dof_acc(data, info),
            "action_rate": self._reward_action_rate(info),
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, commands),
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, commands),
            "stand_still": self._reward_stand_still(data, commands),
            "hip_pos": self._reward_hip_pos(data, commands),
            "calf_pos": self._reward_calf_pos(data, commands),
            "feet_air_time": self._reward_feet_air_time(commands, info),
            # Navigation-specific rewards
            "approach": self._reward_approach(info),
            "arrival_bonus": self._reward_arrival_bonus(info),
            "stop_bonus": self._reward_stop_bonus(data, info),
        }

    # ------------ Walk-style reward functions (from walk_np.py) ----------------

    def _reward_lin_vel_z(self, data):
        # Penalize z axis base linear velocity
        return np.square(self.get_local_linvel(data)[:, 2])

    def _reward_ang_vel_xy(self, data):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.get_gyro(data)[:, :2]), axis=1)

    def _reward_orientation(self, data):
        # Penalize non flat base orientation
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        gravity = Quaternion.rotate_inverse(base_quat, self.gravity_vec)
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _reward_torques(self, data: mtx.SceneData):
        # Penalize torques
        return np.sum(np.square(data.actuator_ctrls), axis=1)

    def _reward_dof_vel(self, data):
        # Penalize dof velocities
        return np.sum(np.square(self.get_dof_vel(data)), axis=1)

    def _reward_dof_acc(self, data, info):
        # Penalize dof accelerations
        return np.sum(
            np.square((info["last_dof_vel"] - self.get_dof_vel(data)) / self._cfg.ctrl_dt),
            axis=1,
        )

    def _reward_action_rate(self, info: dict):
        # Penalize changes in actions
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_termination(self, done):
        # Terminal reward / penalty
        return done

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        return np.exp(-ang_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_stand_still(self, data, commands: np.ndarray):
        # Penalize motion at zero commands
        return np.sum(np.abs(self.get_dof_pos(data) - self.default_angles), axis=1) * (
            np.linalg.norm(commands, axis=1) < 0.1
        )

    def _reward_hip_pos(self, data, commands: np.ndarray):
        return (0.8 - np.abs(commands[:, 1])) * np.sum(
            np.square(self.get_dof_pos(data)[:, self.hip_indices] - self.default_angles[self.hip_indices]),
            axis=1,
        )

    def _reward_calf_pos(self, data, commands: np.ndarray):
        return (0.8 - np.abs(commands[:, 1])) * np.sum(
            np.square(self.get_dof_pos(data)[:, self.calf_indices] - self.default_angles[self.calf_indices]),
            axis=1,
        )

    def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
        # Reward long steps
        feet_air_time = info["feet_air_time"]
        first_contact = (feet_air_time > 0.0) * info["contacts"]
        # reward only on first contact with the ground
        rew_airTime = np.sum((feet_air_time - 0.5) * first_contact, axis=1)
        # no reward for zero command
        rew_airTime *= np.linalg.norm(commands[:, :2], axis=1) > 0.1
        return rew_airTime

    # ------------ Navigation-specific reward functions ----------------

    def _reward_approach(self, info: dict):
        """Distance improvement reward (only when not fully reached)."""
        reached_all = info["reached_all"]
        distance_to_target = info["distance_to_target"]
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach = np.clip(distance_improvement * 4.0, -1.0, 1.0)
        return np.where(reached_all, 0.0, approach)

    def _reward_arrival_bonus(self, info: dict):
        """One-time bonus for first reaching the target."""
        reached_all = info["reached_all"]
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)
        return first_time_reach.astype(np.float32)

    def _reward_stop_bonus(self, data, info: dict):
        """Reward for stopping when target is reached."""
        reached_all = info["reached_all"]
        base_lin_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05
        zero_ang_bonus = np.where(np.logical_and(reached_all, zero_ang_mask), 6.0, 0.0)
        ang_ratio = np.clip(np.abs(gyro[:, 2]) / 0.1, 0, 20)
        stop_base = 2 * (0.8 * np.exp(-(speed_xy / 0.2) ** 2) + 1.2 * np.exp(-ang_ratio ** 4))
        return np.where(reached_all, stop_base + zero_ang_bonus, 0.0)

    # ===================== Navigation helpers =====================

    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _euler_to_quat(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return np.array([qx, qy, qz, qw], dtype=np.float32)

    # ===================== Reset =====================

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg = self._cfg
        num_envs = data.shape[0]

        # Random initial positions (navigation-specific)
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_x = np.random.uniform(pos_range[0], pos_range[2], num_envs)
        robot_init_y = np.random.uniform(pos_range[1], pos_range[3], num_envs)
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)

        # Target positions relative to robot (navigation-specific)
        target_offset = np.random.uniform(
            low=cfg.commands.pose_command_range[:2],
            high=cfg.commands.pose_command_range[3:5],
            size=(num_envs, 2),
        )
        target_positions = robot_init_pos + target_offset

        # Target headings
        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2],
            high=cfg.commands.pose_command_range[5],
            size=(num_envs, 1),
        )

        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # Build initial DOF state
        init_dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1))
        init_dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1))

        noise_pos = np.zeros((*data.shape, self._num_dof_pos), dtype=np.float32)
        noise_pos[:, 3] = robot_init_x - cfg.init_state.pos[0]
        noise_pos[:, 4] = robot_init_y - cfg.init_state.pos[1]

        dof_pos = init_dof_pos + noise_pos
        dof_vel = init_dof_vel

        # Normalize quaternions
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start : self._base_quat_end]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                dof_pos[env_idx, self._base_quat_start : self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start : self._base_quat_end] = np.array(
                    [0.0, 0.0, 0.0, 1.0], dtype=np.float32
                )

            if self._robot_arrow_body is not None:
                raq = dof_pos[env_idx, self._robot_arrow_dof_start + 3 : self._robot_arrow_dof_end]
                n = np.linalg.norm(raq)
                if n > 1e-6:
                    dof_pos[env_idx, self._robot_arrow_dof_start + 3 : self._robot_arrow_dof_end] = raq / n
                else:
                    dof_pos[env_idx, self._robot_arrow_dof_start + 3 : self._robot_arrow_dof_end] = np.array(
                        [0.0, 0.0, 0.0, 1.0], dtype=np.float32
                    )
                daq = dof_pos[env_idx, self._desired_arrow_dof_start + 3 : self._desired_arrow_dof_end]
                n = np.linalg.norm(daq)
                if n > 1e-6:
                    dof_pos[env_idx, self._desired_arrow_dof_start + 3 : self._desired_arrow_dof_end] = daq / n
                else:
                    dof_pos[env_idx, self._desired_arrow_dof_start + 3 : self._desired_arrow_dof_end] = np.array(
                        [0.0, 0.0, 0.0, 1.0], dtype=np.float32
                    )

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # Build info (walk_np style + navigation extensions)
        info = {
            "pose_commands": pose_commands,
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "feet_air_time": np.zeros((num_envs, self.foot_check_num), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.foot_check_num), dtype=np.bool_),
            "ever_reached": np.zeros(num_envs, dtype=bool),
        }

        # Compute navigation state
        self._update_nav_state(data, info)
        info["min_distance"] = info["distance_to_target"].copy()

        # Update visualization
        self._update_target_marker(data, pose_commands)
        root_pos = self._body.get_pose(data)[:, :3]
        base_lin_vel_xy = self.get_local_linvel(data)[:, :2]
        self._update_heading_arrows(data, root_pos, info["desired_vel_xy"], base_lin_vel_xy)

        obs = self._get_obs(data, info)
        assert obs.shape == (num_envs, 54)

        return obs, info

    # ===================== Visualization helpers =====================

    def _normalize_all_quaternions(self, dof_pos: np.ndarray):
        """Ensure all quaternion DOFs are normalized before set_dof_pos."""
        num_envs = dof_pos.shape[0]
        for env_idx in range(num_envs):
            # Base quaternion
            quat = dof_pos[env_idx, self._base_quat_start : self._base_quat_end]
            n = np.linalg.norm(quat)
            if n > 1e-6:
                dof_pos[env_idx, self._base_quat_start : self._base_quat_end] = quat / n
            else:
                dof_pos[env_idx, self._base_quat_start : self._base_quat_end] = [0.0, 0.0, 0.0, 1.0]

            # Arrow quaternions
            if self._robot_arrow_body is not None:
                for start, end in [
                    (self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end),
                    (self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end),
                ]:
                    if end <= dof_pos.shape[1]:
                        q = dof_pos[env_idx, start:end]
                        qn = np.linalg.norm(q)
                        if qn > 1e-6:
                            dof_pos[env_idx, start:end] = q / qn
                        else:
                            dof_pos[env_idx, start:end] = [0.0, 0.0, 0.0, 1.0]
        return dof_pos

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start : self._target_marker_dof_end] = [
                target_x, target_y, target_yaw,
            ]
        all_dof_pos = self._normalize_all_quaternions(all_dof_pos)
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)

    def _update_heading_arrows(
        self,
        data: mtx.SceneData,
        robot_pos: np.ndarray,
        desired_vel_xy: np.ndarray,
        base_lin_vel_xy: np.ndarray,
    ):
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return

        num_envs = data.shape[0]
        arrow_height = 0.76
        all_dof_pos = data.dof_pos.copy()

        for env_idx in range(num_envs):
            # Current heading arrow (green)
            cur_v = base_lin_vel_xy[env_idx]
            cur_yaw = np.arctan2(cur_v[1], cur_v[0]) if np.linalg.norm(cur_v) > 1e-3 else 0.0
            robot_arrow_pos = np.array(
                [robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32
            )
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            n = np.linalg.norm(robot_arrow_quat)
            if n > 1e-6:
                robot_arrow_quat /= n
            all_dof_pos[env_idx, self._robot_arrow_dof_start : self._robot_arrow_dof_end] = np.concatenate(
                [robot_arrow_pos, robot_arrow_quat]
            )

            # Desired heading arrow (blue)
            des_v = desired_vel_xy[env_idx]
            des_yaw = np.arctan2(des_v[1], des_v[0]) if np.linalg.norm(des_v) > 1e-3 else 0.0
            desired_arrow_pos = np.array(
                [robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32
            )
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            n = np.linalg.norm(desired_arrow_quat)
            if n > 1e-6:
                desired_arrow_quat /= n
            all_dof_pos[env_idx, self._desired_arrow_dof_start : self._desired_arrow_dof_end] = np.concatenate(
                [desired_arrow_pos, desired_arrow_quat]
            )

        all_dof_pos = self._normalize_all_quaternions(all_dof_pos)
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
