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

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import VBotNavSection1EnvCfg


@registry.env("vbot_nav_section1", "np")
class VBotNavSection1Env(NpEnv):
    """
    VBot Navigation Section 1 Competition Environment
    Obstacle navigation with smileys, hongbaos, and finish zone
    """
    _cfg: VBotNavSection1EnvCfg

    def __init__(self, cfg: VBotNavSection1EnvCfg, num_envs: int = 1):
        # Call parent class initialization
        super().__init__(cfg, num_envs=num_envs)

        self._debug_logs = bool(getattr(cfg, "debug_logs", False))

        # Initialize robot body and contacts
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()

        # Get target marker body
        self._target_marker_body = self._model.get_body("target_marker")

        # Get arrow bodies (for visualization, does not affect physics)
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None

        # Action and observation spaces
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # Observation space: 75 dimensions (54 base + 21 competition-specific)
        # Base: 54 (linvel:3, gyro:3, gravity:3, joint_angle:12, joint_vel:12, last_actions:12, 
        #           commands:3, position_error:2, heading_error:1, distance:1, reached:1, stop_ready:1)
        # Competition: 21 (smiley_relative_pos:6, hongbao_relative_pos:6, trigger_flags:6, 
        #                   finish_relative_pos:2, finish_flag:1)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(75,), dtype=np.float32)

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        # Find target_marker DOF indices
        self._find_target_marker_dof_indices()

        # Find arrow DOF indices
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()

        # Initialize buffers
        self._init_buffer()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _init_buffer(self):
        """Initialize buffers and parameters"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
        # Default waypoint distance for progress tracking
        self.DEFAULT_WAYPOINT_DISTANCE = 10.0

        # Normalization coefficients
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )

        # Set default joint angles
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        self.action_filter_alpha = 0.3

        # Competition zone coordinates (converted to numpy arrays)
        self.start_zone_center = np.array(cfg.start_zone_center, dtype=np.float32)
        self.start_zone_radius = cfg.start_zone_radius
        
        self.smiley_positions = np.array(cfg.smiley_positions, dtype=np.float32)  # (3, 2)
        self.smiley_radius = cfg.smiley_radius
        
        self.hongbao_positions = np.array(cfg.hongbao_positions, dtype=np.float32)  # (3, 2)
        self.hongbao_radius = cfg.hongbao_radius
        
        self.finish_zone_center = np.array(cfg.finish_zone_center, dtype=np.float32)
        self.finish_zone_radius = cfg.finish_zone_radius
        
        # Boundary limits
        self.boundary_x_min = cfg.boundary_x_min
        self.boundary_x_max = cfg.boundary_x_max
        self.boundary_y_min = cfg.boundary_y_min
        self.boundary_y_max = cfg.boundary_y_max
        
        # Celebration parameters
        self.celebration_duration = cfg.celebration_duration
        self.celebration_movement_threshold = cfg.celebration_movement_threshold
        
        # Fall detection thresholds
        self.fall_threshold_roll_pitch = np.deg2rad(45.0)
        
        # Initialize trigger flags for each environment
        n = self._num_envs
        self.smiley_triggered = np.zeros((n, 3), dtype=bool)  # 3 smileys
        self.hongbao_triggered = np.zeros((n, 3), dtype=bool)  # 3 hongbaos
        self.finish_triggered = np.zeros(n, dtype=bool)
        
        # Celebration tracking
        self.celebration_start_time = np.full(n, -1.0, dtype=np.float32)
        self.celebration_completed = np.zeros(n, dtype=bool)
        
        # Cumulative scores for tracking (not used in reward, just for logging)
        self.cumulative_scores = np.zeros(n, dtype=np.float32)

    def _find_target_marker_dof_indices(self):
        """Find target_marker indices in dof_pos"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10

    def _find_arrow_dof_indices(self):
        """Find arrow indices in dof_pos"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36

        arrow_init_height = self._cfg.init_state.pos[2] + 0.5
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]

    def _init_contact_geometry(self):
        """Initialize contact geometry"""
        try:
            self._base_contact_geom = self._model.get_geom("collision_middle_box")
        except Exception:
            self._base_contact_geom = None

    def _log(self, message: str):
        if self._debug_logs:
            print(message)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """Apply actions to the environment (torque control via position targets)."""
        state.info["last_dof_vel"] = self._body.get_joint_dof_vel(state.data)

        state.info["last_actions"] = state.info.get(
            "current_actions",
            np.zeros((state.data.shape[0], self._num_action), dtype=np.float32),
        )

        state.info["next_actions"] = actions

        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions
                + (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )

        state.info["current_actions"] = state.info["filtered_actions"]

        # Convert actions to target joint positions (position control)
        target_pos = self.default_angles + state.info["current_actions"] * self._cfg.control_config.action_scale
        state.data.actuator_ctrls = target_pos

        return state

    def _extract_root_state(self, data: mtx.SceneData):
        """Extract root position, quaternion and velocity"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_vel

    def _compute_observation(self, data: mtx.SceneData, state: NpEnvState):
        """Compute observations for the environment"""
        cfg = self._cfg

        # Extract state
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self._body.get_joint_dof_pos(data)
        joint_vel = self._body.get_joint_dof_vel(data)

        # Base linear velocity and gyro from sensors
        base_lin_vel = self._model.get_sensor_value(cfg.sensor.base_linvel, data)
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)

        # Gravity in base frame
        gravity_world = np.array([0, 0, -1], dtype=np.float32)
        gravity_world = np.tile(gravity_world, (data.shape[0], 1))
        projected_gravity = Quaternion.rotate_inverse(root_quat, gravity_world)

        # Joint angles relative to default
        joint_pos_rel = joint_pos[:, -self._num_action:] - self.default_angles

        # Current robot heading (yaw angle)
        robot_heading = np.arctan2(
            2 * (root_quat[:, 3] * root_quat[:, 2] + root_quat[:, 0] * root_quat[:, 1]),
            1 - 2 * (root_quat[:, 1]**2 + root_quat[:, 2]**2)
        )

        # Target position (finish zone)
        pose_commands = np.tile(self.finish_zone_center, (data.shape[0], 1))
        position_error = pose_commands - root_pos[:, :2]
        distance_to_target = np.linalg.norm(position_error, axis=-1)

        # Heading to target
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_diff = desired_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)

        # Reached target flag
        position_threshold = self.finish_zone_radius
        reached_all = distance_to_target < position_threshold

        # Compute desired velocity commands (simple P controller)
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        # Desired yaw rate
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_diff) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        # Normalize observations
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        # Task-related observations
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

        smiley_triggered = state.info.get("smiley_triggered")
        if smiley_triggered is None or smiley_triggered.shape[0] != data.shape[0]:
            smiley_triggered = np.zeros((data.shape[0], 3), dtype=bool)
            state.info["smiley_triggered"] = smiley_triggered

        hongbao_triggered = state.info.get("hongbao_triggered")
        if hongbao_triggered is None or hongbao_triggered.shape[0] != data.shape[0]:
            hongbao_triggered = np.zeros((data.shape[0], 3), dtype=bool)
            state.info["hongbao_triggered"] = hongbao_triggered

        finish_triggered = state.info.get("finish_triggered")
        if finish_triggered is None or finish_triggered.shape[0] != data.shape[0]:
            finish_triggered = np.zeros((data.shape[0],), dtype=bool)
            state.info["finish_triggered"] = finish_triggered

        # === Competition-specific observations (21 dimensions) ===
        
        # Relative position to each smiley (3 x 2 = 6 dimensions)
        smiley_relative_pos = []
        for i in range(3):
            smiley_pos = self.smiley_positions[i]
            relative_pos = smiley_pos - root_pos[:, :2]
            smiley_relative_pos.append(relative_pos / 5.0)  # Normalized by 5m
        smiley_relative_pos = np.concatenate(smiley_relative_pos, axis=-1)  # (n_envs, 6)
        
        # Relative position to each hongbao (3 x 2 = 6 dimensions)
        hongbao_relative_pos = []
        for i in range(3):
            hongbao_pos = self.hongbao_positions[i]
            relative_pos = hongbao_pos - root_pos[:, :2]
            hongbao_relative_pos.append(relative_pos / 5.0)  # Normalized by 5m
        hongbao_relative_pos = np.concatenate(hongbao_relative_pos, axis=-1)  # (n_envs, 6)
        
        # Trigger flags (6 dimensions: 3 smileys + 3 hongbaos)
        trigger_flags = np.concatenate([
            smiley_triggered.astype(np.float32),  # (n_envs, 3)
            hongbao_triggered.astype(np.float32),  # (n_envs, 3)
        ], axis=-1)  # (n_envs, 6)
        
        # Relative position to finish zone (2 dimensions)
        finish_relative_pos = (self.finish_zone_center - root_pos[:, :2]) / 5.0  # (n_envs, 2)
        
        # Finish zone trigger flag (1 dimension)
        finish_flag = finish_triggered.astype(np.float32)[:, np.newaxis]  # (n_envs, 1)

        # Concatenate all observations
        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2
                heading_error_normalized[:, np.newaxis],  # 1
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
                # Competition-specific observations
                smiley_relative_pos,  # 6
                hongbao_relative_pos,  # 6
                trigger_flags,  # 6
                finish_relative_pos,  # 2
                finish_flag,  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 75), f"Expected obs shape (*, 75), got {obs.shape}"

        # Prevent NaN/Inf propagation
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Update target marker and arrows
        self._update_target_marker(data, pose_commands)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        # Update trigger states
        self._update_trigger_states(root_pos, state.info)

        # Compute rewards
        reward = self._compute_reward(data, state.info, velocity_commands, root_pos, base_lin_vel)

        # Compute termination
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        return state

    def _update_trigger_states(self, root_pos, info):
        """Update trigger states for smileys, hongbaos, and finish zone"""
        robot_pos = root_pos[:, :2]  # (n_envs, 2)
        env_ids = info.get("env_ids")
        smiley_triggered = info.get("smiley_triggered")
        if smiley_triggered is None or smiley_triggered.shape[0] != robot_pos.shape[0]:
            smiley_triggered = np.zeros((robot_pos.shape[0], 3), dtype=bool)
            info["smiley_triggered"] = smiley_triggered

        hongbao_triggered = info.get("hongbao_triggered")
        if hongbao_triggered is None or hongbao_triggered.shape[0] != robot_pos.shape[0]:
            hongbao_triggered = np.zeros((robot_pos.shape[0], 3), dtype=bool)
            info["hongbao_triggered"] = hongbao_triggered

        finish_triggered = info.get("finish_triggered")
        if finish_triggered is None or finish_triggered.shape[0] != robot_pos.shape[0]:
            finish_triggered = np.zeros((robot_pos.shape[0],), dtype=bool)
            info["finish_triggered"] = finish_triggered

        celebration_start_time = info.get("celebration_start_time")
        if celebration_start_time is None or celebration_start_time.shape[0] != robot_pos.shape[0]:
            celebration_start_time = np.full(robot_pos.shape[0], -1.0, dtype=np.float32)
            info["celebration_start_time"] = celebration_start_time
        
        # Check smiley triggers
        for i in range(3):
            smiley_pos = self.smiley_positions[i]  # (2,)
            dist = np.linalg.norm(robot_pos - smiley_pos, axis=-1)  # (n_envs,)
            newly_triggered = (dist < self.smiley_radius) & ~smiley_triggered[:, i]
            smiley_triggered[:, i] |= newly_triggered
            
            # Log first trigger
            if np.any(newly_triggered):
                local_ids = np.where(newly_triggered)[0]
                for local_id in local_ids:
                    env_id = int(env_ids[local_id]) if env_ids is not None else int(local_id)
                    self._log(f"[ENV {env_id}] Smiley {i+1} triggered! +4 points")
        
        # Check hongbao triggers
        for i in range(3):
            hongbao_pos = self.hongbao_positions[i]  # (2,)
            dist = np.linalg.norm(robot_pos - hongbao_pos, axis=-1)  # (n_envs,)
            newly_triggered = (dist < self.hongbao_radius) & ~hongbao_triggered[:, i]
            hongbao_triggered[:, i] |= newly_triggered
            
            # Log first trigger
            if np.any(newly_triggered):
                local_ids = np.where(newly_triggered)[0]
                for local_id in local_ids:
                    env_id = int(env_ids[local_id]) if env_ids is not None else int(local_id)
                    self._log(f"[ENV {env_id}] Hongbao {i+1} triggered! +2 points")
        
        # Check finish zone trigger
        finish_dist = np.linalg.norm(robot_pos - self.finish_zone_center, axis=-1)  # (n_envs,)
        newly_triggered_finish = (finish_dist < self.finish_zone_radius) & ~finish_triggered
        finish_triggered |= newly_triggered_finish
        
        # Start celebration timer when finish is first triggered
        time_elapsed_value = info.get("time_elapsed", 0.0)
        if isinstance(time_elapsed_value, (int, float)):
            time_elapsed_array = np.full(self._num_envs, time_elapsed_value, dtype=np.float32)
        else:
            time_elapsed_array = time_elapsed_value
            
        for local_idx in np.where(newly_triggered_finish)[0]:
            env_id = int(env_ids[local_idx]) if env_ids is not None else int(local_idx)
            if celebration_start_time[local_idx] < 0:
                celebration_start_time[local_idx] = time_elapsed_array[local_idx] if hasattr(time_elapsed_array, '__getitem__') else time_elapsed_value
                self._log(f"[ENV {env_id}] Finish zone reached! +20 points. Start celebration.")

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands, root_pos, base_lin_vel):
        """Compute rewards for the competition"""
        cfg = self._cfg
        n_envs = data.shape[0]
        smiley_triggered = info.get("smiley_triggered")
        if smiley_triggered is None or smiley_triggered.shape[0] != n_envs:
            smiley_triggered = np.zeros((n_envs, 3), dtype=bool)
            info["smiley_triggered"] = smiley_triggered

        hongbao_triggered = info.get("hongbao_triggered")
        if hongbao_triggered is None or hongbao_triggered.shape[0] != n_envs:
            hongbao_triggered = np.zeros((n_envs, 3), dtype=bool)
            info["hongbao_triggered"] = hongbao_triggered

        finish_triggered = info.get("finish_triggered")
        if finish_triggered is None or finish_triggered.shape[0] != n_envs:
            finish_triggered = np.zeros((n_envs,), dtype=bool)
            info["finish_triggered"] = finish_triggered

        celebration_start_time = info.get("celebration_start_time")
        if celebration_start_time is None or celebration_start_time.shape[0] != n_envs:
            celebration_start_time = np.full(n_envs, -1.0, dtype=np.float32)
            info["celebration_start_time"] = celebration_start_time

        celebration_completed = info.get("celebration_completed")
        if celebration_completed is None or celebration_completed.shape[0] != n_envs:
            celebration_completed = np.zeros(n_envs, dtype=bool)
            info["celebration_completed"] = celebration_completed
        
        # Base reward components
        reward = np.zeros(n_envs, dtype=np.float32)
        
        # 1. Alive bonus (+0.01 per step for stability)
        reward += 0.01
        
        # 2. Waypoint-based progress rewards
        # Compute distance to next active waypoint
        robot_pos = root_pos[:, :2]  # (n_envs, 2)
        
        # Determine active waypoint for each environment
        # Priority: smiley1 -> smiley2 -> smiley3 -> hongbao1 -> hongbao2 -> hongbao3 -> finish
        for env_id in range(n_envs):
            # Find next waypoint
            if not smiley_triggered[env_id, 0]:
                target_pos = self.smiley_positions[0]
            elif not smiley_triggered[env_id, 1]:
                target_pos = self.smiley_positions[1]
            elif not smiley_triggered[env_id, 2]:
                target_pos = self.smiley_positions[2]
            elif not hongbao_triggered[env_id, 0]:
                target_pos = self.hongbao_positions[0]
            elif not hongbao_triggered[env_id, 1]:
                target_pos = self.hongbao_positions[1]
            elif not hongbao_triggered[env_id, 2]:
                target_pos = self.hongbao_positions[2]
            else:
                target_pos = self.finish_zone_center
            
            # Progress reward based on distance reduction
            current_dist = np.linalg.norm(robot_pos[env_id] - target_pos)
            last_dist = info.get("last_waypoint_distance", np.full(n_envs, self.DEFAULT_WAYPOINT_DISTANCE))[env_id]
            progress = last_dist - current_dist
            reward[env_id] += np.clip(progress * 0.5, -0.1, 0.5)  # Dense progress reward
            
            # Update last distance
            if "last_waypoint_distance" not in info:
                info["last_waypoint_distance"] = np.full(n_envs, self.DEFAULT_WAYPOINT_DISTANCE, dtype=np.float32)
            info["last_waypoint_distance"][env_id] = current_dist
        
        # 3. Sparse rewards for triggers
        # Smiley bonuses (first trigger only, +4 each)
        for i in range(3):
            newly_triggered = smiley_triggered[:, i] & ~info.get(f"smiley_{i}_rewarded", np.zeros(n_envs, dtype=bool))
            reward += newly_triggered.astype(np.float32) * 4.0
            if f"smiley_{i}_rewarded" not in info:
                info[f"smiley_{i}_rewarded"] = np.zeros(n_envs, dtype=bool)
            info[f"smiley_{i}_rewarded"] |= newly_triggered
        
        # Hongbao bonuses (first trigger only, +2 each)
        for i in range(3):
            newly_triggered = hongbao_triggered[:, i] & ~info.get(f"hongbao_{i}_rewarded", np.zeros(n_envs, dtype=bool))
            reward += newly_triggered.astype(np.float32) * 2.0
            if f"hongbao_{i}_rewarded" not in info:
                info[f"hongbao_{i}_rewarded"] = np.zeros(n_envs, dtype=bool)
            info[f"hongbao_{i}_rewarded"] |= newly_triggered
        
        # Finish zone bonus (first trigger only, +20)
        newly_triggered_finish = finish_triggered & ~info.get("finish_rewarded", np.zeros(n_envs, dtype=bool))
        reward += newly_triggered_finish.astype(np.float32) * 20.0
        if "finish_rewarded" not in info:
            info["finish_rewarded"] = np.zeros(n_envs, dtype=bool)
        info["finish_rewarded"] |= newly_triggered_finish
        
        # 4. Celebration bonus (+2 for staying still at finish for celebration_duration)
        time_elapsed_value = info.get("time_elapsed", 0.0)
        # Ensure time_elapsed is an array for consistent indexing
        if isinstance(time_elapsed_value, (int, float)):
            time_elapsed = np.full(n_envs, time_elapsed_value, dtype=np.float32)
        else:
            time_elapsed = time_elapsed_value
        
        for env_id in range(n_envs):
            if finish_triggered[env_id] and not celebration_completed[env_id]:
                if celebration_start_time[env_id] >= 0:
                    celebration_time = time_elapsed[env_id] - celebration_start_time[env_id]
                    speed = np.linalg.norm(base_lin_vel[env_id, :2])
                    
                    # Check if robot is staying still
                    if speed < self.celebration_movement_threshold:
                        if celebration_time >= self.celebration_duration:
                            # Award celebration bonus
                            if not info.get("celebration_rewarded", np.zeros(n_envs, dtype=bool))[env_id]:
                                reward[env_id] += 2.0
                                celebration_completed[env_id] = True
                                if "celebration_rewarded" not in info:
                                    info["celebration_rewarded"] = np.zeros(n_envs, dtype=bool)
                                info["celebration_rewarded"][env_id] = True
                                self._log(f"[ENV {env_id}] Celebration completed! +2 points")
                    else:
                        # Reset celebration timer if moving
                        celebration_start_time[env_id] = time_elapsed[env_id]

        info["celebration_start_time"] = celebration_start_time
        info["celebration_completed"] = celebration_completed
        
        # 5. Stability penalties (keep robot stable and energy-efficient)
        root_quat = self._body.get_pose(data)[:, 3:7]
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # Orientation penalty
        orientation_penalty = (np.abs(roll) + np.abs(pitch)) * cfg.reward_config.scales.get("orientation", -0.05)
        reward += orientation_penalty
        
        # Action rate penalty
        last_actions = info["current_actions"]
        current_actions = info.get("next_actions", last_actions)
        action_rate = np.sum(np.square(current_actions - last_actions), axis=-1)
        reward += action_rate * cfg.reward_config.scales.get("action_rate", -0.01)
        
        # Torque penalty (energy efficiency)
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=-1) * cfg.reward_config.scales.get("torques", -1e-5)
        reward += torque_penalty
        
        return reward

    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        """Compute termination conditions"""
        data = state.data
        n_envs = data.shape[0]
        
        # Grace period: no termination in first 20 frames
        GRACE_PERIOD = 20
        steps = state.info.get("steps", np.zeros(n_envs, dtype=np.int32))
        if isinstance(steps, np.ndarray):
            in_grace_period = steps < GRACE_PERIOD
        else:
            in_grace_period = np.full(n_envs, steps < GRACE_PERIOD, dtype=bool)
        
        # 1. Base contact with ground (fall detection)
        try:
            base_contact_value = self._model.get_sensor_value("base_contact", data)
            if base_contact_value.ndim == 0:
                base_contact = np.array([base_contact_value > 0.1], dtype=bool)
            elif base_contact_value.shape[0] != n_envs:
                base_contact = np.full(n_envs, base_contact_value.flatten()[0] > 0.1, dtype=bool)
            else:
                base_contact = (base_contact_value > 0.1).flatten()[:n_envs]
        except Exception as e:
            base_contact = np.zeros(n_envs, dtype=bool)
        
        # 2. Attitude failure (roll/pitch > 45 degrees)
        try:
            root_pos, root_quat, _ = self._extract_root_state(data)
            qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
            pitch = np.arcsin(sinp)
            
            attitude_fail = (np.abs(roll) > self.fall_threshold_roll_pitch) | (np.abs(pitch) > self.fall_threshold_roll_pitch)
        except Exception:
            attitude_fail = np.zeros(n_envs, dtype=bool)
            root_pos = np.zeros((n_envs, 3), dtype=np.float32)
        
        # 3. Out of bounds detection
        out_of_bounds = (
            (root_pos[:, 0] < self.boundary_x_min) |
            (root_pos[:, 0] > self.boundary_x_max) |
            (root_pos[:, 1] < self.boundary_y_min) |
            (root_pos[:, 1] > self.boundary_y_max)
        )
        
        # Combine termination conditions (ignore during grace period)
        terminated = (base_contact | attitude_fail | out_of_bounds) & ~in_grace_period
        
        # Apply termination penalty (-10.0)
        penalty = np.where(terminated, -10.0, 0.0)
        state.reward += penalty
        
        # Log termination reasons
        for env_id in np.where(terminated)[0]:
            reasons = []
            if base_contact[env_id]:
                reasons.append("base_contact")
            if attitude_fail[env_id]:
                reasons.append("attitude_fail")
            if out_of_bounds[env_id]:
                reasons.append("out_of_bounds")
            self._log(f"[TERM] ENV {env_id} terminated: {', '.join(reasons)}")
        
        state.terminated = terminated
        return state

    def _update_target_marker(self, data: mtx.SceneData, pose_commands):
        """Update target marker position for visualization"""
        n_envs = data.shape[0]
        target_x = pose_commands[:, 0]
        target_y = pose_commands[:, 1]
        target_z = np.full(n_envs, 0.05, dtype=np.float32)  # Slight elevation for visibility
        
        for env_idx in range(n_envs):
            data.dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x[env_idx], target_y[env_idx], target_z[env_idx]
            ]

    def _update_heading_arrows(self, data: mtx.SceneData, root_pos, desired_vel_xy, base_lin_vel_xy):
        """Update heading arrows for visualization"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        n_envs = data.shape[0]
        arrow_height = self._cfg.init_state.pos[2] + 0.5
        
        for env_idx in range(n_envs):
            # Robot heading arrow (actual velocity direction)
            actual_vel = base_lin_vel_xy[env_idx]
            actual_speed = np.linalg.norm(actual_vel)
            if actual_speed > 0.01:
                actual_heading = np.arctan2(actual_vel[1], actual_vel[0])
                robot_quat = self._euler_to_quat(0, 0, actual_heading)
            else:
                robot_quat = [0, 0, 0, 1]
            
            data.dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [
                root_pos[env_idx, 0], root_pos[env_idx, 1], arrow_height,
                robot_quat[0], robot_quat[1], robot_quat[2], robot_quat[3]
            ]
            
            # Desired heading arrow (desired velocity direction)
            desired_vel = desired_vel_xy[env_idx]
            desired_speed = np.linalg.norm(desired_vel)
            if desired_speed > 0.01:
                desired_heading = np.arctan2(desired_vel[1], desired_vel[0])
                desired_quat = self._euler_to_quat(0, 0, desired_heading)
            else:
                desired_quat = [0, 0, 0, 1]
            
            data.dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [
                root_pos[env_idx, 0], root_pos[env_idx, 1], arrow_height,
                desired_quat[0], desired_quat[1], desired_quat[2], desired_quat[3]
            ]

    def _euler_to_quat(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion (x, y, z, w)"""
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
        
        return [qx, qy, qz, qw]

    def reset(
        self,
        data: mtx.SceneData,
        done: np.ndarray = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment"""
        cfg = self._cfg
        n_envs = data.shape[0]
        
        # Random spawn in START zone
        # Use uniform random distribution within circle
        angles = np.random.uniform(0, 2*np.pi, n_envs)
        radii = self.start_zone_radius * np.sqrt(np.random.uniform(0, 1, n_envs))
        spawn_x = self.start_zone_center[0] + radii * np.cos(angles)
        spawn_y = self.start_zone_center[1] + radii * np.sin(angles)
        spawn_z = cfg.init_state.pos[2]
        
        # Random yaw orientation
        spawn_yaw = np.random.uniform(-np.pi, np.pi, n_envs)
        spawn_quat = np.array([self._euler_to_quat(0, 0, yaw) for yaw in spawn_yaw])
        
        # Validate quaternions
        quat_norms = np.linalg.norm(spawn_quat, axis=-1)
        invalid_quat = (quat_norms < 0.9) | (quat_norms > 1.1)
        if np.any(invalid_quat):
            spawn_quat[invalid_quat] = np.array([0, 0, 0, 1])
        else:
            spawn_quat = spawn_quat / quat_norms[:, np.newaxis]
        
        dof_pos = np.tile(self._init_dof_pos, (n_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (n_envs, 1))

        # Set base position and orientation
        dof_pos[:, 3:6] = np.column_stack([spawn_x, spawn_y, np.full(n_envs, spawn_z, dtype=np.float32)])
        dof_pos[:, self._base_quat_start:self._base_quat_end] = spawn_quat

        # Normalize base quaternions
        for env_idx in range(n_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        
        # Reset trigger flags (only when mapping is known)
        if done is None and n_envs == self._num_envs:
            self.smiley_triggered.fill(False)
            self.hongbao_triggered.fill(False)
            self.finish_triggered.fill(False)
            self.celebration_start_time.fill(-1.0)
            self.celebration_completed.fill(False)
        elif done is not None and done.shape == (self._num_envs,):
            self.smiley_triggered[done] = False
            self.hongbao_triggered[done] = False
            self.finish_triggered[done] = False
            self.celebration_start_time[done] = -1.0
            self.celebration_completed[done] = False
        
        # Initialize info dict
        info = {
            "current_actions": np.zeros((n_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(n_envs, dtype=np.int32),
            "last_waypoint_distance": np.full(n_envs, self.DEFAULT_WAYPOINT_DISTANCE, dtype=np.float32),
            "time_elapsed": np.zeros(n_envs, dtype=np.float32),  # Store as array for consistency
            "smiley_triggered": np.zeros((n_envs, 3), dtype=bool),
            "hongbao_triggered": np.zeros((n_envs, 3), dtype=bool),
            "finish_triggered": np.zeros(n_envs, dtype=bool),
            "celebration_start_time": np.full(n_envs, -1.0, dtype=np.float32),
            "celebration_completed": np.zeros(n_envs, dtype=bool),
            "celebration_rewarded": np.zeros(n_envs, dtype=bool),
            "finish_rewarded": np.zeros(n_envs, dtype=bool),
            "smiley_0_rewarded": np.zeros(n_envs, dtype=bool),
            "smiley_1_rewarded": np.zeros(n_envs, dtype=bool),
            "smiley_2_rewarded": np.zeros(n_envs, dtype=bool),
            "hongbao_0_rewarded": np.zeros(n_envs, dtype=bool),
            "hongbao_1_rewarded": np.zeros(n_envs, dtype=bool),
            "hongbao_2_rewarded": np.zeros(n_envs, dtype=bool),
        }
        if done is not None:
            info["env_ids"] = np.where(done)[0]
        
        obs = np.zeros((n_envs, self.observation_space.shape[0]), dtype=np.float32)
        reward = np.zeros((n_envs,), dtype=np.float32)
        terminated = np.zeros((n_envs,), dtype=bool)
        truncated = np.zeros((n_envs,), dtype=bool)
        state = NpEnvState(data=data, obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

        # Compute initial observation
        state = self._compute_observation(data, state)

        # Prevent NaN/Inf
        state.obs = np.nan_to_num(state.obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        self._log(f"[RESET] Spawned {n_envs} robots at START zone (x={spawn_x}, y={spawn_y})")
        
        return state.obs, state.info

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Update environment state"""
        data = state.data

        # Compute observation and rewards
        state = self._compute_observation(data, state)
        
        # Update time elapsed (approximate, assuming constant dt)
        current_time = state.info.get("time_elapsed", np.zeros(data.shape[0], dtype=np.float32))
        if isinstance(current_time, (int, float)):
            current_time = np.full(data.shape[0], current_time, dtype=np.float32)
        state.info["time_elapsed"] = current_time + self._cfg.sim_dt
        
        return state
