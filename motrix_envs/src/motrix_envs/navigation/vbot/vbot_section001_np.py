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

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection001EnvCfg


@registry.env("vbot_navigation_section001", "np")
class VBotSection001Env(NpEnv):
    """
    VBot在Section001地形上的导航任务
    继承自NpEnv，使用VBotSection001EnvCfg配置
    """
    _cfg: VBotSection001EnvCfg
    
    def __init__(self, cfg: VBotSection001EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body(用于可视化，不影响物理)
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 观测空间：54维
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)
        
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
    
    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )
        
        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
        
        self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = 0.3

        # --- 初始化摔倒/越界/计分相关变量(优先从cfg读取，缺失时使用合理默认值)
        # 倾斜角阈值(rad)，默认 45 deg
        self.fall_threshold_roll_pitch = getattr(cfg, "fall_threshold_roll_pitch", np.deg2rad(45.0))
        # 足部接触阈值，默认 0.05
        self.fall_contact_threshold = getattr(cfg, "fall_contact_threshold", 0.05)
        # 连续悬空帧数阈值，默认 10
        self.fall_frames_threshold = getattr(cfg, "fall_frames_threshold", 10)

        # 赛场/边界/触发点(两个触发点)
        self.arena_center = np.array(getattr(cfg, "arena_center", np.array([0.0, 0.0], dtype=np.float32)), dtype=np.float32)
        self.boundary_radius = getattr(cfg, "boundary_radius", 10.0)
        self.target_point_a = np.array(getattr(cfg, "target_point_a", self.arena_center), dtype=np.float32)
        self.target_point_b = np.array(getattr(cfg, "target_point_b", self.arena_center), dtype=np.float32)

        # 计分与状态数组，按并行环境数初始化
        try:
            n = self._num_envs
        except Exception:
            n = getattr(self, "num_envs", 1)

        self.dog_fall_frames = np.zeros(n, dtype=np.int32)
        self.dog_penalty_flags = np.zeros(n, dtype=bool)
        self.dog_scores = np.zeros(n, dtype=np.float32)
        self.dog_stage = np.zeros(n, dtype=np.int32)
        self.dog_triggered_a = np.zeros(n, dtype=bool)
        self.dog_triggered_b = np.zeros(n, dtype=bool)

    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
        arrow_init_height = self._cfg.init_state.pos[2] + 0.5 
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
    
    def _init_foot_contact(self):
        self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
        self.num_foot_check = 4

    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        cfg = self._cfg
        try:
            self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)
        except Exception:
            self.ground_index = None

        # 初始化接触检测矩阵
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        """初始化终止接触检测"""
        cfg = self._cfg
        base_indices = []
        for base_name in getattr(cfg.asset, "terminate_after_contacts_on", []):
            try:
                base_idx = self._model.get_geom_index(base_name)
                if base_idx is not None:
                    base_indices.append(base_idx)
                else:
                    print(f"Warning: Geom '{base_name}' not found in model")
            except Exception as e:
                print(f"Warning: Error finding base geom '{base_name}': {e}")

        if base_indices and self.ground_index is not None:
            self.termination_contact = np.array(
                [[idx, self.ground_index] for idx in base_indices],
                dtype=np.uint32,
            )
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            if not base_indices:
                print("Warning: No base contacts configured for termination")
    
    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 保存上一步的关节速度(用于计算加速度)
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        
        state.info["last_actions"] = state.info["current_actions"]
        
        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions + 
                (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )
        
        state.info["current_actions"] = state.info["filtered_actions"]

        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        
        return state
    
    def _compute_torques(self, actions, data):
        """计算PD控制力矩(VBot使用motor执行器，需要力矩控制)"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = 80.0   # 位置增益
        kv = 6.0    # 速度增益
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel
        
        # 限制力矩范围(与XML中的forcerange一致)
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)  # FR, FL, RR, RL
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        return torques
    
    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角(朝向)"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading
    
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        # Ensure base quaternion is valid (normalized). If invalid, reset to identity quaternion.
        try:
            bstart = self._base_quat_start
            bend = self._base_quat_end
            if bend <= all_dof_pos.shape[1]:
                quats = all_dof_pos[:, bstart:bend]
                norms = np.linalg.norm(quats, axis=1)
                for i in range(num_envs):
                    if norms[i] > 1e-6:
                        all_dof_pos[i, bstart:bend] = quats[i] / norms[i]
                    else:
                        all_dof_pos[i, bstart:bend] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        except Exception:
            # If indexing fails for any reason, continue and let downstream checks handle it
            pass

        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置(使用DOF控制freejoint，不影响物理)"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            # 算箭头高度 = 机器人当前高度 + 偏移
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            
            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
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
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        """
        data = state.data
        cfg = self._cfg
        
        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]  # 世界坐标系线速度
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 导航目标
        pose_commands = state.info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        # 计算位置误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 计算朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 达到判定(只看位置，与奖励计算保持一致)
        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只要到达位置即可
        
        # 计算期望速度命令(与平地navigation一致，简单P控制器)
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 角速度命令：跟踪运动方向(从当前位置指向目标)
        # 与vbot_np保持一致的增益和上限，确保转向足够快
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)  # 增益和上限与vbot_np一致
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]
        
        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)
        
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
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差(保留)
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)  # 54维
        
        # 更新目标标记和箭头
        self._update_target_marker(data, pose_commands)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 计算奖励
        reward = self._compute_reward(data, state.info, velocity_commands)
        
        # 计算终止条件
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated
        
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        
        return state
    
    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        """
        重写终止条件，与locomotion stairs完全一致
        """
        data = state.data
        
        # 1. 基座接触地面终止(使用传感器)
        try:
            base_contact_value = self._model.get_sensor_value("base_contact", data)
            if base_contact_value.ndim == 0:
                base_contact = np.array([base_contact_value > 0.01], dtype=bool)
            elif base_contact_value.shape[0] != self._num_envs:
                base_contact = np.full(self._num_envs, base_contact_value.flatten()[0] > 0.01, dtype=bool)
            else:
                base_contact = (base_contact_value > 0.01).flatten()[:self._num_envs]
        except Exception as e:
            print(f"[Warning] 无法读取base_contact传感器: {e}")
            base_contact = np.zeros(self._num_envs, dtype=bool)
        
        # 2. 姿态检测终止 (摔倒检测)
        try:
            root_pos, root_quat, _ = self._extract_root_state(data)
            # 使用 sensor.feet 或类似获取足部接触，这里简化为只用姿态
            # 由于 _detect_fall 需要 foot_contacts (来自 _compute_reward 上下文或 sensor)，
            # 这里为避免复杂依赖，如果 _detect_fall 所需信息不易得，可以简化为只检测角度
            
            # 使用简化的姿态检测
            qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
            pitch = np.arcsin(sinp)
            
            # 阈值与 dog_fall_frames 逻辑保持一致 (45度)
            attitude_fail = (np.abs(roll) > self.fall_threshold_roll_pitch) | (np.abs(pitch) > self.fall_threshold_roll_pitch)
        except Exception:
            attitude_fail = np.zeros(self._num_envs, dtype=bool)

        terminated = base_contact | attitude_fail
        
        return state.replace(terminated=terminated)
    
    def _detect_fall(self, root_quat: np.ndarray, foot_contacts: np.ndarray) -> np.ndarray:
        """
        检测机器狗是否摔倒
        
        返回：摔倒标志数组 [num_envs]
        """
        # 从四元数计算 Roll 和 Pitch
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        
        # Roll (绕X轴旋转)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (绕Y轴旋转)
        sinp = 2 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # 判定摔倒：Roll或Pitch超过阈值，或足部接触过低(长时间悬空)
        exceeded_angle = (np.abs(roll) > self.fall_threshold_roll_pitch) | (np.abs(pitch) > self.fall_threshold_roll_pitch)
        
        # 更新连续悬空帧数，按 runtime batch size 处理
        n = root_quat.shape[0]
        if foot_contacts is not None and foot_contacts.ndim > 1 and foot_contacts.shape[0] == n:
            # 正确维度: [batch, 4]
            all_feet_low_contact = np.all(foot_contacts < self.fall_contact_threshold, axis=1)
            # ensure arrays are large enough
            self._ensure_dog_arrays(n)
            self.dog_fall_frames = np.where(all_feet_low_contact, self.dog_fall_frames + 1, 0)
        else:
            # 传感器维度不匹配，重置计数为长度 n
            self._ensure_dog_arrays(n)
            self.dog_fall_frames[:] = 0
            all_feet_low_contact = np.zeros(n, dtype=bool)

        long_time_airborne = self.dog_fall_frames > self.fall_frames_threshold

        return exceeded_angle | long_time_airborne
    
    def _detect_out_of_bounds(self, robot_pos: np.ndarray) -> np.ndarray:
        """
        检测机器狗是否越界
        
        返回：越界标志数组 [num_envs]
        """
        # 计算机器狗到圆心的距离
        delta = robot_pos[:, :2] - self.arena_center
        distance_from_center = np.linalg.norm(delta, axis=1)
        
        # 越界条件：距离超过边界半径
        return distance_from_center > self.boundary_radius

    def _ensure_dog_arrays(self, n: int):
        """Ensure per-environment dog_* arrays have length at least n.

        If current arrays are shorter, extend while preserving existing values.
        """
        # helper to resize a 1D numpy array while preserving values
        def _resize(arr, new_len, dtype):
            if arr is None:
                return np.zeros(new_len, dtype=dtype)
            cur = arr.shape[0]
            if cur >= new_len:
                return arr
            new = np.zeros(new_len, dtype=dtype)
            new[:cur] = arr
            return new

        # initialize or extend arrays
        self.dog_fall_frames = _resize(getattr(self, "dog_fall_frames", None), n, np.int32)
        self.dog_penalty_flags = _resize(getattr(self, "dog_penalty_flags", None), n, bool)
        self.dog_scores = _resize(getattr(self, "dog_scores", None), n, np.float32)
        self.dog_stage = _resize(getattr(self, "dog_stage", None), n, np.int32)
        self.dog_triggered_a = _resize(getattr(self, "dog_triggered_a", None), n, bool)
        self.dog_triggered_b = _resize(getattr(self, "dog_triggered_b", None), n, bool)
    
    def _check_trigger_points(self, robot_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        检测机器狗是否触发触发点 A(内圈)和 B(圆心)
        
        返回：(触发点A, 触发点B) 布尔数组
        """
        trigger_threshold = 0.3  # 触发半径 0.3 米
        
        # 到触发点A的距离
        dist_to_a = np.linalg.norm(robot_pos[:, :2] - self.target_point_a, axis=1)
        triggered_a = dist_to_a < trigger_threshold
        
        # 到触发点B(圆心)的距离
        dist_to_b = np.linalg.norm(robot_pos[:, :2] - self.target_point_b, axis=1)
        triggered_b = dist_to_b < trigger_threshold
        
        return triggered_a, triggered_b
    
    def _update_dog_scores(self, robot_pos: np.ndarray, root_quat: np.ndarray, foot_contacts: np.ndarray):
        """
        更新每只机器狗的计分和状态
        
        参数：
        - robot_pos: 机器狗的XYZ位置 [num_envs, 3]
        - root_quat: 根节点四元数 [num_envs, 4]
        - foot_contacts: 足部接触值 [num_envs, 4]
        """
        # runtime batch size
        n = robot_pos.shape[0]

        # ensure per-env arrays large enough for this batch
        self._ensure_dog_arrays(n)

        # 检测摔倒
        fallen = self._detect_fall(root_quat, foot_contacts)

        # 检测越界
        out_of_bounds = self._detect_out_of_bounds(robot_pos)

        # 检测触发点
        triggered_a, triggered_b = self._check_trigger_points(robot_pos)

        # 更新得分和状态
        for i in range(n):
            # 检查是否受到惩罚
            if fallen[i] or out_of_bounds[i]:
                self.dog_penalty_flags[i] = True
                self.dog_scores[i] = 0.0
                self.dog_stage[i] = 0
                continue

            # 如果已受惩罚，不再计分
            if self.dog_penalty_flags[i]:
                continue

            # 更新阶段和计分
            # 阶段 0 -> 1：到达内圈(+1 分)
            if self.dog_stage[i] == 0 and triggered_a[i]:
                self.dog_stage[i] = 1
                self.dog_triggered_a[i] = True
                self.dog_scores[i] += 1.0

            # 阶段 1 -> 2：到达圆心(+1 分)
            if self.dog_stage[i] == 1 and triggered_b[i]:
                self.dog_stage[i] = 2
                self.dog_triggered_b[i] = True
                self.dog_scores[i] += 1.0

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """
        改进的导航任务奖励 - 混合密集/稀疏奖励
        """
        # ===== 1. 获取状态 =====
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)

        # defensive checks: ensure info has expected keys and correct shapes
        num_envs = root_pos.shape[0]
        if "pose_commands" not in info:
            raise KeyError("info must contain 'pose_commands' with shape [num_envs, 3]")
        # ensure action history fields exist
        info.setdefault("last_actions", np.zeros((num_envs, self._num_action), dtype=np.float32))
        info.setdefault("current_actions", np.zeros((num_envs, self._num_action), dtype=np.float32))
        
        # ===== 2. 计算目标相关信息 =====
        pose_commands = info["pose_commands"]
        robot_position = root_pos[:, :2]
        target_position = pose_commands[:, :2]
        
        # 当前距离
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 距离进度（相对于起始距离）
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        else:
            info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        
        # 到达判定
        reached = distance_to_target < 0.3
        
        # ===== 3. 密集奖励：距离进步 (Distance Progress) =====
        # 总距离6米 -> 期望贡献 ~9分 (6.0 * 1.5)
        # 修正系数：2.0 -> 1.5
        distance_progress = info.get("last_distance", distance_to_target) - distance_to_target
        info["last_distance"] = distance_to_target.copy()
        
        # 限制单步奖励防止瞬间跳跃
        progress_reward = np.clip(distance_progress * 1.5, -0.5, 0.5)
        
        # ===== 4. 稀疏奖励：到达目标 (Arrival Bonus) =====
        # Only once per episode
        if "ever_reached" not in info:
            info["ever_reached"] = np.zeros(num_envs, dtype=bool)
        ever_reached = info["ever_reached"]
        arrival_bonus = np.where(reached & (~ever_reached), 10.0, 0.0)
        info["ever_reached"] = ever_reached | reached
        
        # ===== 5. 速度/姿态惩罚 (Penalty Only) =====
        # 移除正向速度奖励，只保留惩罚
        # 仅当速度*误差*很大时才扣分，而非给分
        
        # 姿态稳定性 (Orientation Penalty)
        projected_gravity = self._compute_projected_gravity(root_quat)
        # xy分量：对应roll/pitch的倾斜
        orientation_penalty = np.sum(np.square(projected_gravity[:, :2]), axis=1) * 0.1
        
        # 动作平滑惩罚
        action_rate_penalty = np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1) * 0.005
        
        # 能量惩罚
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1) * 0.00002
        
        # ===== 6. 组合奖励 =====
        # 目标: 如果完美运行，Total = ~9 (Progress) + 10 (Arrival) - minimal_penalty ~= 19.0
        # 范围: 0 ~ 20.0 (理想状态)
        
        reward = (
            progress_reward             # 距离势能
            + arrival_bonus             # 到达奖励
            - orientation_penalty       # 姿态惩罚
            - action_rate_penalty       # 动作平滑
            - torque_penalty            # 能量效率
        )
        
        # ===== 7. 失败惩罚 (Terminal Penalty) =====
        # 如果当前步触发了 terminated（意味着摔倒），给予一次性惩罚
        # 注意：这里需要再次调用 terminated 计算逻辑，或者假设外部会处理
        # 简单起见，如果 orientation_penalty 极大 (>0.5 对应倾斜 >~45度)，给额外惩罚
        reward = np.where(orientation_penalty > 0.5, reward - 10.0, reward)
        
        # 防止数值不稳定传播：替换 NaN/inf 并返回 float32
        reward = np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        return reward


    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg: VBotSection001EnvCfg = self._cfg
        num_envs = data.shape[0]
        
        # ===== 采用 AnymalC 方式：使用固定初始位置 =====
        # 所有环境使用相同的起始位置(从配置读取)
        robot_init_pos = np.tile(cfg.init_state.pos, (num_envs, 1))
        
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        
        # 设置 base 的 XYZ位置(DOF 3-5)
        dof_pos[:, 3:6] = robot_init_pos
        
        # 目标位置设定(从cfg.commands.pose_command_range采样)
        cmd_range = cfg.commands.pose_command_range
        if len(cmd_range) != 6:
            raise ValueError("commands.pose_command_range must have 6 values: dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max")

        dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max = cmd_range
        sampled = np.random.uniform(
            low=np.array([dx_min, dy_min, yaw_min], dtype=np.float32),
            high=np.array([dx_max, dy_max, yaw_max], dtype=np.float32),
            size=(num_envs, 3),
        )
        target_positions = robot_init_pos[:, :2] + sampled[:, :2]
        target_headings = sampled[:, 2:3]
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
        
        # 归一化base的四元数(DOF 6-9)
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            # 归一化箭头的四元数(如果箭头body存在)
            if self._robot_arrow_body is not None:
                robot_arrow_quat = dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                
                desired_arrow_quat = dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end]
                quat_norm = np.linalg.norm(desired_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = desired_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        
        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)
        
        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        
        # 关节状态
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 计算速度命令
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只看位置
        
        # 计算期望速度
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # ===== 与reset一致：角速度跟踪运动方向 =====
        # 计算期望的运动方向(从update_state中复制)
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        
        # 添加死区，与update_state保持一致
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        
        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

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
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差(保留)
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
            ],
            axis=-1,
        )
        print(f"obs.shape:{obs.shape}")
        assert obs.shape == (num_envs, 54)  # 54维
        
        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),
        }
        
        return obs, info

    