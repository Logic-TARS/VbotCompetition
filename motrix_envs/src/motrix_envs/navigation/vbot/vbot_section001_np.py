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

        # PD增益（从cfg获取或使用默认值）
        self.kps = np.ones(self._num_action, dtype=np.float32) * getattr(cfg.control_config, 'stiffness', 80.0)
        self.kds = np.ones(self._num_action, dtype=np.float32) * getattr(cfg.control_config, 'damping', 4.0)
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)

        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )

        # 设置默认关节角度 + 收集hip/calf索引
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

        self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = 0.3

        self.random_push_scale = getattr(cfg, "random_push_scale", 0.3)

        # ===== 姿态/高度参数 =====
        # VBot站立高度约0.30m, 初始高度0.5m(含下落)
        self.target_base_height = 0.30  # 目标站立高度
        self.min_base_height = 0.18     # 低于此高度判定趴下/瘫倒

        # --- 初始化摔倒/越界/计分相关变量
        self.fall_threshold_roll_pitch = getattr(cfg, "fall_threshold_roll_pitch", np.deg2rad(30.0))  # 30度，更严格
        self.fall_contact_threshold = getattr(cfg, "fall_contact_threshold", 0.05)
        self.fall_frames_threshold = getattr(cfg, "fall_frames_threshold", 10)

        # 赛场/边界/触发点
        self.arena_center = np.array(getattr(cfg, "arena_center", np.array([0.0, 0.0], dtype=np.float32)), dtype=np.float32)
        self.boundary_radius = getattr(cfg, "boundary_radius", 10.0)
        self.target_point_a = np.array(getattr(cfg, "target_point_a", self.arena_center), dtype=np.float32)
        self.target_point_b = np.array(getattr(cfg, "target_point_b", self.arena_center), dtype=np.float32)

        # 计分与状态数组
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

        # 足部接触检查别名（与 vbot_nav_flat_np 一致）
        self.foot_check = self.foot_contact_check
        self.foot_check_num = self.num_foot_check
        self.termination_check = self.termination_contact
        self.num_check = self.num_termination_check


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
        """初始化足部接触检测（与 vbot_nav_flat_np 一致）"""
        cfg = self._cfg
        foot_indices = []
        for foot_name in cfg.asset.foot_names:
            try:
                idx = self._model.get_geom_index(foot_name)
                if idx is not None:
                    foot_indices.append(idx)
            except Exception:
                pass

        if foot_indices and self.ground_index is not None:
            self.foot_contact_check = np.array(
                [[idx, self.ground_index] for idx in foot_indices], dtype=np.uint32
            )
            self.num_foot_check = self.foot_contact_check.shape[0]
        else:
            self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
            self.num_foot_check = 0
            if not foot_indices:
                print("Warning: No foot geoms found for contact detection")

    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        cfg = self._cfg
        # 尝试多个可能的地面geom名称
        self.ground_index = None
        for ground_name in [cfg.asset.ground_name, "C_B_BODY", "ground_plane"]:
            try:
                idx = self._model.get_geom_index(ground_name)
                if idx is not None:
                    self.ground_index = idx
                    break
            except Exception:
                continue
        if self.ground_index is None:
            print("Warning: No ground geom found for contact detection")

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
        """计算PD控制力矩 (与 walk_np/vbot_nav_flat_np 一致)"""
        actions_scaled = actions * self._cfg.control_config.action_scale
        torques = self.kps * (
            actions_scaled + self.default_angles - self.get_dof_pos(data)
        ) - self.kds * self.get_dof_vel(data)
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

    def update_feet_air_time(self, info: dict):
        """更新足部空中时间 (与 vbot_nav_flat_np 一致)"""
        feet_air_time = info["feet_air_time"]
        feet_air_time += self._cfg.ctrl_dt
        feet_air_time *= ~info["contacts"]
        return feet_air_time

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        结构与 vbot_nav_flat_np 对齐：observation → foot contacts → termination → reward
        """
        data = state.data
        cfg = self._cfg

        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # 传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 导航目标
        pose_commands = state.info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        # 计算位置/朝向误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)

        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold

        # 计算期望速度命令
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        # 存储导航状态到info
        state.info["velocity_commands"] = velocity_commands
        state.info["reached_all"] = reached_all
        state.info["position_error"] = position_error
        state.info["distance_to_target"] = distance_to_target
        state.info["heading_diff"] = heading_diff
        state.info["desired_vel_xy"] = desired_vel_xy

        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        state.info["stop_ready"] = stop_ready

        # ===== 足部接触检测 (关键新增！与 vbot_nav_flat_np 一致) =====
        cquerys = self._model.get_contact_query(data)
        if self.foot_check_num > 0:
            foot_contact = cquerys.is_colliding(self.foot_check)
            state.info["contacts"] = foot_contact.reshape((self._num_envs, self.foot_check_num))
        else:
            state.info["contacts"] = np.zeros((self._num_envs, max(self.foot_check_num, 4)), dtype=bool)
        state.info["feet_air_time"] = self.update_feet_air_time(state.info)

        # 构建观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
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
                heading_error_normalized[:, np.newaxis],  # 1
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # 更新可视化
        self._update_target_marker(data, pose_commands)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        # 更新计分系统
        try:
            self._update_dog_scores(root_pos, root_quat, state.info.get("contacts"))
        except Exception:
            pass

        # ===== 计算终止条件 (先于奖励，以便奖励函数可以用到 terminated) =====
        terminated = self._compute_terminated(state, data, root_pos, root_quat, cquerys)

        # ===== 计算奖励 =====
        reward = self._compute_reward(data, state.info, velocity_commands, terminated)

        # 更新步数
        state.info["steps"] = state.info.get("steps", np.zeros(self._num_envs, dtype=np.int32)) + 1

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        return state

    def _compute_terminated(self, state: NpEnvState, data, root_pos, root_quat, cquerys) -> np.ndarray:
        """
        严格终止条件：
        1. 几何体碰撞检测（躯干/大腿/小腿触地）
        2. 姿态倾斜检测（tilt > 30度）
        3. 基座高度过低（< 0.18m = 趴下/瘫倒）
        4. 越界检测
        5. DOF速度异常
        """
        num_envs = data.shape[0]

        # 保护期：前20帧不终止（缩短保护期，原来60帧太长）
        GRACE_PERIOD = 20
        steps = state.info.get("steps", np.zeros(num_envs, dtype=np.int32))
        if isinstance(steps, np.ndarray):
            in_grace_period = steps < GRACE_PERIOD
        else:
            in_grace_period = np.full(num_envs, steps < GRACE_PERIOD, dtype=bool)

        # 1. 几何体碰撞终止（与 vbot_nav_flat_np 一致）
        if self.num_check > 0:
            termination_colliding = cquerys.is_colliding(self.termination_check)
            termination_colliding = termination_colliding.reshape((num_envs, self.num_check))
            body_contact = termination_colliding.any(axis=1)
        else:
            body_contact = np.zeros(num_envs, dtype=bool)

        # 2. 姿态倾斜终止（使用投影重力向量，与 vbot_nav_flat_np 一致）
        local_gravity = Quaternion.rotate_inverse(root_quat, self.gravity_vec)
        gxy = np.linalg.norm(local_gravity[:, :2], axis=1)
        gz = local_gravity[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        attitude_fail = tilt_angle > np.deg2rad(30)  # 30度，比原来的45度更严格

        # 3. ★关键新增：基座高度过低终止——直接解决"趴窝"问题★
        height = root_pos[:, 2]
        height_too_low = height < self.min_base_height  # < 0.18m → 已经趴下

        # 4. 越界检测
        out_of_bounds = self._detect_out_of_bounds(root_pos)

        # 5. DOF速度异常（与 vbot_nav_flat_np 一致）
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > 100.0
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)

        # 组合终止条件
        terminated = body_contact | attitude_fail | height_too_low | out_of_bounds | vel_overflow | vel_extreme

        # 保护期内忽略终止（但仍终止极端情况如NaN）
        terminated = np.where(in_grace_period, vel_extreme, terminated)

        return terminated

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

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray, terminated: np.ndarray) -> np.ndarray:
        """
        修复版奖励函数 — 解决"站原地不动"问题

        核心问题诊断：旧版站立奖励(~2.5/step) > 移动奖励(~1.8/step)，
        agent 理性选择站着不动以最大化回报。

        修复原则：
        1. 大幅降低站立/高度奖励，只作为基线保障（不能主导总奖励）
        2. 大幅提高导航进步奖励（必须是最大的密集奖励信号）
        3. 速度跟踪奖励放大，让"朝目标移动"比"静止"收益高
        4. 终止惩罚适度（-10），避免agent极度保守不敢动
        5. 添加静止惩罚：站着不动要扣分
        """
        # ===== 1. 获取状态 =====
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)

        num_envs = root_pos.shape[0]
        info.setdefault("last_actions", np.zeros((num_envs, self._num_action), dtype=np.float32))
        info.setdefault("current_actions", np.zeros((num_envs, self._num_action), dtype=np.float32))

        # ===== 2. 基座高度奖励（降低权重，只作为基线保障） =====
        height = root_pos[:, 2]
        height_error = np.abs(height - self.target_base_height)
        # 站立奖励降低到 0.5（旧版 2.0），不再主导总奖励
        base_height_reward = np.exp(-height_error / 0.08) * 0.5
        # 额外惩罚低于安全高度（保留）
        height_penalty = np.where(height < self.min_base_height, 2.0, 0.0)

        # ===== 3. 存活奖励（小幅度，不能高于导航奖励） =====
        is_standing = height > (self.min_base_height + 0.02)  # > 0.20m
        alive_bonus = np.where(is_standing, 0.1, 0.0)  # 降低到 0.1（旧版 0.5）

        # ===== 4. 姿态惩罚：保持水平 =====
        projected_gravity = self._compute_projected_gravity(root_quat)
        orientation_penalty = np.sum(np.square(projected_gravity[:, :2]), axis=1) * 0.5

        # ===== 5. 导航目标相关（大幅提升！这才是核心信号） =====
        pose_commands = info["pose_commands"]
        robot_position = root_pos[:, :2]
        target_position = pose_commands[:, :2]
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # 朝向对齐（提升权重）
        heading_to_target = np.arctan2(position_error[:, 1], position_error[:, 0])
        robot_heading = self._get_heading_from_quat(root_quat)
        heading_alignment = heading_to_target - robot_heading
        heading_alignment = np.where(heading_alignment > np.pi, heading_alignment - 2*np.pi, heading_alignment)
        heading_alignment = np.where(heading_alignment < -np.pi, heading_alignment + 2*np.pi, heading_alignment)
        heading_alignment_reward = np.exp(-np.abs(heading_alignment) / 0.5) * 0.5

        # 距离进步（★核心导航信号★ — 大幅提升 clip 范围和系数）
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        else:
            info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)

        reached = distance_to_target < 0.3

        distance_progress = info.get("last_distance", distance_to_target) - distance_to_target
        info["last_distance"] = distance_to_target.copy()
        # 提升系数 2.0→5.0，clip 范围 0.5→2.0（让接近目标的回报远超站立）
        progress_reward = np.clip(distance_progress * 5.0, -1.0, 2.0)

        # 距离缩短持续奖励：距离越近奖励越高（密集信号）
        distance_shaping = np.exp(-distance_to_target / 3.0) * 1.0

        # 两阶段稀疏奖励（提升到 10.0，让到达目标更有吸引力）
        triggered_a, triggered_b = self._check_trigger_points(root_pos)
        if "triggered_a" not in info:
            info["triggered_a"] = np.zeros(num_envs, dtype=bool)
        first_trigger_a = triggered_a & (~info["triggered_a"])
        info["triggered_a"] = info["triggered_a"] | triggered_a
        stage1_bonus = np.where(first_trigger_a, 10.0, 0.0)

        if "triggered_b" not in info:
            info["triggered_b"] = np.zeros(num_envs, dtype=bool)
        first_trigger_b = triggered_b & (~info["triggered_b"])
        info["triggered_b"] = info["triggered_b"] | triggered_b
        stage2_bonus = np.where(first_trigger_b, 10.0, 0.0)

        # ===== 6. 速度跟踪奖励（提升，鼓励移动） =====
        lin_vel_error = np.sum(np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1)
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25) * 1.5  # 1.0 → 1.5

        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25) * 0.5

        # ===== 7. 步态奖励：feet_air_time =====
        feet_air_time_reward = 0.0
        if self.foot_check_num > 0 and "contacts" in info and "feet_air_time" in info:
            feet_air_time = info["feet_air_time"]
            contacts = info["contacts"]
            first_contact = (feet_air_time > 0.0) * contacts
            rew_airTime = np.sum((feet_air_time - 0.5) * first_contact, axis=1)
            rew_airTime *= np.linalg.norm(velocity_commands[:, :2], axis=1) > 0.1
            feet_air_time_reward = rew_airTime * 1.0

        # ===== 8. ★新增：静止惩罚★ — 站着不动要扣分 =====
        # 当有运动命令但base速度很低时，施加惩罚
        cmd_magnitude = np.linalg.norm(velocity_commands[:, :2], axis=1)
        base_speed = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        # 有命令(cmd > 0.1) 但几乎不动(speed < 0.05): 惩罚
        should_move = cmd_magnitude > 0.1
        not_moving = base_speed < 0.05
        standstill_penalty = np.where(should_move & not_moving, 1.0, 0.0)

        # ===== 9. 运动质量惩罚 =====
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2]) * 0.5
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1) * 0.02
        action_rate_penalty = np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1) * 0.0005
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1) * 0.000005
        last_dof_vel = info.get("last_dof_vel", self.get_dof_vel(data))
        dof_acc = (last_dof_vel - self.get_dof_vel(data)) / self._cfg.ctrl_dt
        dof_acc_penalty = np.sum(np.square(dof_acc), axis=1) * 2.5e-7

        # ===== 10. 组合奖励 =====
        # 设计平衡：站着不动 ≈ 0.6 + 0.1 - 1.0 = -0.3/step（净亏损！）
        #           朝目标走 ≈ 0.5 + 0.1 + 1.0 + 1.0 + 1.5 + 0.5 = ~4.6/step（远高于静止）
        reward = (
            # 正向信号 - 基础姿态（降低，仅保障基线）
            base_height_reward            # 高度奖励 (max ~0.5/step) — 降低4倍
            + alive_bonus                 # 存活奖励 (0.1/step) — 降低5倍
            # 正向信号 - 导航（核心！必须占主导地位）
            + progress_reward             # 距离进步 (max ~2.0/step) ★大幅提升★
            + distance_shaping            # 距离塑形 (max ~1.0/step) ★新增★
            + stage1_bonus                # 内圈奖励 (10.0一次) ★翻倍★
            + stage2_bonus                # 圆心奖励 (10.0一次) ★翻倍★
            + heading_alignment_reward    # 朝向对齐 (max ~0.5)
            + tracking_lin_vel            # 线速度跟踪 (max ~1.5) ★提升★
            + tracking_ang_vel            # 角速度跟踪
            + feet_air_time_reward        # 步态奖励
            # 负向信号
            - orientation_penalty         # 姿态惩罚
            - height_penalty              # 低高度惩罚
            - standstill_penalty          # ★新增：静止惩罚★
            - lin_vel_z_penalty           # Z轴速度惩罚
            - ang_vel_xy_penalty          # XY角速度惩罚
            - action_rate_penalty         # 动作平滑
            - torque_penalty              # 能量效率
            - dof_acc_penalty             # 关节加速度惩罚
        )

        # ===== 11. 终止惩罚（适度降低，避免agent过度保守不敢动） =====
        terminal_penalty = np.where(terminated, -10.0, 0.0)  # -200 → -10
        reward = reward + terminal_penalty

        reward = np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        return reward


    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg: VBotSection001EnvCfg = self._cfg
        num_envs = data.shape[0]

        # ===== 极坐标随机生成在整个平台范围内 =====
        # 使用sqrt保证面积均匀分布，确保与目标(圆心)最小距离由配置决定
        min_spawn_distance = cfg.min_spawn_distance  # 至少距目标(圆心)设定距离(默认2.0m)
        max_spawn_radius = cfg.boundary_radius - 0.5  # 3.0m，留足安全距离防止出生在地面外
        robot_init_xy = np.zeros((num_envs, 2), dtype=np.float32)
        for i in range(num_envs):
            while True:
                theta = np.random.uniform(0, 2 * np.pi)
                radius = max_spawn_radius * np.sqrt(np.random.uniform(0, 1))
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                if np.sqrt(x**2 + y**2) >= min_spawn_distance:
                    robot_init_xy[i] = [x, y]
                    break

        # 添加圆心偏移（如果存在）
        robot_init_xy += np.array(cfg.arena_center, dtype=np.float32)

        # 构造完整的XYZ位置（高度使用配置，避免空中自由落体）
        spawn_height = float(cfg.init_state.pos[2])
        robot_init_pos = np.column_stack([robot_init_xy, np.full(num_envs, spawn_height, dtype=np.float32)])

        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))

        if self.random_push_scale > 0.0 and dof_vel.shape[1] >= 5:
            push_xy = np.random.uniform(-1.0, 1.0, size=(num_envs, 2)).astype(np.float32)
            push_xy *= self.random_push_scale
            dof_vel[:, 3:5] = push_xy

        # 设置 base 的 XYZ位置(DOF 3-5)
        dof_pos[:, 3:6] = robot_init_pos

        # 目标位置设定：固定为圆心(比赛规则)
        # 所有机器狗都以圆心为目标，不使用随机偏移
        arena_center = np.array(cfg.arena_center, dtype=np.float32)
        target_positions = np.tile(arena_center, (num_envs, 1))
        target_headings = np.zeros((num_envs, 1), dtype=np.float32)
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # 归一化base的四元数(DOF 6-9)，并设置随机yaw
        for env_idx in range(num_envs):
            # 随机yaw角 [0, 2π)，roll=0, pitch=0 确保不会侧翻
            random_yaw = np.random.uniform(0, 2 * np.pi)
            random_quat = self._euler_to_quat(0.0, 0.0, random_yaw)
            dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = random_quat
            
            # 验证基座四元数的有效性（非零且归一化）
            base_quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            quat_norm = np.linalg.norm(base_quat)
            if quat_norm < 1e-6:
                # 无效四元数，重置为单位四元数 [0, 0, 0, 1]
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            elif abs(quat_norm - 1.0) > 1e-3:
                # 归一化非单位四元数
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = base_quat / quat_norm

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
        assert obs.shape == (num_envs, 54)  # 54维
        
        # 防止 NaN/Inf 传播到神经网络导致 CUDA 崩溃
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "last_distance": distance_to_target.copy(),
            "triggered_a": np.zeros(num_envs, dtype=bool),
            "triggered_b": np.zeros(num_envs, dtype=bool),
            # ★新增：足部接触和空中时间（与 vbot_nav_flat_np 对齐）★
            "feet_air_time": np.zeros((num_envs, max(self.foot_check_num, 4)), dtype=np.float32),
            "contacts": np.zeros((num_envs, max(self.foot_check_num, 4)), dtype=bool),
        }

        return obs, info

