import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection011EnvCfg


def generate_repeating_array(num_period, num_reset, period_counter):
    """
    生成重复数组，用于在固定位置中循环选择
    num_period: 位置总数
    num_reset: 需要重置的环境数
    period_counter: 当前计数器
    """
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("vbot_navigation_section011", "np")
class VBotSection011Env(NpEnv):
    """
    VBot在Section011地形上的导航任务
    继承自NpEnv，使用VBotSection011EnvCfg配置
    """
    _cfg: VBotSection011EnvCfg
    
    def __init__(self, cfg: VBotSection011EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 观测空间：75维（54 base + 21 competition-specific）
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
        
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
        
        # 初始位置生成参数：从配置文件读取
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)  # 从配置读取
        self.spawn_range = 0.1  # 随机生成范围：±0.1m（0.2m×0.2m区域）
    
        # 导航统计计数器
        self.navigation_stats_step = 0
    
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
        
        # 竞赛区域坐标 - 从配置文件读取
        competition_cfg = getattr(cfg, 'competition', None)
        if competition_cfg is None:
            # 如果配置中没有competition部分，使用默认值
            self.start_zone_center = np.array([0.0, -2.4], dtype=np.float32)
            self.start_zone_radius = 1.0
            self.smiley_positions = np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0]], dtype=np.float32)
            self.smiley_radius = 0.5
            self.hongbao_positions = np.array([[1.0, 2.5], [1.0, 7.5], [1.0, 12.5]], dtype=np.float32)
            self.hongbao_radius = 0.5
            self.finish_zone_center = np.array([0.0, 10.2], dtype=np.float32)
            self.finish_zone_radius = 1.0
            self.boundary_x_min = -20.0
            self.boundary_x_max = 20.0
            self.boundary_y_min = -10.0
            self.boundary_y_max = 20.0
            self.celebration_duration = 1.0
            self.celebration_movement_threshold = 0.1
        else:
            # 从配置读取
            self.start_zone_center = np.array(competition_cfg.start_zone_center, dtype=np.float32)
            self.start_zone_radius = competition_cfg.start_zone_radius
            self.smiley_positions = np.array(competition_cfg.smiley_positions, dtype=np.float32)
            self.smiley_radius = competition_cfg.smiley_radius
            self.hongbao_positions = np.array(competition_cfg.hongbao_positions, dtype=np.float32)
            self.hongbao_radius = competition_cfg.hongbao_radius
            self.finish_zone_center = np.array(competition_cfg.finish_zone_center, dtype=np.float32)
            self.finish_zone_radius = competition_cfg.finish_zone_radius
            self.boundary_x_min = competition_cfg.boundary_x_min
            self.boundary_x_max = competition_cfg.boundary_x_max
            self.boundary_y_min = competition_cfg.boundary_y_min
            self.boundary_y_max = competition_cfg.boundary_y_max
            self.celebration_duration = competition_cfg.celebration_duration
            self.celebration_movement_threshold = competition_cfg.celebration_movement_threshold
        
        # 摔倒检测阈值
        self.fall_threshold_roll_pitch = np.deg2rad(45.0)
        
        # 初始化触发标志
        n = self._num_envs
        self.smiley_triggered = np.zeros((n, 3), dtype=bool)  # 3个emoji
        self.hongbao_triggered = np.zeros((n, 3), dtype=bool)  # 3个红包
        self.finish_triggered = np.zeros(n, dtype=bool)
        
        # 庆祝追踪
        self.celebration_start_time = np.full(n, -1.0, dtype=np.float32)
        self.celebration_completed = np.zeros(n, dtype=bool)
    
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
    
    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        self._init_termination_contact()
        self._init_foot_contact()
    
    def _init_termination_contact(self):
        """初始化终止接触检测：基座geom与地面geom的碰撞"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on
        
        # 获取所有地面geom（遍历所有geom，找到包含ground_subtree名称的）
        ground_geoms = []
        ground_prefix = self._cfg.asset.ground_subtree  # "0ground_root"
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))
        
        # if len(ground_geoms) == 0:
        #     print(f"[Warning] 未找到以 '{ground_prefix}' 开头的地面geom！")
        #     self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
        #     self.num_termination_check = 0
        #     return
        
        # 构建碰撞对：每个基座geom × 每个地面geom
        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except Exception as e:
                print(f"[Warning] 无法找到基座geom '{base_geom_name}': {e}")
        
        if len(termination_contact_list) > 0:
            self.termination_contact = np.array(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = len(termination_contact_list)
            print(f"[Info] 初始化终止接触检测: {len(termination_contact_names)}个基座geom × {len(ground_geoms)}个地面geom = {self.num_termination_check}个检测对")
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("[Warning] 未找到任何终止接触geom，基座接触检测将被禁用！")
    
    def _init_foot_contact(self):
        self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
        self.num_foot_check = 4  
    
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
        # 保存上一步的关节速度（用于计算加速度）
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        
        state.info["last_actions"] = state.info.get("current_actions", np.zeros((state.data.shape[0], self._num_action), dtype=np.float32))
        
        # 保存当前动作用于奖励计算
        state.info["next_actions"] = actions
        
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
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
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
        
        # 限制力矩范围（与XML中的forcerange一致）
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
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading
    
    def _normalize_quaternions_in_dof_pos(self, all_dof_pos: np.ndarray):
        """归一化dof_pos中所有四元数，防止退化导致set_dof_pos崩溃"""
        num_envs = all_dof_pos.shape[0]
        # 基座四元数（index 6-9）
        quat_indices = [(self._base_quat_start, self._base_quat_end)]
        # 箭头四元数（如果存在）
        if self._robot_arrow_body is not None:
            quat_indices.append((self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end))
            quat_indices.append((self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end))
        
        for start, end in quat_indices:
            if end <= all_dof_pos.shape[1]:
                for env_idx in range(num_envs):
                    q = all_dof_pos[env_idx, start:end]
                    norm = np.linalg.norm(q)
                    if norm > 1e-6:
                        all_dof_pos[env_idx, start:end] = q / norm
                    else:
                        all_dof_pos[env_idx, start:end] = [0.0, 0.0, 0.0, 1.0]
        return all_dof_pos

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
        
        all_dof_pos = self._normalize_quaternions_in_dof_pos(all_dof_pos)
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
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
        
        all_dof_pos = self._normalize_quaternions_in_dof_pos(all_dof_pos)
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
        
        # 达到判定（只看位置，与奖励计算保持一致）
        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只要到达位置即可
        
        # 计算期望速度命令（与平地navigation一致，简单P控制器）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 角速度命令：跟踪运动方向（从当前位置指向目标）
        # 与vbot_np保持一致的增益和上限，确保转向足够快
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)  # 增益和上限与vbot_np一致
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
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
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差（保留）
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
                # 竞赛特定观测 (21维)
                self._get_smiley_relative_pos(data, root_pos),  # 6
                self._get_hongbao_relative_pos(data, root_pos),  # 6
                self._get_trigger_flags(state.info),  # 6
                self._get_finish_relative_pos(data, root_pos),  # 2
                self._get_finish_flag(state.info),  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 75), f"Expected obs shape (*, 75), got {obs.shape}"
        
        # 更新目标标记和箭头
        self._update_target_marker(data, pose_commands)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 更新触发状态
        self._update_trigger_states(root_pos, state.info)
        
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
        计算终止条件（支持竞赛模式）
        """
        data = state.data
        n_envs = data.shape[0]
        
        # 宽限期：前20帧不终止
        GRACE_PERIOD = 20
        steps = state.info.get("steps", np.zeros(n_envs, dtype=np.int32))
        if isinstance(steps, np.ndarray):
            in_grace_period = steps < GRACE_PERIOD
        else:
            in_grace_period = np.zeros(n_envs, dtype=bool)
        
        # 1. 基座接触地面（摔倒检测）- 使用碰撞查询（与AnymalC一致的正确做法）
        root_pos = self._body.get_pose(data)[:, :3]
        try:
            if self.num_termination_check > 0:
                # 使用几何碰撞查询（可靠）
                cquerys = self._model.get_contact_query(data)
                termination_check = cquerys.is_colliding(self.termination_contact)
                termination_check = termination_check.reshape((n_envs, self.num_termination_check))
                base_contact = termination_check.any(axis=1)
            else:
                base_contact = np.zeros(n_envs, dtype=bool)
        except Exception as e:
            base_contact = np.zeros(n_envs, dtype=bool)
        
        # 基座碰到地面即判定摔倒（碰撞查询只检测collision_middle_box和collision_head_box与地面的碰撞）
        fall_detected = base_contact
        
        # 2. 姿态失败（roll/pitch > 45度）
        try:
            root_quat = self._body.get_pose(data)[:, 3:7]
            qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
            pitch = np.arcsin(sinp)
            attitude_fail = (np.abs(roll) > self.fall_threshold_roll_pitch) | (np.abs(pitch) > self.fall_threshold_roll_pitch)
        except Exception:
            attitude_fail = np.zeros(n_envs, dtype=bool)
        
        # 3. 超界检测
        out_of_bounds = (
            (root_pos[:, 0] < self.boundary_x_min) |
            (root_pos[:, 0] > self.boundary_x_max) |
            (root_pos[:, 1] < self.boundary_y_min) |
            (root_pos[:, 1] > self.boundary_y_max)
        )
        
        # 合并终止条件（宽限期内忽略）
        terminated = (fall_detected | attitude_fail | out_of_bounds) & ~in_grace_period

        # 应用终止惩罚
        penalty = np.where(terminated, -10.0, 0.0)
        state.reward += penalty

        state.terminated = terminated
        return state
    
    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """
        导航任务奖励计算（支持竞赛模式）
        """
        cfg = self._cfg
        n_envs = data.shape[0]
        
        # 获取触发状态
        smiley_triggered = info.get("smiley_triggered", np.zeros((n_envs, 3), dtype=bool))
        hongbao_triggered = info.get("hongbao_triggered", np.zeros((n_envs, 3), dtype=bool))
        finish_triggered = info.get("finish_triggered", np.zeros(n_envs, dtype=bool))
        
        # 基础奖励
        reward = np.zeros(n_envs, dtype=np.float32)
        
        # 1. 生存奖励 (+0.01每步)
        reward += 0.01
        
        # 2. 触发奖励（仅首次）
        # Emoji奖励 (+4 each)
        for i in range(3):
            newly_triggered = smiley_triggered[:, i] & ~info.get(f"smiley_{i}_rewarded", np.zeros(n_envs, dtype=bool))
            reward += newly_triggered.astype(np.float32) * 4.0
            if f"smiley_{i}_rewarded" not in info:
                info[f"smiley_{i}_rewarded"] = np.zeros(n_envs, dtype=bool)
            info[f"smiley_{i}_rewarded"] |= newly_triggered
        
        # 红包奖励 (+2 each)
        for i in range(3):
            newly_triggered = hongbao_triggered[:, i] & ~info.get(f"hongbao_{i}_rewarded", np.zeros(n_envs, dtype=bool))
            reward += newly_triggered.astype(np.float32) * 2.0
            if f"hongbao_{i}_rewarded" not in info:
                info[f"hongbao_{i}_rewarded"] = np.zeros(n_envs, dtype=bool)
            info[f"hongbao_{i}_rewarded"] |= newly_triggered
        
        # 终点奖励 (+20)
        newly_finished = finish_triggered & ~info.get("finish_rewarded", np.zeros(n_envs, dtype=bool))
        reward += newly_finished.astype(np.float32) * 20.0
        if "finish_rewarded" not in info:
            info["finish_rewarded"] = np.zeros(n_envs, dtype=bool)
        info["finish_rewarded"] |= newly_finished
        
        # 3. 稳定性惩罚
        root_quat = self._body.get_pose(data)[:, 3:7]
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # 姿态惩罚
        orientation_penalty = (np.abs(roll) + np.abs(pitch)) * -0.05
        reward += orientation_penalty
        
        # 动作率惩罚
        last_actions = info.get("current_actions", np.zeros((n_envs, self._num_action), dtype=np.float32))
        current_actions = info.get("next_actions", last_actions)
        if "next_actions" not in info:
            info["next_actions"] = last_actions
        action_rate = np.sum(np.square(current_actions - last_actions), axis=-1)
        reward += action_rate * -0.01
        
        # 力矩惩罚（能耗）
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=-1) * -1e-5
        reward += torque_penalty
        
        return reward
    
    def _get_smiley_relative_pos(self, data: mtx.SceneData, root_pos: np.ndarray) -> np.ndarray:
        """获取相对于emoji的位置观测 (n_envs, 6)"""
        robot_pos = root_pos[:, :2]
        smiley_relative_pos = []
        for i in range(3):
            smiley_pos = self.smiley_positions[i]
            relative_pos = smiley_pos - robot_pos
            smiley_relative_pos.append(relative_pos / 5.0)  # 归一化
        return np.concatenate(smiley_relative_pos, axis=-1)
    
    def _get_hongbao_relative_pos(self, data: mtx.SceneData, root_pos: np.ndarray) -> np.ndarray:
        """获取相对于红包的位置观测 (n_envs, 6)"""
        robot_pos = root_pos[:, :2]
        hongbao_relative_pos = []
        for i in range(3):
            hongbao_pos = self.hongbao_positions[i]
            relative_pos = hongbao_pos - robot_pos
            hongbao_relative_pos.append(relative_pos / 5.0)  # 归一化
        return np.concatenate(hongbao_relative_pos, axis=-1)
    
    def _get_trigger_flags(self, info: dict) -> np.ndarray:
        """获取触发标志观测 (n_envs, 6)"""
        n_envs = info.get("smiley_triggered", np.zeros((1, 3), dtype=bool)).shape[0]
        smiley_triggered = info.get("smiley_triggered", np.zeros((n_envs, 3), dtype=bool))
        hongbao_triggered = info.get("hongbao_triggered", np.zeros((n_envs, 3), dtype=bool))
        return np.concatenate([
            smiley_triggered.astype(np.float32),
            hongbao_triggered.astype(np.float32),
        ], axis=-1)
    
    def _get_finish_relative_pos(self, data: mtx.SceneData, root_pos: np.ndarray) -> np.ndarray:
        """获取相对于终点的位置观测 (n_envs, 2)"""
        robot_pos = root_pos[:, :2]
        finish_relative_pos = (self.finish_zone_center - robot_pos) / 5.0
        return finish_relative_pos
    
    def _get_finish_flag(self, info: dict) -> np.ndarray:
        """获取终点标志观测 (n_envs, 1)"""
        n_envs = info.get("finish_triggered", np.zeros(1, dtype=bool)).shape[0]
        finish_triggered = info.get("finish_triggered", np.zeros(n_envs, dtype=bool))
        return finish_triggered.astype(np.float32)[:, np.newaxis]
    
    def _update_trigger_states(self, root_pos: np.ndarray, info: dict):
        """更新触发状态"""
        robot_pos = root_pos[:, :2]
        n_envs = robot_pos.shape[0]
        
        # 初始化触发状态
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
            finish_triggered = np.zeros(n_envs, dtype=bool)
            info["finish_triggered"] = finish_triggered
        
        # 检测emoji触发
        for i in range(3):
            smiley_pos = self.smiley_positions[i]
            dist = np.linalg.norm(robot_pos - smiley_pos, axis=-1)
            newly_triggered = (dist < self.smiley_radius) & ~smiley_triggered[:, i]
            smiley_triggered[:, i] |= newly_triggered
        
        # 检测红包触发
        for i in range(3):
            hongbao_pos = self.hongbao_positions[i]
            dist = np.linalg.norm(robot_pos - hongbao_pos, axis=-1)
            newly_triggered = (dist < self.hongbao_radius) & ~hongbao_triggered[:, i]
            hongbao_triggered[:, i] |= newly_triggered
        
        # 检测终点触发
        finish_dist = np.linalg.norm(robot_pos - self.finish_zone_center, axis=-1)
        newly_triggered_finish = (finish_dist < self.finish_zone_radius) & ~finish_triggered
        finish_triggered |= newly_triggered_finish

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg = self._cfg
        num_envs = data.shape[0]
        
        # 在高台中央小范围内随机生成位置
        # X, Y: 在spawn_center周围 ±spawn_range 范围内随机
        random_xy = np.random.uniform(
            low=-self.spawn_range,
            high=self.spawn_range,
            size=(num_envs, 2)
        )
        robot_init_xy = self.spawn_center[:2] + random_xy  # [num_envs, 2]
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)  # 使用配置的高度
        
        
        # 组合XYZ坐标
        robot_init_pos = robot_init_xy  # [num_envs, 2]
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])  # [num_envs, 3]
        
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        
        # 设置 base 的 XYZ位置（DOF 3-5）
        dof_pos[:, 3:6] = robot_init_xyz  # [x, y, z] 随机生成的位置
        
        target_offset = np.random.uniform(
            low=cfg.commands.pose_command_range[:2],
            high=cfg.commands.pose_command_range[3:5],
            size=(num_envs, 2)
        )
        target_positions = robot_init_pos + target_offset
        
        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2],
            high=cfg.commands.pose_command_range[5],
            size=(num_envs, 1)
        )
        
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
        
        # 归一化base的四元数（DOF 6-9）
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            # 归一化箭头的四元数（如果箭头body存在）
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
        # 计算期望的运动方向（从update_state中复制）
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
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 初始化info字典（在生成观测之前）
        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),  # 统一使用min_distance机制
            # 与locomotion一致的字段
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),  # 上一步关节速度
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),  # 足部接触状态
            # 竞赛特定字段
            "smiley_triggered": np.zeros((num_envs, 3), dtype=bool),
            "hongbao_triggered": np.zeros((num_envs, 3), dtype=bool),
            "finish_triggered": np.zeros(num_envs, dtype=bool),
            "celebration_start_time": np.full(num_envs, -1.0, dtype=np.float32),
            "celebration_completed": np.zeros(num_envs, dtype=bool),
            "celebration_rewarded": np.zeros(num_envs, dtype=bool),
            "finish_rewarded": np.zeros(num_envs, dtype=bool),
            "smiley_0_rewarded": np.zeros(num_envs, dtype=bool),
            "smiley_1_rewarded": np.zeros(num_envs, dtype=bool),
            "smiley_2_rewarded": np.zeros(num_envs, dtype=bool),
            "hongbao_0_rewarded": np.zeros(num_envs, dtype=bool),
            "hongbao_1_rewarded": np.zeros(num_envs, dtype=bool),
            "hongbao_2_rewarded": np.zeros(num_envs, dtype=bool),
        }
        
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
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差（保留）
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
                # 竞赛特定观测 (21维)
                self._get_smiley_relative_pos(data, root_pos),  # 6
                self._get_hongbao_relative_pos(data, root_pos),  # 6
                self._get_trigger_flags(info),  # 6
                self._get_finish_relative_pos(data, root_pos),  # 2
                self._get_finish_flag(info),  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 75), f"Expected obs shape (*, 75), got {obs.shape}"
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        return obs, info
    