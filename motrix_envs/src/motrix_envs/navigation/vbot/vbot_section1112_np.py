import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection011012EnvCfg


# ==================== Section011+012 场景常量 ====================

# 终点平台（丙午大吉）
FINISH_ZONE_CENTER = np.array([0.0, 24.3], dtype=np.float32)
FINISH_ZONE_RADIUS = 1.5  # 终点判定半径（m）

# 阶段性导航航点（section011 → section012）
STAGE_WAYPOINTS = np.array([
    [0.0, 7.5],    # 第1段终点：section011出口（"2026"平台区域）
    [0.0, 24.3],   # 第2段终点：section012出口（"丙午大吉"平台）
], dtype=np.float32)
STAGE_REACH_RADIUS = 2.0  # 到达中间航点的判定半径（m）
NUM_STAGES = len(STAGE_WAYPOINTS)
# 航点随机化偏移范围 [x_low, y_low, x_high, y_high]（相对中心）
STAGE_WP_RANDOM_RANGE = np.array([-4.0, 0.0, 4.0, 0.0], dtype=np.float32)

# Y轴方向检查点（覆盖section011+012，用于RL稠密奖励塑形）
CHECKPOINTS_Y = np.array([
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0,              # 第1段（section011）
    12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,   # 第2段（section012）
], dtype=np.float32)

# 赛道边界（section011+012范围）
BOUNDARY_X_MIN = -6.0
BOUNDARY_X_MAX = 6.0
BOUNDARY_Y_MIN = -5.0
BOUNDARY_Y_MAX = 28.0

# 物理检测参数
FALL_THRESHOLD_ROLL_PITCH = np.deg2rad(45.0)  # roll/pitch超过45°视为摔倒
MIN_STANDING_HEIGHT_RATIO = 0.4  # 低于目标高度40%视为趴下
GRACE_PERIOD_STEPS = 100  # 重置后的物理宽限期（步数），约1秒

# ==================== RL训练奖励 ====================
TERMINATION_PENALTY = -200.0   # 摔倒/越界/姿态失控 → 重罚
CHECKPOINT_SHAPING = 2.0       # Y轴检查点稠密引导
FINISH_SCORE = 10.0            # 到达终点奖励
HEIGHT_REWARD_SCALE = 2.0
HEIGHT_REWARD_SIGMA = 0.05
FORWARD_VEL_SCALE = 1.0
PROGRESS_SCALE = 2.0


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


@registry.env("vbot_navigation_section011_012", "np")
class VBotSection011012Env(NpEnv):
    """
    VBot在Section011+012合并地形上的导航任务
    使用scene_section011_012.xml，包含楼梯平台(011)和复杂地形(012)
    传感器: s1(011地面) + s2(012地面)，无s3
    """
    _cfg: VBotSection011012EnvCfg
    
    def __init__(self, cfg: VBotSection011012EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # ===== 崎岖地形适应模式（与section012一致） =====
        self._rough_terrain = bool(getattr(cfg, "rough_terrain_mode", False))
        self._state_history_len = int(getattr(cfg, "state_history_length", 3)) if self._rough_terrain else 0
        if self._rough_terrain:
            print(f"[ROUGH TERRAIN] 崎岖地形模式已启用: "
                  f"足部接触力+base_height+状态历史({self._state_history_len}帧)")
        
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
        
        # 动作空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # 计算观测维度（与section012一致）
        # base: 54维
        # rough terrain extras: foot_contacts:4 + foot_forces_body:12 + base_height:1 = 17
        # state history: N * (joint_pos_rel:12 + joint_vel:12 + gravity:3) = N * 27
        self._obs_base_dim = 54
        if self._rough_terrain:
            self._obs_terrain_dim = 17  # 4 + 12 + 1
            self._obs_history_frame_dim = 27  # 12 + 12 + 3
            self._obs_total_dim = (self._obs_base_dim + self._obs_terrain_dim
                                   + self._state_history_len * self._obs_history_frame_dim)
        else:
            self._obs_terrain_dim = 0
            self._obs_history_frame_dim = 0
            self._obs_total_dim = self._obs_base_dim
        
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_total_dim,), dtype=np.float32
        )
        
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
        # 使用配置中的随机化范围 [x_min, y_min, x_max, y_max]
        self.spawn_range = np.array(cfg.init_state.pos_randomization_range, dtype=np.float32)
    
        # 机器人站立目标高度（用于高度维持奖励和摔倒检测）
        self.base_height_target = cfg.init_state.pos[2]  # 从配置读取
    
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

        # PD增益从配置读取（与section012一致，便于迁移预训练权重）
        self.kp = float(getattr(cfg.control_config, 'stiffness', 60.0))
        self.kv = float(getattr(cfg.control_config, 'damping', 0.8))

        # ===== 崎岖地形: 足部传感器（多段地面：s1/s2） =====
        self._num_feet = len(cfg.sensor.feet)
        # 从配置读取地面段标识，默认 ["s1", "s2"]
        self._ground_sections = list(getattr(cfg, "ground_sections", ["s1", "s2"]))
        self._foot_sensor_names_per_section = {
            sec: [f"{foot}_foot_contact_{sec}" for foot in cfg.sensor.feet]
            for sec in self._ground_sections
        }
        self._base_contact_names = [f"base_contact_{sec}" for sec in self._ground_sections]

        # 崎岖地形参数缓存
        if self._rough_terrain:
            self._rough_attitude_scale = float(getattr(cfg, 'rough_attitude_penalty_scale', 0.1))
            self._rough_clearance_scale = float(getattr(cfg, 'rough_foot_clearance_scale', 1.0))
            self._rough_clearance_target = float(getattr(cfg, 'rough_foot_clearance_target', 0.08))
            self._rough_stumble_scale = float(getattr(cfg, 'rough_stumble_penalty_scale', 0.5))
            self._rough_air_time_target = float(getattr(cfg, 'rough_feet_air_time_target', 0.25))
            self._rough_force_penalty_scale = float(getattr(cfg, 'rough_contact_force_penalty_scale', 0.01))

    def _get_foot_contact_forces_body(self, data: mtx.SceneData,
                                       root_quat: np.ndarray) -> np.ndarray:
        """获取4只脚的接触力（转换到body frame），返回 (n_envs, 12)
        
        多段地面场景：每只脚有多个传感器（s1/s2），取各段法向力之和。
        同一时刻脚只能接触一段地面，其余段力为零，所以求和等于实际力。
        """
        n_envs = data.shape[0]
        forces = []
        for foot_idx in range(self._num_feet):
            combined_force = np.zeros((n_envs, 3), dtype=np.float32)
            for sec in self._ground_sections:
                sensor_name = self._foot_sensor_names_per_section[sec][foot_idx]
                try:
                    force_world = self._model.get_sensor_value(sensor_name, data)
                    combined_force += force_world
                except Exception:
                    pass
            force_body = Quaternion.rotate_inverse(root_quat, combined_force)
            forces.append(force_body)
        return np.concatenate(forces, axis=-1)  # (n_envs, 12)

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
        ground_prefix = self._cfg.asset.ground_subtree  # "C_"
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))
        
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
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        pos_error = target_pos - current_pos
        torques = self.kp * pos_error - self.kv * current_vel
        
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
        
        # 导航目标 — 阶段性航点导航
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        
        # 航点推进：到达当前阶段航点后自动切换到下一阶段
        current_stage = state.info["current_stage"]
        stage_waypoints_rand = state.info["stage_waypoints_rand"]  # (n_envs, NUM_STAGES, 2)
        for s in range(NUM_STAGES - 1):
            at_this_stage = current_stage == s
            if not np.any(at_this_stage):
                continue
            wp_s = stage_waypoints_rand[:, s, :]  # (n_envs, 2)
            dist_to_wp = np.linalg.norm(robot_position - wp_s, axis=1)
            advance = at_this_stage & (dist_to_wp < STAGE_REACH_RADIUS)
            current_stage[advance] = s + 1
        
        # 更新 pose_commands 为当前阶段的随机航点
        pose_commands = state.info["pose_commands"]
        num_envs = data.shape[0]
        for env_idx in range(num_envs):
            pose_commands[env_idx, :2] = stage_waypoints_rand[env_idx, current_stage[env_idx]]
        target_position = pose_commands[:, :2]
        
        # 计算位置误差（相对当前航点）
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 计算朝向误差
        target_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 达到判定：仅在最终阶段才算真正到达
        at_final_stage = current_stage == (NUM_STAGES - 1)
        reached_all = at_final_stage & (distance_to_target < FINISH_ZONE_RADIUS)
        
        # 计算期望速度命令（P控制器，指向当前航点）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 角速度命令
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_diff) < deadband_yaw, 0.0, desired_yaw_rate)
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
        
        # ===== 基础观测 (54维) =====
        base_obs = np.concatenate(
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
        
        if self._rough_terrain:
            # ===== 崎岖地形额外观测（与section012一致） =====
            # 1. 足部接触力 (body frame, 4腿 x 3轴 = 12维)
            foot_forces_body = self._get_foot_contact_forces_body(data, root_quat)
            state.info["foot_forces_body"] = foot_forces_body

            # 2. 足部是否接触地面 (4维 binary)
            foot_contacts = (np.linalg.norm(
                foot_forces_body.reshape(num_envs, self._num_feet, 3), axis=-1
            ) > 0.1).astype(np.float32)  # (n_envs, 4)
            state.info["foot_contacts"] = foot_contacts

            # 3. 归一化基座高度 (1维)
            base_height_normalized = (root_pos[:, 2] / self.base_height_target)[:, np.newaxis]

            # 4. 状态历史 (N * 27维) — 用于隐式地形推断
            current_frame = np.concatenate([
                noisy_joint_angle,    # 12
                noisy_joint_vel,      # 12
                projected_gravity,    # 3
            ], axis=-1)  # (n_envs, 27)

            history_buffer = state.info.get("state_history", None)
            if history_buffer is None or history_buffer.shape[0] != num_envs:
                history_buffer = np.tile(
                    current_frame[:, np.newaxis, :],
                    (1, self._state_history_len, 1)
                )
            else:
                history_buffer = np.concatenate([
                    current_frame[:, np.newaxis, :],
                    history_buffer[:, :-1, :]
                ], axis=1)
            state.info["state_history"] = history_buffer

            history_flat = history_buffer.reshape(num_envs, -1)

            obs = np.concatenate([
                base_obs,                  # 54
                foot_contacts,             # 4
                foot_forces_body / 50.0,   # 12 (归一化)
                base_height_normalized,    # 1
                history_flat,              # N * 27
            ], axis=-1)
        else:
            obs = base_obs
            state.info["foot_contacts"] = np.zeros((num_envs, self._num_feet), dtype=np.float32)

        assert obs.shape == (num_envs, self._obs_total_dim), (
            f"Expected obs shape (*, {self._obs_total_dim}), got {obs.shape}"
        )
        
        # 更新目标标记和箭头
        self._update_target_marker(data, pose_commands)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 更新收集/进度状态（检查点、终点）
        self._update_collection_states(data, root_pos, state.info)
        
        # 计算奖励（传递root状态）
        reward = self._compute_reward(data, state.info, root_pos, root_quat, root_vel)
        
        # 计算终止条件
        terminated = self._compute_terminated(data, state.info, root_pos, root_quat, root_vel)
        
        # 将终止惩罚叠加到奖励中
        termination_penalty = state.info.get("termination_penalty", np.zeros(data.shape[0], dtype=np.float32))
        reward = reward + termination_penalty
        
        # 更新时间和步数
        state.info["time_elapsed"] = state.info.get(
            "time_elapsed", np.zeros(data.shape[0], dtype=np.float32)
        ) + self._cfg.sim_dt
        state.info["steps"] = state.info.get(
            "steps", np.zeros(data.shape[0], dtype=np.int32)
        ) + 1
        
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        
        return state
    
    def _compute_terminated(self, data: mtx.SceneData, info: dict,
                            root_pos: np.ndarray, root_quat: np.ndarray,
                            root_vel: np.ndarray) -> np.ndarray:
        """
        终止条件（满足任一则终止且施加重罚）：
        1. 基座接触地面（摔倒）
        2. 姿态失控（roll/pitch > 45°）
        3. 越界（超出赛道范围）
        4. 高度过低（≤ 40%目标高度 = 坍塌/趴下）
        5. 关节速度异常（overflow / NaN / Inf）
        6. episode超时
        """
        n_envs = data.shape[0]
        steps = info.get("steps", np.zeros(n_envs, dtype=np.int32))
        in_grace = steps < GRACE_PERIOD_STEPS
        
        # 1. 基座接触地面（摔倒）— 合并多段地面传感器
        base_contact = np.zeros(n_envs, dtype=bool)
        for bc_name in self._base_contact_names:
            try:
                bc_val = self._model.get_sensor_value(bc_name, data)
                if bc_val.ndim == 0:
                    base_contact |= np.array([bc_val > 0.01], dtype=bool)
                elif bc_val.shape[0] != n_envs:
                    base_contact |= np.full(n_envs, bc_val.flatten()[0] > 0.01, dtype=bool)
                else:
                    base_contact |= (bc_val > 0.01).flatten()[:n_envs]
            except Exception:
                pass
        
        # 2. 姿态失控
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        attitude_fail = (np.abs(roll) > FALL_THRESHOLD_ROLL_PITCH) | (np.abs(pitch) > FALL_THRESHOLD_ROLL_PITCH)
        
        # 3. 越界检测
        out_of_bounds = (
            (root_pos[:, 0] < BOUNDARY_X_MIN) |
            (root_pos[:, 0] > BOUNDARY_X_MAX) |
            (root_pos[:, 1] < BOUNDARY_Y_MIN) |
            (root_pos[:, 1] > BOUNDARY_Y_MAX)
        )
        
        # 4. 高度过低
        height_too_low = root_pos[:, 2] < (self.base_height_target * MIN_STANDING_HEIGHT_RATIO)
        
        # 5. 关节速度异常
        dof_vel = self.get_dof_vel(data)
        if dof_vel.ndim > 1:
            vel_max = np.abs(dof_vel).max(axis=1)
            vel_nan_inf = np.isnan(dof_vel).any(axis=1) | np.isinf(dof_vel).any(axis=1)
        else:
            vel_max = np.abs(dof_vel)
            vel_nan_inf = np.isnan(dof_vel) | np.isinf(dof_vel)
        vel_overflow = vel_max > 100.0
        
        # 6. 超时
        timeout = steps >= self._cfg.max_episode_steps
        
        # 综合：物理终止条件在宽限期内不生效
        physical_termination = (
            base_contact | attitude_fail | out_of_bounds |
            height_too_low | vel_overflow | vel_nan_inf
        ) & ~in_grace
        
        terminated = physical_termination | timeout
        
        # 物理终止施加重罚（超时不额外罚分）
        info["termination_penalty"] = np.where(
            physical_termination, TERMINATION_PENALTY, 0.0
        ).astype(np.float32)
        
        return terminated
    
    # ==================== 收集/进度状态更新 ====================
    
    def _update_collection_states(self, data: mtx.SceneData,
                                  root_pos: np.ndarray, info: dict):
        """
        更新进度状态：Y轴检查点 + 终点到达检测
        """
        robot_xy = root_pos[:, :2]
        robot_y = root_pos[:, 1]
        
        # ===== Y轴检查点（RL稠密塑形） =====
        checkpoints_reached = info["checkpoints_reached"]
        for i, cp_y in enumerate(CHECKPOINTS_Y):
            newly_reached = (robot_y >= cp_y) & ~checkpoints_reached[:, i]
            checkpoints_reached[:, i] |= newly_reached
        
        # ===== 终点到达检测 =====
        finish_dist = np.linalg.norm(robot_xy - FINISH_ZONE_CENTER, axis=-1)
        finish_reached = info["finish_reached"]
        newly_finished = (finish_dist < FINISH_ZONE_RADIUS) & ~finish_reached
        finish_reached |= newly_finished
        info["finish_reached"] = finish_reached
    
    def _compute_reward(self, data: mtx.SceneData, info: dict,
                        root_pos: np.ndarray, root_quat: np.ndarray,
                        root_vel: np.ndarray) -> np.ndarray:
        """
        Section011+012 导航任务奖励计算
        
        === 里程碑奖励 ===
        到达终点（丙午大吉平台）→ +FINISH_SCORE
        
        === RL训练塑形（稠密信号）===
        高度维持、前进速度、Y进度、距离缩减、
        检查点引导、稳定性惩罚
        """
        cfg = self._cfg
        n_envs = data.shape[0]
        reward = np.zeros(n_envs, dtype=np.float32)
        
        base_lin_vel = root_vel[:, :3]
        base_height = root_pos[:, 2]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        robot_y = root_pos[:, 1]
        robot_xy = root_pos[:, :2]
        
        # 判断是否站立
        is_standing = base_height > (self.base_height_target * 0.6)
        standing_mask = is_standing.astype(np.float32)
        
        # ================================================================
        #  里程碑奖励（一次性）
        # ================================================================
        
        # ============ 到达终点（一次性 +FINISH_SCORE） ============
        finish_reached = info["finish_reached"]
        finish_rewarded = info["finish_rewarded"]
        newly_finish = finish_reached & ~finish_rewarded
        reward += newly_finish.astype(np.float32) * FINISH_SCORE
        finish_rewarded |= newly_finish
        info["finish_rewarded"] = finish_rewarded
        
        # ================================================================
        #  RL训练塑形奖励（稠密信号）
        # ================================================================
        
        # ============ 1. 站立高度维持（高斯核奖励） ============
        height_error_sq = np.square(base_height - self.base_height_target)
        reward += HEIGHT_REWARD_SCALE * np.exp(-height_error_sq / HEIGHT_REWARD_SIGMA)
        
        # 存活奖励（站立+0.01，趴下-0.02）
        reward += np.where(is_standing, 0.01, -0.02)
        
        # ============ 2. Y方向前进速度奖励 ============
        forward_vel_y = base_lin_vel[:, 1]
        reward += FORWARD_VEL_SCALE * np.clip(forward_vel_y, -0.3, 1.5) * standing_mask
        
        # ============ 3. Y坐标进度奖励 ============
        last_y = info.get("last_y", robot_y.copy())
        delta_y = robot_y - last_y
        reward += PROGRESS_SCALE * np.clip(delta_y, -0.05, 0.3) * standing_mask
        info["last_y"] = robot_y.copy()
        
        # ============ 4. 终点距离缩减奖励 ============
        distance_to_target = np.linalg.norm(robot_xy - FINISH_ZONE_CENTER, axis=-1)
        last_dist = info.get("last_distance_to_finish", distance_to_target.copy())
        dist_reduction = last_dist - distance_to_target
        reward += np.clip(dist_reduction * 0.5, -0.05, 0.3)
        info["last_distance_to_finish"] = distance_to_target.copy()
        
        # ============ 5. Y轴检查点引导（RL塑形） ============
        cp_reached = info["checkpoints_reached"]
        cp_rewarded = info["checkpoints_rewarded"]
        for i in range(len(CHECKPOINTS_Y)):
            newly = cp_reached[:, i] & ~cp_rewarded[:, i]
            reward += newly.astype(np.float32) * CHECKPOINT_SHAPING
            cp_rewarded[:, i] |= newly
        
        # ============ 稳定性惩罚 ============
        # 姿态惩罚（roll/pitch）
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        reward += (np.abs(roll) + np.abs(pitch)) * -0.5
        
        # 垂直速度惩罚
        reward += np.square(base_lin_vel[:, 2]) * -0.5
        
        # XY角速度惩罚
        reward += np.sum(np.square(gyro[:, :2]), axis=-1) * -0.05
        
        # 动作变化率惩罚
        current_actions = info["current_actions"]
        last_actions = info.get("last_actions", current_actions)
        action_rate = np.sum(np.square(current_actions - last_actions), axis=-1)
        reward += action_rate * -0.01
        
        # 力矩惩罚
        reward += np.sum(np.square(data.actuator_ctrls), axis=-1) * -1e-5
        
        # 关节速度惩罚
        joint_vel = self.get_dof_vel(data)
        reward += np.sum(np.square(joint_vel), axis=-1) * -5e-5
        
        # NaN保护
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-10.0)
        return reward

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        重置环境
        
        机器人初始位置从配置中读取（section011起始区域）
        """
        cfg: VBotSection011012EnvCfg = self._cfg
        num_envs = data.shape[0]
        
        # ===== 随机生成初始位置 =====
        random_offset = np.random.uniform(
            low=self.spawn_range[:2],
            high=self.spawn_range[2:],
            size=(num_envs, 2)
        )
        robot_init_xy = self.spawn_center[:2] + random_offset  # [num_envs, 2]
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])  # [num_envs, 3]
        
        # 随机初始朝向（大致朝向+Y方向 = 赛道前进方向）
        spawn_yaw = np.random.uniform(-np.pi / 6, np.pi / 6, num_envs)
        spawn_quat = np.array([self._euler_to_quat(0, 0, yaw) for yaw in spawn_yaw])
        
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        
        # 设置 base 的 XYZ位置（DOF 3-5）和朝向（DOF 6-9）
        dof_pos[:, 3:6] = robot_init_xyz
        dof_pos[:, self._base_quat_start:self._base_quat_end] = spawn_quat
        
        # 为每个 env 生成带随机偏移的航点
        wp_offsets = np.random.uniform(
            low=STAGE_WP_RANDOM_RANGE[:2],
            high=STAGE_WP_RANDOM_RANGE[2:],
            size=(num_envs, NUM_STAGES, 2)
        ).astype(np.float32)
        stage_waypoints_rand = np.tile(STAGE_WAYPOINTS, (num_envs, 1, 1)) + wp_offsets  # (num_envs, NUM_STAGES, 2)
        
        # 目标位置：初始航点为第1段随机航点，后续由 update_state 自动推进
        target_positions = stage_waypoints_rand[:, 0, :]  # (num_envs, 2)
        target_headings = np.zeros((num_envs, 1), dtype=np.float32)
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
        
        # 归一化所有四元数
        for env_idx in range(num_envs):
            # base四元数
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            qn = np.linalg.norm(quat)
            dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = (
                quat / qn if qn > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )
            
            # 箭头四元数（如果存在）
            if self._robot_arrow_body is not None:
                for start, end in [
                    (self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end),
                    (self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end),
                ]:
                    if end <= len(dof_pos[env_idx]):
                        aq = dof_pos[env_idx, start:end]
                        aqn = np.linalg.norm(aq)
                        dof_pos[env_idx, start:end] = (
                            aq / aqn if aqn > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                        )
        
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
        
        # 计算速度命令（指向第1段航点的P控制器）
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        reached_all = np.zeros(num_envs, dtype=bool)  # 初始时不可能已到达终点
        
        # 计算期望速度
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 朝向误差
        target_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 角速度命令
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
        
        # ===== 基础观测 (54维) =====
        base_obs = np.concatenate(
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
        
        if self._rough_terrain:
            # ===== 崎岖地形额外观测（与update_state一致） =====
            foot_forces_body = self._get_foot_contact_forces_body(data, root_quat)
            foot_contacts = (np.linalg.norm(
                foot_forces_body.reshape(num_envs, self._num_feet, 3), axis=-1
            ) > 0.1).astype(np.float32)
            base_height_normalized = (root_pos[:, 2] / self.base_height_target)[:, np.newaxis]
            
            # 初始状态历史：用当前帧填充所有历史槽位
            current_frame = np.concatenate([
                noisy_joint_angle, noisy_joint_vel, projected_gravity
            ], axis=-1)  # (num_envs, 27)
            history_buffer = np.tile(
                current_frame[:, np.newaxis, :],
                (1, self._state_history_len, 1)
            )
            history_flat = history_buffer.reshape(num_envs, -1)
            
            obs = np.concatenate([
                base_obs,                  # 54
                foot_contacts,             # 4
                foot_forces_body / 50.0,   # 12
                base_height_normalized,    # 1
                history_flat,              # N * 27
            ], axis=-1)
        else:
            obs = base_obs
        
        assert obs.shape == (num_envs, self._obs_total_dim), (
            f"Expected obs shape (*, {self._obs_total_dim}), got {obs.shape}"
        )
        
        # ===== 构建初始信息 =====
        distance_to_finish = np.linalg.norm(robot_init_xy - FINISH_ZONE_CENTER, axis=-1)
        
        info = {
            # 目标命令
            "pose_commands": pose_commands,
            # 动作
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            # 计步与计时
            "steps": np.zeros(num_envs, dtype=np.int32),
            "time_elapsed": np.zeros(num_envs, dtype=np.float32),
            # Y进度追踪
            "last_y": robot_init_xy[:, 1].copy(),
            "last_distance_to_finish": distance_to_finish.copy(),
            # Y轴检查点（RL稠密塑形）
            "checkpoints_reached": np.zeros((num_envs, len(CHECKPOINTS_Y)), dtype=bool),
            "checkpoints_rewarded": np.zeros((num_envs, len(CHECKPOINTS_Y)), dtype=bool),
            # 到达终点
            "finish_reached": np.zeros(num_envs, dtype=bool),
            "finish_rewarded": np.zeros(num_envs, dtype=bool),
            # 阶段性航点导航
            "current_stage": np.zeros(num_envs, dtype=np.int32),  # 0=第1段, 1=第2段
            "stage_waypoints_rand": stage_waypoints_rand,  # (num_envs, NUM_STAGES, 2)
            # 物理
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),
            # 足部接触（崎岖地形用）
            "foot_contacts": np.zeros((num_envs, self._num_feet), dtype=np.float32),
        }
        
        # 崎岖地形：初始化状态历史缓冲区
        if self._rough_terrain:
            info["state_history"] = history_buffer
            info["foot_forces_body"] = foot_forces_body
        
        return obs, info
