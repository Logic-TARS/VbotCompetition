import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotFullEnvCfg


# ==================== 竞赛场景常量 ====================

# 终点平台（"中国结"）- 视觉模型中心 (0, 32.3)
FINISH_ZONE_CENTER = np.array([0.0, 32.3], dtype=np.float32)
FINISH_ZONE_RADIUS = 1.5  # 终点判定半径（m），身体任何部位进入即算到达

# 阶段性导航航点（每段赛道的目标位置，机器人依次导航到各航点） 当前位置是人工调整很正确
STAGE_WAYPOINTS = np.array([
    [0.0, 7.5],    # 第1段终点：section011出口（"2026"平台区域）
    [0.0, 24.3],    # 第2段终点：section012出口（"丙午大吉"平台）
    [0.0, 32.3],    # 第3段终点：section013终点（"中国结"平台）
], dtype=np.float32)
STAGE_REACH_RADIUS = 2.0  # 到达中间航点的判定半径（m），进入即切换下一航点
NUM_STAGES = len(STAGE_WAYPOINTS)
# 航点随机化偏移范围 [x_low, y_low, x_high, y_high]（相对中心）
STAGE_WP_RANDOM_RANGE = np.array([-4.0, 0.0, 4.0, 0.0], dtype=np.float32)

# Y轴方向检查点（递增排列，用于RL稠密奖励塑形 - 全程覆盖3阶段）
CHECKPOINTS_Y = np.array([
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0,       # 第1段
    12.0, 14.0, 16.0, 18.0, 20.0,          # 第2段
    22.0, 24.0, 26.0, 27.5, 29.0, 30.5, 32.0,  # 第3段
], dtype=np.float32)

# 金球位置（滚动球障碍），来自0126_C_section03.xml
GOLD_BALL_POSITIONS = np.array([
    [3.0, 31.23],
    [0.0, 31.23],
    [-3.0, 31.23],
], dtype=np.float32)
GOLD_BALL_CONTACT_RADIUS = 1.5  # 判定与球接触的半径

# 随机地形区域（heightfield centered at Y≈29.33）
RANDOM_TERRAIN_PASSED_Y = 30.5  # Y坐标超过此值视为穿越随机地形

# 赛道边界（全程：从第1段到第3段终点）
BOUNDARY_X_MIN = -6.0
BOUNDARY_X_MAX = 6.0
BOUNDARY_Y_MIN = -5.0
BOUNDARY_Y_MAX = 36.0

# 物理检测参数
FALL_THRESHOLD_ROLL_PITCH = np.deg2rad(45.0)  # roll/pitch超过45°视为摔倒
MIN_STANDING_HEIGHT_RATIO = 0.4  # 低于目标高度40%视为趴下
GRACE_PERIOD_STEPS = 100  # 重置后的物理宽限期（步数），约1秒（基类step()每步+1）

# 庆祝参数
CELEBRATION_DURATION = 2.0  # 需要在终点停留的时间（秒）

# ==================== 竞赛计分（满分25分） ====================
# 阶段一：穿越滚动球区域（二选一，互斥）
#   策略A（避障优先）：不触碰滚球，安全通过 → +10分
#   策略B（鲁棒优先）：触碰滚球且保持不摔倒 → +15分
PHASE1_COMPLETE_Y = 31.5   # Y阈值：过了球区初始位置Y=31.23后算完成阶段一
PHASE1_AVOID_SCORE = 10.0  # 策略A得分
PHASE1_ROBUST_SCORE = 15.0 # 策略B得分
# 阶段二：穿越随机地形并到达终点"中国结" → +5分
PHASE2_FINISH_SCORE = 5.0
# 阶段三：终点庆祝动作 → +5分
PHASE3_CELEBRATION_SCORE = 5.0

# ==================== RL训练塑形奖励（非竞赛计分） ====================
TERMINATION_PENALTY = -200.0   # 摔倒/越界/姿态失控 → 重罚
CHECKPOINT_SHAPING = 2.0      # Y轴检查点稠密引导
BALL_SURVIVE_SHAPING = 2.0     # 碰球后仍站立的即时鼓励（引导策略B）
HEIGHT_REWARD_SCALE = 2.0
HEIGHT_REWARD_SIGMA = 0.05
FORWARD_VEL_SCALE = 1.0
PROGRESS_SCALE = 2.0

# ==================== 多起点出生配置（防止灾难性遗忘） ====================
# 每个 env 随机选择一个赛段起点，同时训练所有地形
SPAWN_CONFIGS = [
    {  # 第1段: 赛道起点 → section011出口 (Y≈7.5)
        "center": np.array([0.0, -2.4], dtype=np.float32),
        "z": 0.5,
        "xy_range": np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float32),
        "initial_stage": 0,
        "base_height_target": 0.5,
    },
    {  # 第2段: section012起点 → section012出口 (Y≈24.3)
        "center": np.array([0.0, 5.83], dtype=np.float32),
        "z": 1.756,
        "xy_range": np.array([-3.0, -2.0, 3.0, 2.0], dtype=np.float32),
        "initial_stage": 1,
        "base_height_target": 1.756,
    },
    {  # 第3段: section013起点 → 终点中国结 (Y≈32.3)
        "center": np.array([0.0, 26.0], dtype=np.float32),
        "z": 1.756,
        "xy_range": np.array([-2.0, -1.0, 2.0, 1.0], dtype=np.float32),
        "initial_stage": 2,
        "base_height_target": 1.756,
    },
]
NUM_SPAWN_CONFIGS = len(SPAWN_CONFIGS)
SEGMENT_COMPLETE_BONUS = 10.0  # 完成当前赛段的一次性奖励


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


@registry.env("vbot_navigation_full", "np")
class VBotFullEnv(NpEnv):
    """
    VBot完整赛道导航任务（含rough_terrain、竞赛3阶段计分）
    继承自NpEnv，使用VBotFullEnvCfg配置
    """
    _cfg: VBotFullEnvCfg
    
    def __init__(self, cfg: VBotFullEnvCfg, num_envs: int = 1):
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
        # 平台表面Z≈1.294 + 机器人站立高度≈0.462
        self.base_height_target = cfg.init_state.pos[2]  # 从配置读取（1.756）
    
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

        # ===== 崎岖地形: 足部传感器（全图：每段地面独立传感器） =====
        self._num_feet = len(cfg.sensor.feet)
        # 全图场景使用 _s1/_s2/_s3 后缀的传感器，按段合并
        self._ground_sections = ["s1", "s2", "s3"]
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
        
        全图场景：每只脚有3个传感器（s1/s2/s3），取各段法向力之和。
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
        # 增益从配置读取（与section012一致: kp=60, kv=0.8）
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
    
    def _normalize_all_quaternions(self, all_dof_pos: np.ndarray) -> np.ndarray:
        """归一化所有freejoint四元数，防止degenerate quaternion导致panic"""
        num_envs = all_dof_pos.shape[0]
        # 所有需要归一化的四元数区间 [start, end)
        quat_ranges = [(self._base_quat_start, self._base_quat_end)]
        if self._robot_arrow_body is not None:
            quat_ranges.append((self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end))
            quat_ranges.append((self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end))
        
        for qs, qe in quat_ranges:
            if qe > all_dof_pos.shape[1]:
                continue
            quats = all_dof_pos[:, qs:qe]  # (n_envs, 4)
            # NaN / Inf 检测：含非有限值的四元数直接置为单位四元数
            has_bad = ~np.all(np.isfinite(quats), axis=-1)  # (n_envs,)
            quats[has_bad] = [0.0, 0.0, 0.0, 1.0]
            norms = np.linalg.norm(quats, axis=-1, keepdims=True)
            # 退化四元数用单位四元数替换
            degenerate = (norms < 1e-6).flatten()
            quats = np.where(norms > 1e-6, quats / norms, 0.0)
            quats[degenerate] = [0.0, 0.0, 0.0, 1.0]
            all_dof_pos[:, qs:qe] = quats
        return all_dof_pos

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        # 全局NaN/Inf清理：替换为0（四元数部分会由normalize修复）
        all_dof_pos = np.nan_to_num(all_dof_pos, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 向量化设置目标标记位置
        s, e = self._target_marker_dof_start, self._target_marker_dof_end
        all_dof_pos[:, s:e] = pose_commands[:, :3]  # [x, y, yaw]
        
        all_dof_pos = self._normalize_all_quaternions(all_dof_pos)
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        all_dof_pos = np.nan_to_num(all_dof_pos, nan=0.0, posinf=0.0, neginf=0.0)
        
        arrow_height = robot_pos[:, 2] + arrow_offset  # (num_envs,)
        
        # 当前运动方向箭头（向量化）
        cur_speed = np.linalg.norm(base_lin_vel_xy, axis=-1)  # (num_envs,)
        cur_yaw = np.where(cur_speed > 1e-3,
                           np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
                           0.0)
        half_cur = cur_yaw * 0.5
        robot_arrow_pos = np.column_stack([robot_pos[:, 0], robot_pos[:, 1], arrow_height])
        robot_arrow_quat = np.zeros((num_envs, 4), dtype=np.float64)
        robot_arrow_quat[:, 2] = np.sin(half_cur)
        robot_arrow_quat[:, 3] = np.cos(half_cur)
        rs, re = self._robot_arrow_dof_start, self._robot_arrow_dof_end
        all_dof_pos[:, rs:rs+3] = robot_arrow_pos
        all_dof_pos[:, rs+3:re] = robot_arrow_quat
        
        # 期望运动方向箭头（向量化）
        des_speed = np.linalg.norm(desired_vel_xy, axis=-1)
        des_yaw = np.where(des_speed > 1e-3,
                           np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]),
                           0.0)
        half_des = des_yaw * 0.5
        desired_arrow_pos = np.column_stack([robot_pos[:, 0], robot_pos[:, 1], arrow_height])
        desired_arrow_quat = np.zeros((num_envs, 4), dtype=np.float64)
        desired_arrow_quat[:, 2] = np.sin(half_des)
        desired_arrow_quat[:, 3] = np.cos(half_des)
        ds, de = self._desired_arrow_dof_start, self._desired_arrow_dof_end
        all_dof_pos[:, ds:ds+3] = desired_arrow_pos
        all_dof_pos[:, ds+3:de] = desired_arrow_quat
        
        all_dof_pos = self._normalize_all_quaternions(all_dof_pos)
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
        
        # 计算位置误差（相对当前航点，距离更近，梯度信号更有效）
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 计算朝向误差（与section012一致：target_heading = 指向当前航点的方向）
        target_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 达到判定：仅在最终阶段（中国结）才算真正到达
        at_final_stage = current_stage == (NUM_STAGES - 1)
        reached_all = at_final_stage & (distance_to_target < FINISH_ZONE_RADIUS)
        
        # 计算期望速度命令（P控制器，指向终点）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 角速度命令（复用heading_diff，与section012一致）
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
        
        num_envs = data.shape[0]
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

            # 3. 归一化基座高度 (1维, per-env目标高度)
            per_env_ht = state.info.get("base_height_target",
                np.full(num_envs, self.base_height_target, dtype=np.float32))
            base_height_normalized = (root_pos[:, 2] / per_env_ht)[:, np.newaxis]

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
        
        # 更新收集/进度状态（检查点、球接触、地形穿越、终点、庆祝）
        self._update_collection_states(data, root_pos, state.info)
        
        # 检查赛段完成（多起点训练：到达本段终点即完成）
        spawn_stage = state.info.get("spawn_stage", np.zeros(data.shape[0], dtype=np.int32))
        segment_done = np.where(
            spawn_stage < (NUM_STAGES - 1),
            current_stage > spawn_stage,          # 非终点段：航点推进即完成
            state.info["finish_reached"]          # 终点段：到达中国结即完成
        )
        state.info["segment_done"] = segment_done
        
        # 计算奖励（传递root状态）
        reward = self._compute_reward(data, state.info, root_pos, root_quat, root_vel)
        
        # 计算终止条件（完整的摔倒/越界/姿态/超时检测）
        terminated = self._compute_terminated(data, state.info, root_pos, root_quat, root_vel)
        
        # 将终止惩罚叠加到奖励中
        termination_penalty = state.info.get("termination_penalty", np.zeros(data.shape[0], dtype=np.float32))
        reward = reward + termination_penalty
        
        # 更新时间（步数由基类step()统一管理，不在此处递增）
        state.info["time_elapsed"] = state.info.get(
            "time_elapsed", np.zeros(data.shape[0], dtype=np.float32)
        ) + self._cfg.sim_dt
        
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
        
        # 1. 基座接触地面（摔倒）— 合并3段地面传感器
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
        
        # 4. 高度过低（per-env目标高度）
        per_env_ht = info.get("base_height_target",
            np.full(n_envs, self.base_height_target, dtype=np.float32))
        height_too_low = root_pos[:, 2] < (per_env_ht * MIN_STANDING_HEIGHT_RATIO)
        
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
        
        # 赛段完成终止（正常完成，无惩罚）
        segment_done = info.get("segment_done", np.zeros(n_envs, dtype=bool))
        
        terminated = physical_termination | timeout | segment_done
        
        # 物理终止施加重罚（超时和赛段完成不额外罚分）
        info["termination_penalty"] = np.where(
            physical_termination, TERMINATION_PENALTY, 0.0
        ).astype(np.float32)
        
        return terminated
    
    # ==================== 收集/进度状态更新 ====================
    
    def _update_collection_states(self, data: mtx.SceneData,
                                  root_pos: np.ndarray, info: dict):
        """
        更新三阶段竞赛状态 + RL辅助检查点
        
        竞赛三阶段（满分25分）：
          阶段一：穿越滚动球区域（Y > PHASE1_COMPLETE_Y）
            - 策略A（未碰球）→ +10分
            - 策略B（碰球且存活）→ +15分
          阶段二：到达终点"中国结" → +5分
          阶段三：终点庆祝动作 → +5分
        """
        robot_xy = root_pos[:, :2]
        robot_y = root_pos[:, 1]
        n_envs = root_pos.shape[0]
        
        # 判断当前是否站立（用于球接触存活判定，per-env目标高度）
        per_env_ht = info.get("base_height_target",
            np.full(n_envs, self.base_height_target, dtype=np.float32))
        is_standing = root_pos[:, 2] > (per_env_ht * 0.6)
        
        # ===== Y轴检查点（RL稠密塑形，非竞赛计分） =====
        checkpoints_reached = info["checkpoints_reached"]
        for i, cp_y in enumerate(CHECKPOINTS_Y):
            newly_reached = (robot_y >= cp_y) & ~checkpoints_reached[:, i]
            checkpoints_reached[:, i] |= newly_reached
        
        # ===== 金球接触检测（2D距离检测） =====
        ball_contacted = info["ball_contacted"]
        for i, ball_pos in enumerate(GOLD_BALL_POSITIONS):
            dist = np.linalg.norm(robot_xy - ball_pos, axis=-1)
            newly_contacted = (dist < GOLD_BALL_CONTACT_RADIUS) & ~ball_contacted[:, i]
            ball_contacted[:, i] |= newly_contacted
        
        # 即时球接触存活标记（碰球 + 仍站立 → 用于RL塑形奖励）
        ball_survival_rewarded = info["ball_survival_rewarded"]
        for i in range(len(GOLD_BALL_POSITIONS)):
            newly = ball_contacted[:, i] & ~ball_survival_rewarded[:, i] & is_standing
            ball_survival_rewarded[:, i] |= newly
        
        # ===== 随机地形穿越检测 =====
        terrain_passed = info["terrain_passed"]
        newly_passed = (robot_y >= RANDOM_TERRAIN_PASSED_Y) & ~terrain_passed
        terrain_passed |= newly_passed
        info["terrain_passed"] = terrain_passed
        
        # ===== 阶段一完成检测：Y > PHASE1_COMPLETE_Y（通过球区） =====
        phase1_completed = info["phase1_completed"]
        newly_phase1 = (robot_y >= PHASE1_COMPLETE_Y) & ~phase1_completed
        phase1_completed |= newly_phase1
        info["phase1_completed"] = phase1_completed
        
        # ===== 阶段二：终点到达检测（身体任何部位进入中国结区域即算到达） =====
        finish_dist = np.linalg.norm(robot_xy - FINISH_ZONE_CENTER, axis=-1)
        finish_reached = info["finish_reached"]
        newly_finished = (finish_dist < FINISH_ZONE_RADIUS) & ~finish_reached
        finish_reached |= newly_finished
        info["finish_reached"] = finish_reached
        
        # ===== 阶段三：庆祝 =====
        # 庆祝计时开始
        time_elapsed = info.get("time_elapsed", np.zeros(n_envs, dtype=np.float32))
        celebration_start = info["celebration_start_time"]
        for idx in np.where(newly_finished)[0]:
            if celebration_start[idx] < 0:
                celebration_start[idx] = time_elapsed[idx]
        
        # 庆祝完成检测：到达终点后停留指定时间并有明显关节运动
        celebration_completed = info["celebration_completed"]
        joint_vel = self.get_dof_vel(data)
        for idx in range(n_envs):
            if finish_reached[idx] and not celebration_completed[idx] and celebration_start[idx] >= 0:
                elapsed = time_elapsed[idx] - celebration_start[idx]
                if finish_dist[idx] < FINISH_ZONE_RADIUS:
                    # 在终点区域内，检查是否有明显的关节运动（庆祝动作）
                    joint_speed = np.linalg.norm(joint_vel[idx])
                    if elapsed >= CELEBRATION_DURATION and joint_speed > 0.1:
                        celebration_completed[idx] = True
                else:
                    # 离开终点区域，重置计时
                    celebration_start[idx] = time_elapsed[idx]
    
    def _compute_reward(self, data: mtx.SceneData, info: dict,
                        root_pos: np.ndarray, root_quat: np.ndarray,
                        root_vel: np.ndarray) -> np.ndarray:
        """
        Section013 导航任务奖励计算
        
        === 竞赛计分（满分25分，互斥阶段一 + 阶段二 + 阶段三）===
        阶段一（二选一，互斥）：
          策略A：不触碰滚球，安全通过球区 → +10分
          策略B：触碰滚球且保持不摔倒 → +15分（鼓励抗扰动）
        阶段二：穿越随机地形并到达终点 → +5分
        阶段三：终点庆祝动作 → +5分
        
        === RL训练塑形（稠密信号，非竞赛计分）===
        高度维持、前进速度、Y进度、距离缩减、
        检查点引导、球接触存活鼓励、稳定性惩罚
        """
        cfg = self._cfg
        n_envs = data.shape[0]
        reward = np.zeros(n_envs, dtype=np.float32)
        
        base_lin_vel = np.clip(root_vel[:, :3], -100.0, 100.0)
        base_height = root_pos[:, 2]
        gyro = np.clip(self._model.get_sensor_value(cfg.sensor.base_gyro, data), -100.0, 100.0)
        robot_y = root_pos[:, 1]
        robot_xy = root_pos[:, :2]
        
        # 判断是否站立（per-env目标高度）
        per_env_ht = info.get("base_height_target",
            np.full(n_envs, self.base_height_target, dtype=np.float32))
        is_standing = base_height > (per_env_ht * 0.6)
        standing_mask = is_standing.astype(np.float32)
        
        # ================================================================
        #  竞赛计分奖励（一次性里程碑）
        # ================================================================
        
        # ============ 阶段一：穿越滚动球区域（一次性，互斥） ============
        # 当机器人Y > PHASE1_COMPLETE_Y时判定完成
        # 根据是否碰过球选择策略A(+10)或策略B(+15)
        phase1_completed = info["phase1_completed"]
        phase1_rewarded = info["phase1_rewarded"]
        newly_phase1 = phase1_completed & ~phase1_rewarded
        if np.any(newly_phase1):
            # 判断每个环境是否碰过任何球
            any_ball_contacted = info["ball_contacted"].any(axis=1)
            phase1_score = np.where(any_ball_contacted,
                                    PHASE1_ROBUST_SCORE,   # 策略B: +15
                                    PHASE1_AVOID_SCORE)    # 策略A: +10
            reward += newly_phase1.astype(np.float32) * phase1_score
            phase1_rewarded |= newly_phase1
        info["phase1_rewarded"] = phase1_rewarded
        
        # ============ 阶段二：到达终点"中国结"（一次性 +5） ============
        finish_reached = info["finish_reached"]
        finish_rewarded = info["finish_rewarded"]
        newly_finish = finish_reached & ~finish_rewarded
        reward += newly_finish.astype(np.float32) * PHASE2_FINISH_SCORE
        finish_rewarded |= newly_finish
        info["finish_rewarded"] = finish_rewarded
        
        # ============ 阶段三：庆祝完成（一次性 +5） ============
        celebration_completed = info["celebration_completed"]
        celebration_rewarded = info["celebration_rewarded"]
        newly_celebration = celebration_completed & ~celebration_rewarded
        reward += newly_celebration.astype(np.float32) * PHASE3_CELEBRATION_SCORE
        celebration_rewarded |= newly_celebration
        info["celebration_rewarded"] = celebration_rewarded
        
        # ============ 赛段完成奖励（多起点训练） ============
        segment_done = info.get("segment_done", np.zeros(n_envs, dtype=bool))
        segment_done_rewarded = info.get("segment_done_rewarded", np.zeros(n_envs, dtype=bool))
        newly_segment = segment_done & ~segment_done_rewarded
        reward += newly_segment.astype(np.float32) * SEGMENT_COMPLETE_BONUS
        segment_done_rewarded |= newly_segment
        info["segment_done_rewarded"] = segment_done_rewarded
        
        # ================================================================
        #  RL训练塑形奖励（稠密信号）
        # ================================================================
        
        # ============ 1. 站立高度维持（高斯核奖励） ============
        height_error_sq = np.square(np.clip(base_height - per_env_ht, -10.0, 10.0))
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
        
        # ============ 4. 当前航点距离缩减奖励 ============
        target_xy = info["pose_commands"][:, :2]
        distance_to_waypoint = np.linalg.norm(robot_xy - target_xy, axis=-1)
        last_dist = info.get("last_distance_to_target", distance_to_waypoint.copy())
        dist_reduction = last_dist - distance_to_waypoint
        reward += np.clip(dist_reduction * 0.5, -0.05, 0.3)
        info["last_distance_to_target"] = distance_to_waypoint.copy()
        
        # ============ 5. Y轴检查点引导（RL塑形，非竞赛分） ============
        cp_reached = info["checkpoints_reached"]
        cp_rewarded = info["checkpoints_rewarded"]
        for i in range(len(CHECKPOINTS_Y)):
            newly = cp_reached[:, i] & ~cp_rewarded[:, i]
            reward += newly.astype(np.float32) * CHECKPOINT_SHAPING
            cp_rewarded[:, i] |= newly
        
        # ============ 6. 球接触存活鼓励（RL塑形，鼓励策略B） ============
        # 碰到球且仍站立 → 小额即时奖励，引导agent接受碰撞而非绕路
        ball_contacted = info["ball_contacted"]
        ball_survival_rewarded = info["ball_survival_rewarded"]
        ball_survival_shaping_given = info.get("ball_survival_shaping_given",
            np.zeros((n_envs, len(GOLD_BALL_POSITIONS)), dtype=bool))
        for i in range(len(GOLD_BALL_POSITIONS)):
            newly = ball_survival_rewarded[:, i] & ~ball_survival_shaping_given[:, i]
            reward += newly.astype(np.float32) * BALL_SURVIVE_SHAPING
            ball_survival_shaping_given[:, i] |= newly
        info["ball_survival_shaping_given"] = ball_survival_shaping_given
        
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
        reward += np.square(np.clip(base_lin_vel[:, 2], -50.0, 50.0)) * -0.5
        
        # XY角速度惩罚（clip防止overflow）
        gyro_clipped = np.clip(gyro[:, :2], -50.0, 50.0)
        reward += np.sum(np.square(gyro_clipped), axis=-1) * -0.05
        
        # 动作变化率惩罚
        current_actions = info["current_actions"]
        last_actions = info.get("last_actions", current_actions)
        action_rate = np.sum(np.square(current_actions - last_actions), axis=-1)
        reward += action_rate * -0.01
        
        # 力矩惩罚
        reward += np.sum(np.square(np.clip(data.actuator_ctrls, -100.0, 100.0)), axis=-1) * -1e-5
        
        # 关节速度惩罚（clip防止overflow）
        joint_vel = np.clip(self.get_dof_vel(data), -100.0, 100.0)
        reward += np.sum(np.square(joint_vel), axis=-1) * -5e-5
        
        # NaN保护
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-10.0)
        return reward

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        重置环境
        
        关键要求：机器人初始位置必须随机分布在"丙午大吉"平台上（一票否决项）
        """
        cfg: VBotFullEnvCfg = self._cfg
        num_envs = data.shape[0]
        
        # ===== 多起点随机出生（向量化） =====
        spawn_indices = np.random.randint(0, NUM_SPAWN_CONFIGS, size=num_envs)
        
        # 预提取所有config字段为数组，用fancy indexing向量化
        all_centers = np.array([sc["center"] for sc in SPAWN_CONFIGS])         # (3, 2)
        all_z = np.array([sc["z"] for sc in SPAWN_CONFIGS])                    # (3,)
        all_ht = np.array([sc["base_height_target"] for sc in SPAWN_CONFIGS])  # (3,)
        all_stages = np.array([sc["initial_stage"] for sc in SPAWN_CONFIGS])   # (3,)
        all_xy_lo = np.array([sc["xy_range"][:2] for sc in SPAWN_CONFIGS])     # (3, 2)
        all_xy_hi = np.array([sc["xy_range"][2:] for sc in SPAWN_CONFIGS])     # (3, 2)
        
        centers = all_centers[spawn_indices]           # (num_envs, 2)
        xy_lo = all_xy_lo[spawn_indices]               # (num_envs, 2)
        xy_hi = all_xy_hi[spawn_indices]               # (num_envs, 2)
        offsets = np.random.uniform(xy_lo, xy_hi).astype(np.float32)
        robot_init_xy = (centers + offsets).astype(np.float32)
        terrain_heights = all_z[spawn_indices].astype(np.float32)
        per_env_height_target = all_ht[spawn_indices].astype(np.float32)
        initial_stages = all_stages[spawn_indices].astype(np.int32)
        
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])
        
        # 随机初始朝向（向量化，无for循环）
        spawn_yaw = np.random.uniform(-np.pi, np.pi, num_envs)
        # 批量 euler_to_quat: quat = [0, 0, sin(yaw/2), cos(yaw/2)]
        half_yaw = spawn_yaw * 0.5
        spawn_quat = np.zeros((num_envs, 4), dtype=np.float64)
        spawn_quat[:, 2] = np.sin(half_yaw)
        spawn_quat[:, 3] = np.cos(half_yaw)
        
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
        stage_waypoints_rand = np.tile(STAGE_WAYPOINTS, (num_envs, 1, 1)) + wp_offsets  # (num_envs, 3, 2)
        
        # 目标位置：根据每个env的初始阶段选择对应航点（向量化）
        env_indices = np.arange(num_envs)
        target_positions = stage_waypoints_rand[env_indices, initial_stages]  # (num_envs, 2)
        target_headings = np.zeros((num_envs, 1), dtype=np.float32)
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
        
        # 向量化归一化所有四元数（使用批量函数）
        dof_pos = self._normalize_all_quaternions(dof_pos)
        
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
        
        # 计算速度命令（指向终点的P控制器）
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        reached_all = np.zeros(num_envs, dtype=bool)  # 初始时不可能已到达终点
        
        # 计算期望速度（指向第1段航点）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 朝向误差（与section012一致：target_heading = 指向终点的方向）
        target_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 角速度命令（复用heading_diff，与section012一致）
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
            base_height_normalized = (root_pos[:, 2] / per_env_height_target)[:, np.newaxis]
            
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
        
        # ===== 构建初始信息（包含所有追踪字段） =====
        # 计算到当前航点目标的距离（per-env）
        distance_to_target_init = np.linalg.norm(
            robot_init_xy - target_positions, axis=-1)
        
        # 预标记出生点之后的检查点（向量化，避免虚假奖励）
        # CHECKPOINTS_Y: (num_checkpoints,), robot_init_xy[:, 1]: (num_envs,)
        behind = CHECKPOINTS_Y[np.newaxis, :] <= robot_init_xy[:, 1:2]  # (num_envs, num_checkpoints)
        checkpoints_reached_init = behind.copy()
        checkpoints_rewarded_init = behind.copy()
        
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
            "last_distance_to_target": distance_to_target_init.copy(),
            # Y轴检查点（RL稠密塑形，出生点之后的已预标记）
            "checkpoints_reached": checkpoints_reached_init,
            "checkpoints_rewarded": checkpoints_rewarded_init,
            # 滚动球接触检测（3个金球）
            "ball_contacted": np.zeros((num_envs, len(GOLD_BALL_POSITIONS)), dtype=bool),
            "ball_survival_rewarded": np.zeros((num_envs, len(GOLD_BALL_POSITIONS)), dtype=bool),
            "ball_survival_shaping_given": np.zeros((num_envs, len(GOLD_BALL_POSITIONS)), dtype=bool),
            # 随机地形穿越
            "terrain_passed": np.zeros(num_envs, dtype=bool),
            # === 竞赛三阶段追踪 ===
            # 阶段一：穿越球区（互斥评分：策略A +10 / 策略B +15）
            "phase1_completed": np.zeros(num_envs, dtype=bool),
            "phase1_rewarded": np.zeros(num_envs, dtype=bool),
            # 阶段二：到达终点（+5）
            "finish_reached": np.zeros(num_envs, dtype=bool),
            "finish_rewarded": np.zeros(num_envs, dtype=bool),
            # 阶段三：庆祝（+5）
            "celebration_start_time": np.full(num_envs, -1.0, dtype=np.float32),
            "celebration_completed": np.zeros(num_envs, dtype=bool),
            "celebration_rewarded": np.zeros(num_envs, dtype=bool),
            # 阶段性航点导航（多起点：initial_stage 由出生点决定）
            "current_stage": initial_stages.copy(),
            "spawn_stage": initial_stages.copy(),
            "stage_waypoints_rand": stage_waypoints_rand,
            # 多起点训练
            "base_height_target": per_env_height_target.copy(),
            "segment_done": np.zeros(num_envs, dtype=bool),
            "segment_done_rewarded": np.zeros(num_envs, dtype=bool),
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

    