import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import VBotSection011EnvCfg

# ==================== 竞赛场景常量（与section012一致） ====================
FALL_THRESHOLD_ROLL_PITCH = np.deg2rad(45.0)
MIN_STANDING_HEIGHT_RATIO = 0.4   # 低于目标高度40%视为摔倒
GRACE_PERIOD_STEPS = 10           # 重置后的宽限期（步数）— 与012一致
TERMINATION_PENALTY = -50.0       # 终止惩罚（与012一致，-200→-50）


@registry.env("vbot_navigation_section011", "np")
class VBotSection011Env(NpEnv):
    """
    VBot Navigation Section 011 Competition Environment (Ported from VBotNavSection1)
    Obstacle navigation with smileys, hongbaos, and finish zone
    """
    _cfg: VBotSection011EnvCfg

    def __init__(self, cfg: VBotSection011EnvCfg, num_envs: int = 1):
        # Call parent class initialization
        super().__init__(cfg, num_envs=num_envs)

        self._debug_logs = bool(getattr(cfg, "debug_logs", False))

        # ===== 课程学习模式 =====
        # 当从section001预训练模型继续训练时，设为True
        # 使观测空间(54维)和动作控制(PD力矩)与section001完全一致
        self._curriculum_from_001 = bool(getattr(cfg, "curriculum_from_001", False))
        if self._curriculum_from_001:
            self._log("[CURRICULUM] 课程学习模式已启用: obs=54维, action=PD力矩控制 (兼容section001)")

        # ===== 崎岖地形适应模式 =====
        self._rough_terrain = bool(getattr(cfg, "rough_terrain_mode", False))
        self._state_history_len = int(getattr(cfg, "state_history_length", 3)) if self._rough_terrain else 0
        if self._rough_terrain:
            self._log(f"[ROUGH TERRAIN] 崎岖地形模式已启用: "
                      f"足部接触力+base_height+状态历史({self._state_history_len}帧)")

        # 互斥校验：课程模式要求观测空间=54维（兼容section001），崎岖地形会扩展观测维度
        if self._curriculum_from_001 and self._rough_terrain:
            raise ValueError(
                "curriculum_from_001 与 rough_terrain_mode 不可同时启用！"
                "课程模式要求观测空间为54维以兼容section001预训练权重，"
                "而崎岖地形模式会扩展观测空间维度，导致权重无法加载。"
            )

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

        if self._curriculum_from_001:
            # 课程模式: 54维观测空间，与section001完全一致
            # linvel:3, gyro:3, gravity:3, joint_angle:12, joint_vel:12, last_actions:12,
            # commands:3, position_error:2, heading_error:1, distance:1, reached:1, stop_ready:1
            self._obs_base_dim = 54
            self._obs_terrain_dim = 0
            self._obs_history_frame_dim = 0
            self._obs_total_dim = 54
        elif self._rough_terrain:
            # 崎岖地形模式: 54 + 17 + N*27 维观测空间（与section012完全一致）
            # base_54 + foot_contacts:4 + foot_forces_body:12 + base_height:1
            # + N * (joint_pos_rel:12 + joint_vel:12 + gravity:3) = 71 + N*27
            self._obs_base_dim = 54
            self._obs_terrain_dim = 17  # 4(foot_contacts) + 12(foot_forces_body) + 1(base_height)
            self._obs_history_frame_dim = 27  # 12 + 12 + 3
            self._obs_total_dim = (self._obs_base_dim + self._obs_terrain_dim
                                   + self._state_history_len * self._obs_history_frame_dim)
        else:
            # 标准模式: 81维观测空间 (60 base + 21 competition-specific)
            # Base: 60 (linvel:3, gyro:3, gravity:3, joint_angle:12, joint_vel:12, last_actions:12,
            #           commands:3, position_error:2, heading_error:1, distance:1, reached:1, stop_ready:1,
            #           base_height:1, base_height_error:1, foot_contacts:4)
            # Competition: 21 (smiley_relative_pos:6, hongbao_relative_pos:6, trigger_flags:6,
            #                   finish_relative_pos:2, finish_flag:1)
            self._obs_base_dim = 81
            self._obs_terrain_dim = 0
            self._obs_history_frame_dim = 0
            self._obs_total_dim = 81

        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_total_dim,), dtype=np.float32
        )

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

        # Fix: 将默认关节角写入 _init_dof_pos（与section012一致）
        self._init_dof_pos[-self._num_action:] = self.default_angles

        self.action_filter_alpha = 0.3

        # PD增益（与section012一致，始终初始化）
        self.kp = float(getattr(cfg.control_config, 'stiffness', 60.0))
        self.kv = float(getattr(cfg.control_config, 'damping', 0.8))

        # === Anti-reward-hacking parameters ===
        # Target base height (from init_state.pos[2], the robot's nominal standing height)
        self.base_height_target = cfg.init_state.pos[2]  # 0.5m for VBot
        self.base_height_sigma = 0.05  # Gaussian kernel sigma for height reward
        self.base_height_reward_scale = 2.0  # Strong incentive to maintain standing height
        self.forward_velocity_reward_scale = 1.0  # Reward for moving toward target
        self.alive_bonus_height_gated = True  # Only award alive bonus when standing
        self.min_standing_height_ratio = 0.6  # Must be at least 60% of target height to count as "standing"

        # Competition configuration handling
        competition_cfg = getattr(cfg, 'competition', None)
        
        if hasattr(cfg, 'start_zone_center'):
            self.start_zone_center = np.array(cfg.start_zone_center, dtype=np.float32)
            self.start_zone_radius = cfg.start_zone_radius
            self.smiley_positions = np.array(cfg.smiley_positions, dtype=np.float32)
            self.smiley_radius = cfg.smiley_radius
            self.hongbao_positions = np.array(cfg.hongbao_positions, dtype=np.float32)
            self.hongbao_radius = cfg.hongbao_radius
            self.finish_zone_center = np.array(cfg.finish_zone_center, dtype=np.float32)
            self.finish_zone_radius = cfg.finish_zone_radius
            self.boundary_x_min = cfg.boundary_x_min
            self.boundary_x_max = cfg.boundary_x_max
            self.boundary_y_min = cfg.boundary_y_min
            self.boundary_y_max = cfg.boundary_y_max
            self.celebration_duration = cfg.celebration_duration
            self.celebration_movement_threshold = cfg.celebration_movement_threshold
        elif competition_cfg is not None:
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
        else:
            self.start_zone_center = np.array([0.0, -2.4], dtype=np.float32)
            self.start_zone_radius = 1.0
            self.smiley_positions = np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0]], dtype=np.float32)
            self.smiley_radius = 0.5
            self.hongbao_positions = np.array([[1.0, 2.5], [1.0, 7.5], [1.0, 12.5]], dtype=np.float32)
            self.hongbao_radius = 0.5
            self.finish_zone_center = np.array([0.0, 7.83], dtype=np.float32)  # 2026平台中心
            self.finish_zone_radius = 1.0
            self.boundary_x_min = -20.0
            self.boundary_x_max = 20.0
            self.boundary_y_min = -10.0
            self.boundary_y_max = 20.0
            self.celebration_duration = 1.0
            self.celebration_movement_threshold = 0.1

        # Fall detection thresholds
        self.fall_threshold_roll_pitch = np.deg2rad(45.0)

        # Illegal contact body names (thigh/calf touching ground = termination)
        # These are body parts that should NEVER touch the ground during normal locomotion
        self.terminate_contact_bodies = [
            "collision_middle_box", "collision_head_box",  # base/torso
        ]
        # Penalize (but not terminate) contact for thigh/calf
        self.penalize_contact_bodies = [
            "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh",  # thighs
            "FR_calf", "FL_calf", "RR_calf", "RL_calf",  # calves
        ]

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

        # ===== base_contact 传感器预校验（与section012一致） =====
        self._base_contact_sensor = getattr(cfg.sensor, 'base_contact', 'base_contact')
        try:
            self._model.get_sensor_value(self._base_contact_sensor,
                                         mtx.SceneData(self._model, batch=[1]))
            self._base_contact_available = True
        except Exception as e:
            print(f"[WARNING] base_contact 传感器 '{self._base_contact_sensor}' 不存在: {e}")
            print("[WARNING] 基座接触摔倒检测将被禁用！")
            self._base_contact_available = False

        # ===== 崎岖地形: 足部传感器 =====
        self._foot_sensor_names = [f"{foot}_foot_contact" for foot in cfg.sensor.feet]
        self._num_feet = len(cfg.sensor.feet)

        # 崎岖地形参数缓存
        if self._rough_terrain:
            self._rough_attitude_scale = float(getattr(cfg, 'rough_attitude_penalty_scale', 0.1))
            self._rough_clearance_scale = float(getattr(cfg, 'rough_foot_clearance_scale', 1.0))
            self._rough_clearance_target = float(getattr(cfg, 'rough_foot_clearance_target', 0.08))
            self._rough_stumble_scale = float(getattr(cfg, 'rough_stumble_penalty_scale', 0.5))

    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置

        DOF layout:
          0-2:  target_marker (slide x, slide y, hinge yaw)
          3-5:  base position (x, y, z)
          6-9:  base quaternion (qx, qy, qz, qw)
          10-21: joint angles (12)
        """
        n_dof = self._model.num_dof_pos
        n_actuators = self._model.num_actuators

        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._base_pos_start = 3
        self._base_pos_end = 6
        self._base_quat_start = 6
        self._base_quat_end = 10
        self._joint_dof_start = 10
        self._joint_dof_end = 10 + n_actuators

        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]

        # 验证模型DOF布局是否匹配预期（与section012一致）
        expected_min = 3 + 7 + n_actuators  # marker(3) + base(7) + joints
        assert n_dof >= expected_min, (
            f"DOF layout mismatch: expected at least {expected_min} DOFs "
            f"(3 marker + 7 base + {n_actuators} joints), got {n_dof}"
        )

    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置

        基于joint终止位置推算（与section012一致），而非硬编码。

        DOF layout (following joints):
          joint_end ~ joint_end+6: robot_heading_arrow freejoint (3 pos + 4 quat)
          joint_end+7 ~ joint_end+13: desired_heading_arrow freejoint (3 pos + 4 quat)
        """
        arrow_base = self._joint_dof_end
        self._robot_arrow_dof_start = arrow_base
        self._robot_arrow_dof_end = arrow_base + 7
        self._desired_arrow_dof_start = arrow_base + 7
        self._desired_arrow_dof_end = arrow_base + 14

        # 验证不越界
        assert self._desired_arrow_dof_end <= self._model.num_dof_pos, (
            f"Arrow DOF range [{self._robot_arrow_dof_start}:{self._desired_arrow_dof_end}] "
            f"exceeds model DOF count {self._model.num_dof_pos}"
        )

        arrow_init_height = self._cfg.init_state.pos[2] + 0.5
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [
                0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0
            ]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [
                0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0
            ]

    def _init_contact_geometry(self):
        """Initialize contact geometry"""
        try:
            self._base_contact_geom = self._model.get_geom("collision_middle_box")
        except Exception:
            self._base_contact_geom = None

    def _log(self, message: str):
        if self._debug_logs:
            print(message)

    def _get_foot_contact_forces_body(self, data: mtx.SceneData,
                                       root_quat: np.ndarray) -> np.ndarray:
        """获取4只脚的接触力（转换到body frame），返回 (n_envs, 12)"""
        n_envs = data.shape[0]
        forces = []
        for sensor_name in self._foot_sensor_names:
            try:
                force_world = self._model.get_sensor_value(sensor_name, data)  # (n_envs, 3)
                force_body = Quaternion.rotate_inverse(root_quat, force_world)
                forces.append(force_body)
            except Exception:
                forces.append(np.zeros((n_envs, 3), dtype=np.float32))
        return np.concatenate(forces, axis=-1)  # (n_envs, 12)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """应用动作：带低通滤波的PD力矩控制（与section012一致）"""
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)

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
        state.data.actuator_ctrls = self._compute_torques(
            state.info["filtered_actions"], state.data
        )

        return state

    def _compute_torques(self, actions, data):
        """PD力矩控制（与section012一致）

        课程模式: 无力矩限幅（与section001一致）
        标准模式: 限幅 [17, 17, 34]*4
        """
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled

        current_pos = self.get_dof_pos(data)
        current_vel = self.get_dof_vel(data)

        pos_error = target_pos - current_pos
        torques = self.kp * pos_error - self.kv * current_vel

        if not self._curriculum_from_001:
            # 标准模式: 限幅保护（与section012一致）
            torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)
            torques = np.clip(torques, -torque_limits, torque_limits)

        return torques

    # ==================== 状态访问 ====================

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def _extract_root_state(self, data: mtx.SceneData):
        """Extract root position, quaternion and velocity"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_vel

    def _quat_to_roll_pitch(self, root_quat: np.ndarray):
        """从四元数提取roll和pitch（与section012一致）"""
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        return roll, pitch

    def _compute_observation(self, data: mtx.SceneData, state: NpEnvState):
        """Compute observations for the environment"""
        cfg = self._cfg

        # Extract state
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # Base linear velocity and gyro from sensors
        base_lin_vel = self._model.get_sensor_value(cfg.sensor.base_linvel, data)
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)

        # Gravity in base frame
        gravity_world = np.array([0, 0, -1], dtype=np.float32)
        gravity_world = np.tile(gravity_world, (data.shape[0], 1))
        projected_gravity = Quaternion.rotate_inverse(root_quat, gravity_world)

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

        # === NEW: Base height observation (anti-reward-hacking) ===
        base_height = root_pos[:, 2]  # Z coordinate of base
        base_height_normalized = base_height / self.base_height_target  # 1.0 = nominal height
        base_height_error = (base_height - self.base_height_target) / self.base_height_target

        # === NEW: Foot contact observations ===
        foot_contacts = np.zeros((data.shape[0], 4), dtype=np.float32)
        foot_sensor_names = ["FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact"]
        for i, sensor_name in enumerate(foot_sensor_names):
            try:
                contact_val = self._model.get_sensor_value(sensor_name, data)
                if contact_val.ndim == 0:
                    foot_contacts[:, i] = float(contact_val > 0.1)
                else:
                    foot_contacts[:, i] = (contact_val.flatten()[:data.shape[0]] > 0.1).astype(np.float32)
            except Exception:
                pass  # Sensor not available, leave as 0

        # Store foot contacts in info for reward computation
        state.info["foot_contacts"] = foot_contacts
        state.info["base_height"] = base_height

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

        # Concatenate observations
        if self._curriculum_from_001:
            # 课程模式: 54维观测，与section001完全一致
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
            assert obs.shape == (data.shape[0], 54), f"[CURRICULUM] Expected obs shape (*, 54), got {obs.shape}"
        else:
            if self._rough_terrain:
                # ===== 崎岖地形模式: 54 + 17 + N*27（与section012完全一致）=====
                num_envs = data.shape[0]

                # 54维基础观测（与 curriculum / section012 base 一致）
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

                # 1. 足部接触力 (body frame, 4腿 x 3轴 = 12维)
                foot_forces_body = self._get_foot_contact_forces_body(data, root_quat)
                state.info["foot_forces_body"] = foot_forces_body

                # 2. 足部是否接触地面 (4维 binary，与section012一致)
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

                # 更新历史缓冲区 (FIFO)
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

                # 拼接完整观测（顺序与section012完全一致）
                obs = np.concatenate([
                    base_obs,                      # 54
                    foot_contacts,                 # 4
                    foot_forces_body / 50.0,       # 12 (归一化)
                    base_height_normalized,        # 1
                    history_flat,                  # N * 27
                ], axis=-1)

            else:
                # 标准模式: 81维基础观测（含competition-specific）
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
                        # Anti-reward-hacking observations
                        base_height_normalized[:, np.newaxis],  # 1
                        base_height_error[:, np.newaxis],  # 1
                        foot_contacts,  # 4
                        # Competition-specific observations
                        smiley_relative_pos,  # 6
                        hongbao_relative_pos,  # 6
                        trigger_flags,  # 6
                        finish_relative_pos,  # 2
                        finish_flag,  # 1
                    ],
                    axis=-1,
                )

            assert obs.shape == (data.shape[0], self._obs_total_dim), \
                f"Expected obs shape (*, {self._obs_total_dim}), got {obs.shape}"

        # Prevent NaN/Inf propagation
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # 将中间量存入info，供 update_state 中的 reward/viz 使用
        state.info["velocity_commands"] = velocity_commands
        state.info["desired_vel_xy"] = desired_vel_xy

        state.obs = obs
        return state, root_pos, root_quat, base_lin_vel

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
        """Compute rewards for the competition (anti-reward-hacking version)
        
        Key design principles to prevent lying-down exploitation:
        1. Alive bonus is GATED by base height — lying down gets ZERO alive bonus
        2. Gaussian kernel base height reward — strong incentive to maintain standing posture
        3. Forward velocity reward — moving toward target is the ONLY way to score high
        4. Strengthened orientation penalty — large roll/pitch is heavily penalized
        5. Vertical velocity penalty — discourages falling/bouncing
        """
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
        
        reward = np.zeros(n_envs, dtype=np.float32)
        
        # =====================================================
        # 1. HEIGHT-GATED ALIVE BONUS (prevents lying-down exploitation)
        # =====================================================
        base_height = info.get("base_height", root_pos[:, 2])
        is_standing = base_height > (self.base_height_target * self.min_standing_height_ratio)
        # Only award alive bonus when the robot is actually standing
        alive_bonus = np.where(is_standing, 0.01, -0.02)  # Penalty for being too low
        reward += alive_bonus
        
        # =====================================================
        # 2. BASE HEIGHT REWARD (Gaussian kernel — core anti-hack)
        # =====================================================
        # r_height = scale * exp(-(h - h_target)^2 / sigma)
        # This provides maximum reward at target height and smoothly decays
        height_error_sq = np.square(base_height - self.base_height_target)
        reward_height = self.base_height_reward_scale * np.exp(-height_error_sq / self.base_height_sigma)
        reward += reward_height
        
        # =====================================================
        # 3. FORWARD VELOCITY REWARD (incentivize locomotion)
        # =====================================================
        robot_pos = root_pos[:, :2]  # (n_envs, 2)
        
        # Determine active waypoint for each environment
        target_positions = np.zeros((n_envs, 2), dtype=np.float32)
        for env_id in range(n_envs):
            if not smiley_triggered[env_id, 0]:
                target_positions[env_id] = self.smiley_positions[0]
            elif not smiley_triggered[env_id, 1]:
                target_positions[env_id] = self.smiley_positions[1]
            elif not smiley_triggered[env_id, 2]:
                target_positions[env_id] = self.smiley_positions[2]
            elif not hongbao_triggered[env_id, 0]:
                target_positions[env_id] = self.hongbao_positions[0]
            elif not hongbao_triggered[env_id, 1]:
                target_positions[env_id] = self.hongbao_positions[1]
            elif not hongbao_triggered[env_id, 2]:
                target_positions[env_id] = self.hongbao_positions[2]
            else:
                target_positions[env_id] = self.finish_zone_center
        
        # Direction to target (unit vector)
        to_target = target_positions - robot_pos
        dist_to_target = np.linalg.norm(to_target, axis=-1, keepdims=True)
        target_dir = np.where(dist_to_target > 0.01, to_target / dist_to_target, 0.0)
        
        # Forward velocity = projection of base_lin_vel onto target direction
        forward_vel = np.sum(base_lin_vel[:, :2] * target_dir, axis=-1)  # Scalar
        # Reward positive forward velocity, penalize backward movement
        reward_forward_vel = self.forward_velocity_reward_scale * np.clip(forward_vel, -0.5, 2.0)
        # Gate by height: no forward reward if lying down
        reward_forward_vel *= is_standing.astype(np.float32)
        reward += reward_forward_vel
        
        # =====================================================
        # 4. WAYPOINT PROGRESS REWARD (distance reduction)
        # =====================================================
        for env_id in range(n_envs):
            current_dist = np.linalg.norm(robot_pos[env_id] - target_positions[env_id])
            last_dist = info.get("last_waypoint_distance", np.full(n_envs, self.DEFAULT_WAYPOINT_DISTANCE))[env_id]
            progress = last_dist - current_dist
            reward[env_id] += np.clip(progress * 0.5, -0.1, 0.5)
            
            if "last_waypoint_distance" not in info:
                info["last_waypoint_distance"] = np.full(n_envs, self.DEFAULT_WAYPOINT_DISTANCE, dtype=np.float32)
            info["last_waypoint_distance"][env_id] = current_dist
        
        # =====================================================
        # 5. SPARSE TRIGGER REWARDS (competition elements)
        # =====================================================
        # Smiley bonuses (+4 each)
        for i in range(3):
            newly_triggered = smiley_triggered[:, i] & ~info.get(f"smiley_{i}_rewarded", np.zeros(n_envs, dtype=bool))
            reward += newly_triggered.astype(np.float32) * 4.0
            if f"smiley_{i}_rewarded" not in info:
                info[f"smiley_{i}_rewarded"] = np.zeros(n_envs, dtype=bool)
            info[f"smiley_{i}_rewarded"] |= newly_triggered
        
        # Hongbao bonuses (+2 each)
        for i in range(3):
            newly_triggered = hongbao_triggered[:, i] & ~info.get(f"hongbao_{i}_rewarded", np.zeros(n_envs, dtype=bool))
            reward += newly_triggered.astype(np.float32) * 2.0
            if f"hongbao_{i}_rewarded" not in info:
                info[f"hongbao_{i}_rewarded"] = np.zeros(n_envs, dtype=bool)
            info[f"hongbao_{i}_rewarded"] |= newly_triggered
        
        # Finish zone bonus (+20)
        newly_triggered_finish = finish_triggered & ~info.get("finish_rewarded", np.zeros(n_envs, dtype=bool))
        reward += newly_triggered_finish.astype(np.float32) * 20.0
        if "finish_rewarded" not in info:
            info["finish_rewarded"] = np.zeros(n_envs, dtype=bool)
        info["finish_rewarded"] |= newly_triggered_finish
        
        # =====================================================
        # 6. CELEBRATION BONUS (+2 for staying still at finish)
        # =====================================================
        time_elapsed_value = info.get("time_elapsed", 0.0)
        if isinstance(time_elapsed_value, (int, float)):
            time_elapsed = np.full(n_envs, time_elapsed_value, dtype=np.float32)
        else:
            time_elapsed = time_elapsed_value
        
        for env_id in range(n_envs):
            if finish_triggered[env_id] and not celebration_completed[env_id]:
                if celebration_start_time[env_id] >= 0:
                    celebration_time = time_elapsed[env_id] - celebration_start_time[env_id]
                    speed = np.linalg.norm(base_lin_vel[env_id, :2])
                    if speed < self.celebration_movement_threshold:
                        if celebration_time >= self.celebration_duration:
                            if not info.get("celebration_rewarded", np.zeros(n_envs, dtype=bool))[env_id]:
                                reward[env_id] += 2.0
                                celebration_completed[env_id] = True
                                if "celebration_rewarded" not in info:
                                    info["celebration_rewarded"] = np.zeros(n_envs, dtype=bool)
                                info["celebration_rewarded"][env_id] = True
                                self._log(f"[ENV {env_id}] Celebration completed! +2 points")
                    else:
                        celebration_start_time[env_id] = time_elapsed[env_id]

        info["celebration_start_time"] = celebration_start_time
        info["celebration_completed"] = celebration_completed
        
        # =====================================================
        # 7. STABILITY PENALTIES (strengthened to prevent lying)
        # =====================================================
        root_quat = self._body.get_pose(data)[:, 3:7]
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # STRENGTHENED orientation penalty (10x from original -0.05)
        orientation_penalty = (np.abs(roll) + np.abs(pitch)) * cfg.reward_config.scales.get("orientation", -0.5)
        reward += orientation_penalty
        
        # Vertical velocity penalty (penalize falling/bouncing)
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2]) * cfg.reward_config.scales.get("lin_vel_z", -0.5)
        reward += lin_vel_z_penalty
        
        # Angular velocity penalty (XY axes — prevents tumbling)
        ang_vel_xy = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        ang_vel_xy_penalty = np.sum(np.square(ang_vel_xy[:, :2]), axis=-1) * cfg.reward_config.scales.get("ang_vel_xy", -0.05)
        reward += ang_vel_xy_penalty
        
        # Action rate penalty (keep moderate — not too high to discourage movement)
        last_actions = info["current_actions"]
        current_actions = info.get("next_actions", last_actions)
        action_rate = np.sum(np.square(current_actions - last_actions), axis=-1)
        reward += action_rate * cfg.reward_config.scales.get("action_rate", -0.01)
        
        # Torque penalty (keep small — don't over-penalize necessary actuation)
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=-1) * cfg.reward_config.scales.get("torques", -1e-5)
        reward += torque_penalty
        
        # Joint velocity penalty
        joint_vel = self._body.get_joint_dof_vel(data)
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=-1) * cfg.reward_config.scales.get("dof_vel", -5e-5)
        reward += dof_vel_penalty
        
        return reward

    def _compute_terminated(self, data: mtx.SceneData, info: dict,
                            root_pos: np.ndarray, root_quat: np.ndarray,
                            root_vel: np.ndarray) -> np.ndarray:
        """
        终止条件（与section012完全一致）：
        1. 基座接触地面（摔倒）
        2. 姿态失控（roll/pitch > 45°）
        3. 越界（超出赛道范围）
        4. 高度过低（≤ 40%目标高度 = 坍塌/趴下）
        5. 关节速度异常（overflow / NaN / Inf）

        注意：episode超时由基类 _update_truncate 统一处理为 truncated。
        """
        n_envs = data.shape[0]
        steps = info.get("steps", np.zeros(n_envs, dtype=np.int32))
        in_grace = steps < GRACE_PERIOD_STEPS

        # 1. 基座接触地面 — 仅在传感器可用时检测
        if not self._base_contact_available:
            base_contact = np.zeros(n_envs, dtype=bool)
        else:
          try:
            base_contact_val = self._model.get_sensor_value(self._base_contact_sensor, data)
            if base_contact_val.ndim == 0:
                base_contact = np.full(n_envs, float(base_contact_val) > 0.01, dtype=bool)
            elif base_contact_val.ndim == 1:
                if base_contact_val.shape[0] == n_envs:
                    base_contact = (base_contact_val > 0.01).astype(bool)
                else:
                    base_contact = np.full(n_envs,
                        np.linalg.norm(base_contact_val) > 0.01, dtype=bool)
            else:
                base_contact = (np.linalg.norm(base_contact_val, axis=-1) > 0.01
                                ).flatten()[:n_envs].astype(bool)
          except Exception:
            base_contact = np.zeros(n_envs, dtype=bool)

        # 2. 姿态失控
        roll, pitch = self._quat_to_roll_pitch(root_quat)
        attitude_fail = (np.abs(roll) > FALL_THRESHOLD_ROLL_PITCH) | (np.abs(pitch) > FALL_THRESHOLD_ROLL_PITCH)

        # 3. 越界检测
        out_of_bounds = (
            (root_pos[:, 0] < self.boundary_x_min) |
            (root_pos[:, 0] > self.boundary_x_max) |
            (root_pos[:, 1] < self.boundary_y_min) |
            (root_pos[:, 1] > self.boundary_y_max)
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

        # 综合：物理终止条件在宽限期内不生效
        physical_termination = (
            base_contact | attitude_fail | out_of_bounds |
            height_too_low | vel_overflow | vel_nan_inf
        ) & ~in_grace

        terminated = physical_termination

        # 物理终止施加惩罚（通过info传递，由update_state叠加到reward）
        info["termination_penalty"] = np.where(
            physical_termination, TERMINATION_PENALTY, 0.0
        ).astype(np.float32)

        return terminated

    # ==================== 可视化辅助 ====================

    def _update_visualization(self, data: mtx.SceneData, root_pos: np.ndarray,
                              root_vel: np.ndarray, info: dict):
        """合并可视化更新，只调用一次 forward_kinematic（与section012一致）"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()

        # ---- target marker ----
        for env_idx in range(num_envs):
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                float(self.finish_zone_center[0]), float(self.finish_zone_center[1]), 0.0
            ]

        # ---- heading arrows ----
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            desired_vel_xy = info.get("desired_vel_xy", np.zeros((num_envs, 2), dtype=np.float32))
            base_lin_vel_xy = root_vel[:, :2]
            arrow_offset = 0.5

            for env_idx in range(num_envs):
                arrow_height = root_pos[env_idx, 2] + arrow_offset

                # 当前运动方向箭头
                cur_v = base_lin_vel_xy[env_idx]
                cur_yaw = np.arctan2(cur_v[1], cur_v[0]) if np.linalg.norm(cur_v) > 1e-3 else 0.0
                robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
                qn = np.linalg.norm(robot_arrow_quat)
                robot_arrow_quat = robot_arrow_quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
                all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                    np.array([root_pos[env_idx, 0], root_pos[env_idx, 1], arrow_height], dtype=np.float32),
                    robot_arrow_quat
                ])

                # 期望运动方向箭头
                des_v = desired_vel_xy[env_idx]
                des_yaw = np.arctan2(des_v[1], des_v[0]) if np.linalg.norm(des_v) > 1e-3 else 0.0
                desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
                qn = np.linalg.norm(desired_arrow_quat)
                desired_arrow_quat = desired_arrow_quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
                all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                    np.array([root_pos[env_idx, 0], root_pos[env_idx, 1], arrow_height], dtype=np.float32),
                    desired_arrow_quat
                ])

        # 一次性写入 + forward_kinematic
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)

    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角 → 四元数 [qx, qy, qz, qw]"""
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

    def reset(
        self,
        data: mtx.SceneData,
        done: np.ndarray = None,
    ) -> tuple[np.ndarray, dict]:
        """重置环境

        当 done 不为 None 时，仅重置 done=True 的环境（部分重置）。
        当基类 _reset_done_envs 调用时 data 已切片，done=None，全量重置即可。
        """
        cfg = self._cfg
        num_envs = data.shape[0]

        # ===== 部分重置支持（与section012一致）=====
        if done is not None:
            reset_mask = done.astype(bool)
            if not np.any(reset_mask):
                # 无需重置，直接返回当前观测
                obs = np.zeros((num_envs, self.observation_space.shape[0]), dtype=np.float32)
                info = {
                    "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
                }
                return obs, info
            reset_indices = np.where(reset_mask)[0]
            num_reset = len(reset_indices)
        else:
            reset_indices = np.arange(num_envs)
            num_reset = num_envs

        # ===== 在 START zone 随机生成位置（X轴随机，Y固定）=====
        spawn_x = np.random.uniform(-4.0, 4.0, num_reset)  # 留1m边距，避免平台边缘出生
        spawn_y = np.full(num_reset, self.start_zone_center[1], dtype=np.float32)
        spawn_z = cfg.init_state.pos[2]

        # 初始朝向：面向+Y方向（朝终点），小范围随机 ±30°
        spawn_yaw = np.random.uniform(np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6, num_reset)
        spawn_quat = np.array([self._euler_to_quat(0, 0, yaw) for yaw in spawn_yaw])

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))

        # 使用命名索引设置基座位姿
        dof_pos[:, self._base_pos_start:self._base_pos_end] = np.column_stack(
            [spawn_x, spawn_y, np.full(num_reset, spawn_z, dtype=np.float32)]
        )
        dof_pos[:, self._base_quat_start:self._base_quat_end] = spawn_quat

        # 归一化所有四元数
        for env_idx in range(num_reset):
            # 基座四元数
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            qn = np.linalg.norm(quat)
            dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = (
                quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
            )

            # 箭头四元数（与section012一致）
            if self._robot_arrow_body is not None:
                for start, end in [
                    (self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end),
                    (self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end),
                ]:
                    if end <= len(dof_pos[env_idx]):
                        aq = dof_pos[env_idx, start:end]
                        aqn = np.linalg.norm(aq)
                        dof_pos[env_idx, start:end] = (
                            aq / aqn if aqn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
                        )

        # ===== 写入物理状态（支持部分重置）=====
        if done is not None:
            # 仅重置子集 — 保留非 done 环境的 dof_pos 和 dof_vel
            all_dof_pos = data.dof_pos.copy()
            all_dof_vel = data.dof_vel.copy() if hasattr(data, 'dof_vel') else np.zeros_like(all_dof_pos)
            for local_i, global_i in enumerate(reset_indices):
                all_dof_pos[global_i] = dof_pos[local_i]
                all_dof_vel[global_i] = self._init_dof_vel
            data.reset(self._model)
            data.set_dof_vel(all_dof_vel)
            data.set_dof_pos(all_dof_pos, self._model)
        else:
            data.reset(self._model)
            data.set_dof_vel(dof_vel)
            data.set_dof_pos(dof_pos, self._model)

        self._model.forward_kinematic(data)

        # 重置触发标志
        if done is None and num_envs == self._num_envs:
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

        # ===== 构建初始信息（补全缺失的 key，与section012一致）=====
        info = {
            # 动作
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            # 计步与计时
            "steps": np.zeros(num_envs, dtype=np.int32),
            "time_elapsed": np.zeros(num_envs, dtype=np.float32),
            "last_waypoint_distance": np.full(num_envs, self.DEFAULT_WAYPOINT_DISTANCE, dtype=np.float32),
            # 物理
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            # 竞赛触发状态
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

        # 崎岖地形额外缓冲区
        if self._rough_terrain:
            info["feet_air_time"] = np.zeros((num_envs, self._num_feet), dtype=np.float32)
            info["first_contact_air_time"] = np.zeros((num_envs, self._num_feet), dtype=np.float32)
            info["foot_contacts"] = np.zeros((num_envs, self._num_feet), dtype=np.float32)
            info["foot_forces_body"] = np.zeros((num_envs, 12), dtype=np.float32)
            info["state_history"] = None  # 将在 _compute_observation 中首次填充

        if done is not None:
            info["env_ids"] = np.where(done)[0]

        obs = np.zeros((num_envs, self.observation_space.shape[0]), dtype=np.float32)
        reward = np.zeros((num_envs,), dtype=np.float32)
        terminated = np.zeros((num_envs,), dtype=bool)
        truncated = np.zeros((num_envs,), dtype=bool)
        state = NpEnvState(data=data, obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

        # 计算初始观测
        state, root_pos, root_quat, root_vel = self._compute_observation(data, state)

        # 防止 NaN/Inf
        state.obs = np.nan_to_num(state.obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # 可视化更新（与section012一致：reset末尾也调用一次）
        self._update_visualization(data, root_pos, root_vel, state.info)

        self._log(f"[RESET] Spawned {num_reset} robots at START zone")

        return state.obs, state.info

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """更新环境状态（观测、奖励、终止）— 与section012结构一致"""
        data = state.data
        info = state.info

        # 1. 计算观测（纯观测计算，不包含奖励/终止/可视化）
        state, root_pos, root_quat, root_vel = self._compute_observation(data, state)

        # 2. 更新触发状态（smileys, hongbaos, finish）
        self._update_trigger_states(root_pos, info)

        # 3. 计算奖励
        velocity_commands = info.get("velocity_commands",
                                     np.zeros((data.shape[0], 3), dtype=np.float32))
        reward = self._compute_reward(data, info, velocity_commands, root_pos, root_vel)

        # 4. 计算终止条件
        terminated = self._compute_terminated(data, info, root_pos, root_quat, root_vel)

        # 5. 将终止惩罚叠加到奖励中（与section012一致）
        termination_penalty = info.get("termination_penalty",
                                        np.zeros(data.shape[0], dtype=np.float32))
        reward = reward + termination_penalty

        # 6. 更新时间
        info["time_elapsed"] = info.get(
            "time_elapsed", np.zeros(data.shape[0], dtype=np.float32)
        ) + self._cfg.sim_dt

        # 7. 可视化更新放到最后（与section012一致）
        info["desired_vel_xy"] = info.get("desired_vel_xy",
                                           np.zeros((data.shape[0], 2), dtype=np.float32))
        self._update_visualization(data, root_pos, root_vel, info)

        state.reward = reward
        state.terminated = terminated
        return state

