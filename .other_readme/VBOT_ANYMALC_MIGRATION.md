# VBot 采用 AnymalC 初始化方式 - 迁移总结

## ✅ 完成的改动

### 1. **移除竞技场初始化代码**
```python
# ❌ 删除了以下代码：
- 竞技场参数（arena_outer_radius, arena_inner_radius, boundary_radius）
- 三阶段目标位置（target_point_a, target_point_b）
- 摔倒判定标准（fall_threshold_roll_pitch, fall_contact_threshold）
- 状态追踪数组（dog_scores, dog_stage, dog_triggered_a/b, etc.）
```

### 2. **简化 `_init_buffer()` 方法**
**之前**: ~60 行（包含竞技场配置）  
**之后**: ~20 行（仅保留基础控制参数）

```python
def _init_buffer(self):
    """初始化缓存和参数"""
    cfg = self._cfg
    self.default_angles = np.zeros(self._num_action, dtype=np.float32)
    
    # 归一化系数
    self.commands_scale = np.array(...)
    
    # 设置默认关节角度
    for i in range(self._model.num_actuators):
        for name, angle in cfg.init_state.default_joint_angles.items():
            if name in self._model.actuator_names[i]:
                self.default_angles[i] = angle
    
    self._init_dof_pos[-self._num_action:] = self.default_angles
    self.action_filter_alpha = 0.3
```

### 3. **修改 `reset()` 方法使用固定初始位置**

**之前（竞技场模式）**:
```python
# 极坐标随机生成 num_envs 个不同位置
robot_init_xy = np.zeros((num_envs, 2), dtype=np.float32)
for i in range(num_envs):
    theta = np.random.uniform(0, 2 * np.pi)
    radius = self.arena_outer_radius + np.random.uniform(-0.1, 0.1)
    robot_init_xy[i, 0] = radius * np.cos(theta)
    robot_init_xy[i, 1] = radius * np.sin(theta)
```

**之后（AnymalC 方式）**:
```python
# 所有环境使用相同的固定初始位置
robot_init_pos = np.tile(cfg.init_state.pos, (num_envs, 1))
dof_pos[:, 3:6] = robot_init_pos
```

### 4. **修改目标位置设定**

**之前**:
```python
# 竞技场模式：三阶段目标
target_positions = np.tile(self.target_point_a, (num_envs, 1))
```

**之后**:
```python
# 标准导航：使用初始配置位置
target_positions = np.tile(cfg.init_state.pos[:2], (num_envs, 1))
```

### 5. **完全重写 `_compute_reward()` 方法**

**之前**:
```python
# 竞技场计分模式
self._update_dog_scores(root_pos, root_quat, foot_contacts)
reward = self.dog_scores.copy()  # 返回分数 [num_envs]
```

**之后**:
```python
# 标准导航奖励（与AnymalC一致）
reward = (
    1.5 * tracking_lin_vel          # 线速度追踪
    + 0.3 * tracking_ang_vel        # 角速度追踪
    - 0.1 * orientation_penalty      # 朝向稳定性
    - 0.00001 * torque_penalty       # 力矩惩罚
    - 0.001 * action_rate_penalty    # 动作变化惩罚
)
```

### 6. **简化 reset() 返回信息字典**

**之前**:
```python
info = {
    "pose_commands": pose_commands,
    "last_actions": ...,
    "steps": ...,
    "current_actions": ...,
    "filtered_actions": ...,
    "ever_reached": ...,
    "min_distance": ...,
    "last_dof_vel": ...,
    "contacts": ...,
    "dog_scores": ...,        # ❌ 删除
    "dog_stage": ...,         # ❌ 删除
    "total_score": ...,       # ❌ 删除
}
```

**之后**:
```python
info = {
    "pose_commands": pose_commands,
    "last_actions": ...,
    "current_actions": ...,
    "filtered_actions": ...,
    "ever_reached": ...,
    "min_distance": ...,
}
```

### 7. **移除 update_state() 中的竞技场计分更新**

**之前**:
```python
# ===== 更新竞技场计分信息到 info =====
state.info["dog_scores"] = self.dog_scores.copy()
state.info["dog_stage"] = self.dog_stage.copy()
state.info["total_score"] = float(np.sum(self.dog_scores))
```

**之后**: 删除了这些行

---

## 📊 代码量对比

| 指标 | 之前 | 之后 | 变化 |
|------|------|------|------|
| 总行数 | 880 行 | ~780 行 | ↓ 11% |
| `_init_buffer()` | ~60 行 | ~20 行 | ↓ 67% |
| `reset()` | ~200 行 | ~120 行 | ↓ 40% |
| `_compute_reward()` | ~30 行 | ~45 行 | ↑ 50% |
| 状态数组 | 6 个 | 0 个 | ✅ 移除 |

---

## 🔄 架构对比

### 之前（竞技场模式）
```
┌─────────────────────────────────┐
│  VBot 环境 (竞技场模式)          │
├─────────────────────────────────┤
│ • num_envs 个不同起点           │
│ • 圆形竞技场配置               │
│ • 三阶段导航逻辑               │
│ • 独立计分系统 (0-20 分)        │
│ • 摔倒/越界检测                 │
│ • 复杂的奖励函数               │
└─────────────────────────────────┘
```

### 之后（AnymalC 方式）
```
┌─────────────────────────────────┐
│  VBot 环境 (标准导航)           │
├─────────────────────────────────┤
│ • num_envs 个相同起点           │
│ • 标准平地环境                 │
│ • 简单导航目标                 │
│ • 速度追踪奖励                 │
│ • 朝向稳定性奖励               │
│ • 与 AnymalC 一致的奖励        │
└─────────────────────────────────┘
```

---

## ✨ 主要改变

| 维度 | 之前 | 之后 |
|------|------|------|
| **初始位置** | 动态极坐标随机 | 固定（cfg.init_state.pos） |
| **环境数量** | num_envs 个不同位置 | num_envs 个相同位置 |
| **计分系统** | ✅ 完整的竞技场计分 | ❌ 移除（改用标准奖励） |
| **状态数组** | 6 个（dog_scores 等） | 0 个（全部删除） |
| **复杂度** | 高（竞技场逻辑） | 低（标准导航） |
| **与 AnymalC 兼容性** | ❌ 不兼容 | ✅ 兼容 |

---

## 🧪 测试结果

### 编译测试
```bash
$ python -m py_compile motrix_envs/src/motrix_envs/navigation/vbot/vbot_section001_np.py
✅ 编译成功（无语法错误）
```

### 运行测试
```bash
$ uv run scripts/view.py --env vbot_navigation_section001 --num-envs 5
✅ 环境成功创建（无运行时错误）
```

---

## 📝 保留的代码（未使用但保留框架）

为了保持代码的完整性，以下方法仍然保留在代码中，但不再被调用：

- `_detect_fall()` - 摔倒检测逻辑
- `_detect_out_of_bounds()` - 越界检测逻辑
- `_check_trigger_points()` - 触发点检测逻辑
- `_update_dog_scores()` - 计分更新逻辑

这些方法可以在需要时重新启用（例如未来实现竞技场模式 v2）。

---

## 🎯 结论

VBot 环境现在采用 **AnymalC 的初始化和奖励方式**：

✅ **所有环境从相同位置开始**  
✅ **简化的奖励函数（速度追踪 + 稳定性）**  
✅ **与 AnymalC 架构兼容**  
✅ **代码行数减少 ~11%**  
✅ **无语法错误，可正常运行**  

---

## 🚀 运行命令

```bash
# 查看 VBot 环境（5个并行环境）
python scripts/view.py --env vbot_navigation_section001 --num-envs 5

# 训练 VBot 环境
python scripts/train.py --env vbot_navigation_section001 --num-envs 16
```

---

**迁移完成时间**: 2026-02-07  
**状态**: ✅ 完成并测试通过

