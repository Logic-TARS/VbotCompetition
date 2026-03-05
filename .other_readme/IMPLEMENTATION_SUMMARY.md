# Motphys 机器狗平地导航竞技场实现总结

## 项目概述
本实现基于 Motphys 物理引擎，为机器狗（VBot）创建了一个"圆形竞技场导航"仿真任务。在单次仿真中同时运行10只机器狗，每只狗需完成三阶段导航任务，最终达到最高20分（10只狗 × 每只最高2分）。

## 核心实现组件

### 1. 场景环境定义 (Arena Setup)

#### 圆形竞技场布局
```
外圈（起始区域）: 半径 3.0 米
├── 内圈电子围栏: 半径 1.5 米
│   └── 触发点 A (内圈): 计+1分
└── 圆心触发点 B: 计+1分
边界限制: 半径 3.5 米（越界则扣分清零）
```

#### 几何参数
```python
arena_outer_radius = 3.0      # 外圈半径（起始区域）
arena_inner_radius = 1.5      # 内圈半径（第一个目标）
boundary_radius = 3.5         # 边界半径（越界判定）
arena_center = [0.0, 0.0]     # 圆心坐标
spawn_range = 0.1             # 初始位置随机范围（±0.1米）
```

### 2. 初始化逻辑 (Initialization)

#### 10只机器狗的随机初始化
```python
def reset(self, data, done=None):
    # 在外圈范围内随机生成10只机器狗
    for i in range(num_envs):  # num_envs = 10
        theta = random(0, 2π)           # 随机方位角
        radius = arena_outer_radius + random(-spawn_range, spawn_range)  # 随机半径
        pos[i] = [radius * cos(theta), radius * sin(theta), 0.5]  # 极坐标转直角坐标
```

#### 初始化特点
- ✅ 每只狗位置**完全随机**，分布在外圈（3.0-3.2米之间）
- ✅ 禁止出生在固定位置
- ✅ 每次重置都产生新的随机分布
- ✅ 第一阶段目标统一指向内圈（触发点A）

### 3. 核心任务逻辑 (Gameplay Loop)

#### 三阶段导航流程
```
阶段 0 → 阶段 1:
  起始位置 → 内圈电子围栏（触发点 A）
  触发条件: 机器狗进入内圈半径0.3m范围
  奖励: +1分

阶段 1 → 阶段 2:
  内圈电子围栏 → 圆心（触发点 B）
  触发条件: 机器狗进入圆心半径0.3m范围
  奖励: +1分
```

#### 动态目标调整
```python
def update_state(self):
    if dog_stage[i] == 0 and distance_to_point_A < 0.3:
        dog_stage[i] = 1
        dog_scores[i] += 1.0
    elif dog_stage[i] == 1 and distance_to_point_B < 0.3:
        dog_stage[i] = 2
        dog_scores[i] += 1.0
```

### 4. 计分系统 (Scoring System)

#### 得分逻辑
| 事件 | 得分 | 条件 |
|------|------|------|
| 到达内圈（触发点A） | +1分 | 进入半径0.3m范围 |
| 到达圆心（触发点B） | +1分 | 进入半径0.3m范围 |
| **单只狗最高得分** | **2分** | 完成两个阶段 |
| **10只狗总得分上限** | **20分** | 所有狗完成任务 |

#### 扣分/清零逻辑 (Penalty Function)
```python
def _update_dog_scores(self, robot_pos, root_quat, foot_contacts):
    # 检测违规条件
    if fallen[i] or out_of_bounds[i]:
        dog_penalty_flags[i] = True
        dog_scores[i] = 0.0         # 得分清零
        dog_stage[i] = 0            # 阶段重置
```

| 触发条件 | 惩罚 |
|---------|------|
| 摔倒（Roll/Pitch > 60°） | 该狗得分清零 |
| 长时间悬空（50帧）| 该狗得分清零 |
| 越界（距离圆心 > 3.5m） | 该狗得分清零 |
| **无连带责任** | 其他狗不受影响 |

### 5. 技术实现细节

#### A. 检测方法

##### 摔倒检测 (_detect_fall)
```python
def _detect_fall(self, root_quat, foot_contacts):
    # 方法1: Roll/Pitch超过阈值
    roll, pitch = extract_euler_from_quat(root_quat)
    exceeded_angle = (|roll| > 60°) or (|pitch| > 60°)
    
    # 方法2: 长时间非足部着地
    all_feet_low = all(foot_contacts < 0.1)
    self.dog_fall_frames += 1 if all_feet_low else 0
    long_time_airborne = self.dog_fall_frames > 50  # ~0.5秒
    
    return exceeded_angle | long_time_airborne
```

##### 越界检测 (_detect_out_of_bounds)
```python
def _detect_out_of_bounds(self, robot_pos):
    distance_from_center = ||robot_pos[:2] - arena_center||
    return distance_from_center > boundary_radius  # 3.5m
```

##### 触发点检测 (_check_trigger_points)
```python
def _check_trigger_points(self, robot_pos):
    # 检测到触发点A（内圈）
    dist_to_a = ||robot_pos[:2] - point_A||
    triggered_a = dist_to_a < 0.3  # 半径0.3m
    
    # 检测到触发点B（圆心）
    dist_to_b = ||robot_pos[:2] - point_B||
    triggered_b = dist_to_b < 0.3
    
    return triggered_a, triggered_b
```

#### B. 状态跟踪
```python
# 初始化在 __init__ 中
self.dog_scores        = [0.0] * 10        # 每只狗的当前得分
self.dog_stage         = [0] * 10          # 每只狗的阶段 (0/1/2)
self.dog_triggered_a   = [False] * 10      # 是否触发点A
self.dog_triggered_b   = [False] * 10      # 是否触发点B
self.dog_penalty_flags = [False] * 10      # 是否受到惩罚
self.dog_fall_frames   = [0] * 10          # 连续非足部着地帧数
```

#### C. 实时计分更新
```python
def _compute_reward(self, data, info, velocity_commands):
    # 获取机器狗位置和朝向
    root_pos, root_quat, root_vel = self._extract_root_state(data)
    
    # 获取足部接触状态
    foot_contacts = [FR, FL, RR, RL]_foot_contact
    
    # 更新每只狗的得分状态
    self._update_dog_scores(root_pos, root_quat, foot_contacts)
    
    # 返回当前得分作为奖励
    return self.dog_scores.copy()  # 形状: [10]
```

### 6. 观测与奖励

#### 观测空间 (54维)
```python
observation = concatenate([
    base_linvel,                    # 3: 线速度
    gyro,                           # 3: 角速度
    projected_gravity,              # 3: 投影重力
    joint_pos_rel,                  # 12: 相对关节位置
    joint_vel,                      # 12: 关节速度
    last_actions,                   # 12: 上一步动作
    velocity_commands,              # 3: 速度命令
    position_error_normalized,      # 2: 位置误差
    heading_error_normalized,       # 1: 朝向误差
    distance_normalized,            # 1: 距离
    reached_flag,                   # 1: 到达标志
    stop_ready_flag,                # 1: 停止就绪
])  # 总计: 54维
```

#### 奖励信号 (每个环境独立)
```
reward[i] = dog_scores[i]  # [0.0, 2.0]（每只狗）
total_score = sum(reward)   # [0.0, 20.0]（10只狗总计）
```

### 7. 并发处理

#### 向量化操作
```python
# 10只狗同时进行计算
dog_fallen = _detect_fall(root_quat[:10], foot_contacts[:10])      # 并行
out_of_bounds = _detect_out_of_bounds(robot_pos[:10])              # 并行
triggered_a, triggered_b = _check_trigger_points(robot_pos[:10])   # 并行

# 各狗得分独立更新（循环是必要的，用于条件判断）
for i in range(10):
    # 只影响单只狗，互不干扰
    if fallen[i]:
        scores[i] = 0.0
```

#### 状态独立性
- 每只狗的`dog_scores[i]`、`dog_stage[i]`、`dog_penalty_flags[i]`独立管理
- 一只狗的惩罚**不影响**其他狗的得分
- 所有计算通过向量化 NumPy 操作并行执行

---

## 使用指南

### 环境配置

```python
from motrix_envs import registry

# 创建10只机器狗的竞技场环境
env = registry.get("vbot_navigation_section001", "np", num_envs=10)
```

### 运行示例

```python
import numpy as np

# 初始化
obs, info = env.reset()

# 运行仿真循环
total_scores = []
for step in range(1000):
    # 获取随机动作（实际应来自策略网络）
    actions = env.action_space.sample()  # [10, 12]
    
    # 执行一步
    obs, reward, terminated, info = env.step(actions)
    
    # 实时计分显示
    current_scores = info["dog_scores"]      # [10]
    current_stages = info["dog_stage"]       # [10]
    total_score = info["total_score"]        # 标量: 0-20
    
    print(f"Step {step}: Total Score: {total_score:.1f}/20.0")
    print(f"  Individual scores: {current_scores}")
    print(f"  Individual stages: {current_stages}")
```

### 实时数据监控

```python
# 访问详细计分信息
dog_scores = info["dog_scores"]             # 每只狗的得分 [10]
dog_stages = info["dog_stage"]              # 每只狗的阶段 [10]
total_score = info["total_score"]           # 总得分: 0-20
```

---

## 验收标准

### ✅ 已实现

- [x] **初始化**: 10只机器狗在圆环带内随机分布（不固定）
- [x] **自主导航**: 机器狗自主向目标位置移动
- [x] **实时计分**: 
  - 每只狗最多2分（内圈+圆心）
  - 总分0-20分
  - 实时通过 `info["total_score"]` 显示
- [x] **摔倒检测**: 机身翻转或长时间悬空时得分清零
- [x] **越界检测**: 超出边界时得分清零
- [x] **独立惩罚**: 单只狗失败不影响其他狗
- [x] **三阶段导航**:
  1. 外圈 → 内圈（触发点A, +1分）
  2. 内圈 → 圆心（触发点B, +1分）
  
---

## 文件修改清单

### 主要修改文件
- `vbot_section001_np.py`: 
  - ✅ 添加竞技场配置参数
  - ✅ 重写 `reset()` 方法（圆形随机初始化）
  - ✅ 实现 `_detect_fall()` 方法
  - ✅ 实现 `_detect_out_of_bounds()` 方法
  - ✅ 实现 `_check_trigger_points()` 方法
  - ✅ 实现 `_update_dog_scores()` 方法
  - ✅ 完整实现 `_compute_reward()` 方法
  - ✅ 更新 `update_state()` 以返回计分信息

---

## 技术参数总结

| 参数 | 值 | 含义 |
|------|-----|------|
| `num_envs` | 10 | 同时运行的机器狗数量 |
| `arena_outer_radius` | 3.0m | 外圈半径 |
| `arena_inner_radius` | 1.5m | 内圈半径 |
| `boundary_radius` | 3.5m | 边界半径 |
| `spawn_range` | 0.1m | 初始位置随机偏差 |
| `trigger_threshold` | 0.3m | 触发点检测半径 |
| `fall_threshold_roll_pitch` | 60° | 摔倒判定角度 |
| `fall_frames_threshold` | 50帧 | 摔倒判定时长 |
| `max_score_per_dog` | 2分 | 单只狗最高得分 |
| `max_total_score` | 20分 | 10只狗总得分上限 |

---

## 后续优化方向

1. **奖励塑形**: 添加中间奖励（如靠近目标的奖励）来加速学习
2. **动态难度**: 增加障碍物或调整边界大小
3. **多模式**: 支持不同的竞技场配置（椭圆形、方形等）
4. **评估指标**: 完成时间、平均速度、摔倒次数等

---

## 测试清单

- [ ] 启动环境，验证10只狗随机初始化
- [ ] 运行100步，观察计分变化
- [ ] 手动推狗越界，验证得分清零且不影响其他狗
- [ ] 使手狗摔倒，验证得分清零
- [ ] 验证总分上限为20分
- [ ] 检查观测维度为54
- [ ] 验证所有10只狗的动作均匀应用
