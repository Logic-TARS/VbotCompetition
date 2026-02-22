# 训练环境配置

MotrixLab 提供了灵活的配置系统，允许用户自定义强化学习训练参数。本节介绍如何配置训练环境和强化学习算法参数。

## RL 训练配置 (PPOCfg)

训练配置定义了基于 PPO 算法的强化学习算法的参数。MotrixLab 现在支持为不同训练后端配置不同的参数。

### 完整配置示例

```python
@dataclass
class CompletePPOConfig(PPOCfg):
    """
    完整的强化学习训练配置示例
    包含了从基础到高级的所有配置参数
    """

    # ===== 基础训练参数 =====
    seed: Optional[int] = None         # 随机种子
    num_envs: int = 2048               # 训练时并行环境数量
    play_num_envs: int = 16            # 评估时并行环境数量
    max_env_steps: int = 2_048_000     # 最大训练步数
    check_point_interval: int = 1000   # 检查点保存间隔

    # ===== PPO算法核心参数 =====
    learning_rate: float = 3e-4        # 学习率
    rollouts: int = 32                 # 经验回放轮数
    learning_epochs: int = 2           # 每次更新的训练轮数
    mini_batches: int = 32             # 小批量数量
    discount_factor: float = 0.99      # 折扣因子
    lambda_param: float = 0.95         # GAE参数
    grad_norm_clip: float = 1.0        # 梯度裁剪

    # ===== PPO裁剪参数 =====
    ratio_clip: float = 0.2            # PPO裁剪比率
    value_clip: float = 0.2            # 价值裁剪
    clip_predicted_values: bool = True # 裁剪预测值

    # ===== 损失函数参数 =====
    entropy_loss_scale: float = 0.0    # 熵损失系数
    value_loss_scale: float = 2.0      # 价值损失系数
    kl_threshold: float = 0            # KL散度阈值

    # ===== 学习率调度器 =====
    learning_rate_scheduler_kl_threshold: float = 0.008  # 自适应学习率KL阈值

    # ===== 网络架构配置 =====
    # 小型网络（适合简单任务如 CartPole）
    # policy_hidden_layer_sizes: tuple[int, ...] = (128, 64)
    # value_hidden_layer_sizes: tuple[int, ...] = (128, 64)

    # 中型网络（默认配置，适合大部分任务）
    policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
    value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    # 大型网络（适合复杂任务如机器人控制）
    # policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    # value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    # ===== 网络共享配置 =====
    share_policy_value_features: bool = True  # 策略和价值网络共享特征提取层

    # ===== 训练控制参数 =====
    random_timesteps: int = 0          # 随机步数
    learning_starts: int = 0           # 开始学习的步数
    time_limit_bootstrap: bool = True  # 时间限制引导

    # ===== 奖励整形 =====
    rewards_shaper_scale: float = 1.0  # 奖励缩放因子
```

## 配置使用方法

### 1. 默认配置使用

```bash
# 使用代码中给定的配置
uv run scripts/train.py --env my-task

# 指定训练后端，系统会自动选择对应的后端配置
uv run scripts/train.py --env my-task --train-backend jax
uv run scripts/train.py --env my-task --train-backend torch
```

### 2. 命令行参数覆盖

```bash
# 覆盖支持的命令行参数
uv run scripts/train.py --env my-task \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np

# 系统会自动选择JAX后端对应的配置
```

### 3. 配置优先级

系统按以下优先级选择配置：

1. **后端特定配置**: 如果存在 `@rlcfg(env_name, backend="jax/torch")` 装饰的配置
2. **通用配置**: 如果存在 `@rlcfg(env_name)` 装饰的配置（无 backend 参数）

例如：

```python
# 最高优先级 - 后端特定配置
@rlcfg("my-task", backend="jax")
@dataclass
class MyTaskJAXCfg(PPOCfg):
    mini_batches: int = 4

# 次优先级 - 通用配置
@rlcfg("my-task")
@dataclass
class MyTaskRLCfg(PPOCfg):
    mini_batches: int = 32

# 当使用 --train-backend jax 时，系统会选择 MyTaskJAXCfg
# 当使用 --train-backend torch 时，系统会选择 MyTaskRLCfg
```

## SKRL 框架配置映射

在 MotrixLab 中，用户通过 `PPOCfg` 配置类设置参数，这些参数会被映射到 SKRL 框架的配置字典中。

### 用户可配置参数

| MotrixLab 配置类                       | SKRL 框架参数                                 | 说明                 |
| -------------------------------------- | --------------------------------------------- | -------------------- |
| `learning_rate`                        | `learning_rate`                               | 学习率               |
| `rollouts`                             | `rollouts`                                    | 经验回放轮数         |
| `learning_epochs`                      | `learning_epochs`                             | 训练轮数             |
| `mini_batches`                         | `mini_batches`                                | 小批量数量           |
| `discount_factor`                      | `discount_factor`                             | 折扣因子             |
| `grad_norm_clip`                       | `grad_norm_clip`                              | 梯度裁剪             |
| `lambda_param`                         | `lambda`                                      | GAE 参数             |
| `ratio_clip`                           | `ratio_clip`                                  | PPO 裁剪比率         |
| `value_clip`                           | `value_clip`                                  | 价值裁剪             |
| `clip_predicted_values`                | `clip_predicted_values`                       | 裁剪预测值           |
| `entropy_loss_scale`                   | `entropy_loss_scale`                          | 熵损失系数           |
| `value_loss_scale`                     | `value_loss_scale`                            | 价值损失系数         |
| `kl_threshold`                         | `kl_threshold`                                | KL 散度阈值          |
| `random_timesteps`                     | `random_timesteps`                            | 随机步数             |
| `learning_starts`                      | `learning_starts`                             | 开始学习的步数       |
| `time_limit_bootstrap`                 | `time_limit_bootstrap`                        | 时间限制引导         |
| `learning_rate_scheduler_kl_threshold` | `learning_rate_scheduler_kwargs.kl_threshold` | 自适应学习率 KL 阈值 |
| `check_point_interval`                 | `experiment.write_interval`                   | 日志写入间隔         |
| `check_point_interval`                 | `experiment.checkpoint_interval`              | 检查点保存间隔       |
| `rewards_shaper_scale`                 | `rewards_shaper`                              | 奖励缩放函数         |

### 预处理器参数

| SKRL 框架参数        | 类型                  | 说明       |
| -------------------- | --------------------- | ---------- |
| `state_preprocessor` | RunningStandardScaler | 状态标准化 |
| `value_preprocessor` | RunningStandardScaler | 价值标准化 |

### 配置层次总结

```
用户配置类 (PPOCfg)
    ↓ 后端特定选择
后端配置 (JAX/Torch)
    ↓ 参数映射
SKRL 框架配置字典
    ↓ 传递给
PPO Agent
    ↓ 执行
强化学习训练
```

这种设计允许用户：

1. 通过简单的配置类来控制复杂的训练参数
2. 为不同训练后端配置不同的参数以获得最佳性能
3. 保持与 SKRL 框架的完全兼容性
