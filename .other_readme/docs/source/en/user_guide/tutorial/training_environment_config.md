# Training Environment Configuration

MotrixLab provides a flexible configuration system that allows users to customize reinforcement learning training parameters. This section introduces how to configure training environments and reinforcement learning algorithm parameters.

## RL Training Configuration (PPOCfg)

Training configuration defines parameters for reinforcement learning algorithms. MotrixLab now supports configuring different parameters for different training backends.

### Basic Training Configuration

```python
from dataclasses import dataclass
from motrix_rl.skrl.cfg import PPOCfg
from motrix_rl import registry

# Universal configuration (applies to all backends)
@registry.rlcfg("my-task")
@dataclass
class MyTaskRLCfg(PPOCfg):
    # Environment parameters
    num_envs: int = 2048              # Number of parallel environments during training
    play_num_envs: int = 16           # Number of parallel environments during evaluation

    # PPO algorithm parameters
    learning_rate: float = 3e-4       # Learning rate
    rollouts: int = 32                # Experience replay rounds
    learning_epochs: int = 10         # Number of epochs per update
    mini_batches: int = 32            # Number of mini-batches
    discount_factor: float = 0.99     # Discount factor
    grad_norm_clip: float = 1.0       # Gradient clipping

    # Network structure parameters
    policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)  # Policy network hidden layers
    value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)   # Value network hidden layers

    # Training control parameters
    max_env_steps: int = 1_000_000    # Maximum environment steps
    check_point_interval: int = 10_000 # Checkpoint interval
```

### Different Backend Configurations

```python
# JAX backend specific configuration
@registry.rlcfg("my-task", backend="jax")
@dataclass
class MyTaskJAXCfg(PPOCfg):
    # JAX optimized configuration
    learning_rate: float = 2e-4       # JAX backend uses smaller learning rate
    mini_batches: int = 4             # JAX supports large batches, fewer mini-batches
    learning_epochs: int = 4          # Training rounds
    num_envs: int = 2048              # More parallel environments

# Torch backend specific configuration
@registry.rlcfg("my-task", backend="torch")
@dataclass
class MyTaskTorchCfg(PPOCfg):
    # Torch optimized configuration
    learning_rate: float = 1e-4       # Torch backend uses even smaller learning rate
    mini_batches: int = 32            # Torch needs more mini-batches
    learning_epochs: int = 2          # Fewer training rounds
    num_envs: int = 1024              # Fewer parallel environments
```

### Complete Configuration Example

```python
@dataclass
class CompletePPOConfig(PPOCfg):
    """
    Complete reinforcement learning training configuration example
    Contains all configuration parameters from basic to advanced
    """

    # ===== Basic Training Parameters =====
    seed: Optional[int] = None         # Random seed
    num_envs: int = 2048               # Number of parallel environments during training
    play_num_envs: int = 16            # Number of parallel environments during evaluation
    max_env_steps: int = 2_048_000     # Maximum training steps
    check_point_interval: int = 1000   # Checkpoint save interval

    # ===== PPO Algorithm Core Parameters =====
    learning_rate: float = 3e-4        # Learning rate
    rollouts: int = 32                 # Experience replay rounds
    learning_epochs: int = 2           # Number of training rounds per update
    mini_batches: int = 32             # Number of mini-batches
    discount_factor: float = 0.99      # Discount factor
    lambda_param: float = 0.95         # GAE parameter
    grad_norm_clip: float = 1.0        # Gradient clipping

    # ===== PPO Clipping Parameters =====
    ratio_clip: float = 0.2            # PPO clipping ratio
    value_clip: float = 0.2            # Value clipping
    clip_predicted_values: bool = True # Clip predicted values

    # ===== Loss Function Parameters =====
    entropy_loss_scale: float = 0.0    # Entropy loss coefficient
    value_loss_scale: float = 2.0      # Value loss coefficient
    kl_threshold: float = 0            # KL divergence threshold

    # ===== Learning Rate Scheduler =====
    learning_rate_scheduler_kl_threshold: float = 0.008  # Adaptive learning rate KL threshold

    # ===== Network Architecture Configuration =====
    # Small network (suitable for simple tasks like CartPole)
    # policy_hidden_layer_sizes: tuple[int, ...] = (128, 64)
    # value_hidden_layer_sizes: tuple[int, ...] = (128, 64)

    # Medium network (default configuration, suitable for most tasks)
    policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
    value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    # Large network (suitable for complex tasks like robot control)
    # policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    # value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    # ===== Network Sharing Configuration =====
    share_policy_value_features: bool = True  # Policy and value networks share feature extraction layers

    # ===== Training Control Parameters =====
    random_timesteps: int = 0          # Random timesteps
    learning_starts: int = 0           # Timesteps to start learning
    time_limit_bootstrap: bool = True  # Time limit bootstrap

    # ===== Reward Shaping =====
    rewards_shaper_scale: float = 1.0  # Reward scaling factor
```

## Configuration Usage Methods

### 1. Default Configuration Usage

```bash
# Use configuration given in code
uv run scripts/train.py --env my-task

# Specify training backend, system will automatically select corresponding backend configuration
uv run scripts/train.py --env my-task --train-backend jax
uv run scripts/train.py --env my-task --train-backend torch
```

### 2. Command Line Parameter Override

```bash
# Override supported command line parameters
uv run scripts/train.py --env my-task \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np

# System will automatically select JAX backend configuration
```

### 3. Configuration Priority

System selects configuration in the following priority:

1. **Backend-specific configuration**: If there is a configuration decorated with `@rlcfg(env_name, backend="jax/torch")`
2. **Universal configuration**: If there is a configuration decorated with `@rlcfg(env_name)` (no backend parameter)
3. **Configuration override**: Command line `cfg_override` parameter

For example:

```python
# Highest priority - backend specific configuration
@rlcfg("my-task", backend="jax")
@dataclass
class MyTaskJAXCfg(PPOCfg):
    mini_batches: int = 4

# Second priority - universal configuration
@rlcfg("my-task")
@dataclass
class MyTaskRLCfg(PPOCfg):
    mini_batches: int = 32

# When using --train-backend jax, system will select MyTaskJAXCfg
# When using --train-backend torch, system will select MyTaskRLCfg
```

#### User Configurable Parameters

| MotrixLab Configuration Class          | SKRL Framework Parameter                      | Description                         |
| -------------------------------------- | --------------------------------------------- | ----------------------------------- |
| `learning_rate`                        | `learning_rate`                               | Learning rate                       |
| `rollouts`                             | `rollouts`                                    | Experience replay rounds            |
| `learning_epochs`                      | `learning_epochs`                             | Training rounds                     |
| `mini_batches`                         | `mini_batches`                                | Number of mini-batches              |
| `discount_factor`                      | `discount_factor`                             | Discount factor                     |
| `grad_norm_clip`                       | `grad_norm_clip`                              | Gradient clipping                   |
| `lambda_param`                         | `lambda`                                      | GAE parameter                       |
| `ratio_clip`                           | `ratio_clip`                                  | PPO clipping ratio                  |
| `value_clip`                           | `value_clip`                                  | Value clipping                      |
| `clip_predicted_values`                | `clip_predicted_values`                       | Clip predicted values               |
| `entropy_loss_scale`                   | `entropy_loss_scale`                          | Entropy loss coefficient            |
| `value_loss_scale`                     | `value_loss_scale`                            | Value loss coefficient              |
| `kl_threshold`                         | `kl_threshold`                                | KL divergence threshold             |
| `random_timesteps`                     | `random_timesteps`                            | Random timesteps                    |
| `learning_starts`                      | `learning_starts`                             | Learning start timesteps            |
| `time_limit_bootstrap`                 | `time_limit_bootstrap`                        | Time limit bootstrap                |
| `learning_rate_scheduler_kl_threshold` | `learning_rate_scheduler_kwargs.kl_threshold` | Adaptive learning rate KL threshold |
| `check_point_interval`                 | `experiment.write_interval`                   | Log write interval                  |
| `check_point_interval`                 | `experiment.checkpoint_interval`              | Checkpoint save interval            |
| `rewards_shaper_scale`                 | `rewards_shaper`                              | Reward scaling function             |

#### Preprocessor Parameters

| SKRL Framework Parameter | Type                  | Description         |
| ------------------------ | --------------------- | ------------------- |
| `state_preprocessor`     | RunningStandardScaler | State normalization |
| `value_preprocessor`     | RunningStandardScaler | Value normalization |

### Configuration Hierarchy Summary

```
User Configuration Class (PPOCfg)
    ↓ Backend specific selection
Backend Configuration (JAX/Torch)
    ↓ Parameter mapping
SKRL Framework Configuration Dictionary
    ↓ Pass to
PPO Agent
    ↓ Execute
Reinforcement Learning Training
```

This design allows users to:

1. Control complex training parameters through simple configuration classes
2. Configure different parameters for different training backends to achieve optimal performance
3. Maintain full compatibility with the SKRL framework
