# Training Execution and Result Analysis

This section introduces how to execute reinforcement learning training and how to analyze and use training results.

## Start Training

### Basic Training Commands

```bash
# Train with default parameters
uv run scripts/train.py --env cartpole

# Specify simulation backend
uv run scripts/train.py --env cartpole --sim-backend np

# Specify training backend
uv run scripts/train.py --env cartpole --train-backend jax
uv run scripts/train.py --env cartpole --train-backend torch
```

### Advanced Training Configuration

```bash
# Customize training parameters
uv run scripts/train.py --env cartpole \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np

# Enable rendering to monitor training process
uv run scripts/train.py --env cartpole --render
```

### Different Backend Configuration Differences

The system supports configuring different reinforcement learning parameters for different training backends (JAX/Torch). For example:

-   **dm-walker environment**:

    -   JAX backend: `mini_batches: 4`
    -   Torch backend: `mini_batches: 32`

-   **dm-runner environment**:
    -   JAX backend: `learning_epochs: 4`
    -   Torch backend: `learning_epochs: 2`

These differences are implemented through the `@rlcfg(env_name, backend="jax/torch")` decorator in configuration classes. The system automatically applies the corresponding configuration based on the selected training backend.

### Supported Command Line Parameters

| Parameter         | Description                     | Default Value |
| ----------------- | ------------------------------- | ------------- |
| `--env`           | Environment name                | `cartpole`    |
| `--sim-backend`   | Simulation backend (np)         | Auto select   |
| `--train-backend` | Training backend (jax/torch)    | Auto select   |
| `--num-envs`      | Number of parallel environments | 2048          |
| `--render`        | Enable rendering                | False         |

> **Note**: Other parameters such as learning rate, network structure, etc., can be set in configuration files. Some environments support configuring different parameters for different training backends.

## Training Process Monitoring

### TensorBoard Monitoring

Start TensorBoard to view training progress:

```bash
uv run tensorboard --logdir runs/{env-name}
```

For example:

```bash
uv run tensorboard --logdir runs/cartpole
```

## Model Evaluation and Testing

### Using Trained Policies

```bash
# Automatically find best policy for testing (recommended)
uv run scripts/play.py --env cartpole

# Manually specify policy file for testing
uv run scripts/play.py --env cartpole --policy runs/cartpole/nn/best_policy.pickle

# Specify number of test environments
uv run scripts/play.py --env cartpole --num-envs 100
```

> **Note**: The system will automatically find the latest and best policy files in the `runs/cartpole/` directory for testing.
