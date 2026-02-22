# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MotrixLab is a reinforcement learning framework built on top of MotrixSim simulation backend. It provides a unified interface for training RL agents using multiple simulation backends (MotrixSim) and primarily integrates with the SKRL library. The framework is designed for robotics simulation and supports various environments including basic cartpole, locomotion tasks, and manipulation tasks.

## Development Setup

This project uses UV for dependency management and Python 3.10.

### Installation

```bash
uv sync --all-packages --all-extras
```

For SKRL framework with specific backend:

```bash
uv sync --all-packages --extra skrl-jax  # JAX backend
uv sync --all-packages --extra skrl-torch  # PyTorch backend
```

**Available dependency groups in MotrixLab:**

-   `skrl-jax`: SKRL RL framework with JAX backend
-   `skrl-torch`: SKRL RL framework with PyTorch backend
-   `test`: Test dependencies (pytest)

**Note**: This is a workspace project with two main packages: `motrix_envs` (simulation environments) and `motrix_rl` (RL framework integration).

## Common Commands

### Training

```bash
uv run scripts/train.py --env cartpole
```

### Environment Visualization

View environment without training:

```bash
uv run scripts/view.py --env cartpole
```

### Playing/Evaluation

```bash
uv run scripts/play.py --env cartpole
```

Specify policy file:

```bash
uv run scripts/play.py --env cartpole --policy <path/to/best.[pickle/pt]>
```

### Rendering

Add `--render` flag to training for visualization:

```bash
uv run scripts/train.py --env cartpole --render
```

### TensorBoard

```bash
uv run tensorboard --logdir runs/{env-name}
```

### Testing

```bash
uv run pytest
```

## Architecture

### Core Components

1. **Workspace Structure**:

    - `motrix_envs/`: Simulation environment definitions using MotrixSim backend
    - `motrix_rl/`: RL framework integration (primarily SKRL) and training utilities

2. **Scripts** (`scripts/`):

    - `train.py`: Main training script with configurable environments and backends
    - `view.py`: Environment visualization without training
    - `play.py`: Policy evaluation and testing

3. **Environment Registry**: Environments are registered via `motrix_envs.registry` and accessed using string names like "cartpole"

### Key Architecture Points

-   **Workspace Project**: Uses UV workspace with two packages sharing dependencies
-   **MotrixSim Backend**: Built on MotrixSim simulation engine for physics simulation
-   **SKRL Integration**: Primary RL framework supporting both JAX and PyTorch backends
-   **Environment Naming**: Simple string-based environment identification (e.g., "cartpole")
-   **Automatic Backend Selection**: Training script automatically selects JAX or PyTorch based on GPU availability
-   **Multi-Backend Training**: Supports different simulation backends for the same environment

### Environment Usage Pattern

```python
# Environment creation handled internally by scripts
# Use string names to specify environments
uv run scripts/train.py --env cartpole
```

## Results Storage

Training results are saved to `runs/{env-name}/` directory structure with checkpoints and tensorboard logs.

## Important Notes

-   **Python Version**: Requires exactly Python 3.10.\*
-   **GPU Support**: Includes CUDA support for both JAX and PyTorch backends
-   **Private PyPI**: Uses internal PyPI server for MotrixSim packages
-   **No Manual Tests**: No test files found in the repository structure
