# 🏆 谋先飞机器人比赛（VBot Competition）项目展示

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![RL](https://img.shields.io/badge/RL-SKRL-FF6F00)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-2E8B57)
![Backend](https://img.shields.io/badge/Backend-JAX%20%7C%20PyTorch-FF6F00)

> 本仓库用于展示我在 **谋先飞机器人比赛（VBot Competition）** 中的强化学习算法与工程实践成果。  
> 项目基于 **MotrixSim** 仿真平台与 **SKRL** 训练框架，聚焦四足机器人导航任务。

---

## 🏅 竞赛成果

- **奖项**：三等奖 🥉
- **排名**：8 / 30 名
- **技术路线**：基于 PPO 算法的端到端强化学习策略，在 MotrixSim 仿真环境中完成四足机器人（VBot）的导航任务训练

---

## 🎯 比赛任务（满分 25 分）

完整赛道分三阶段，机器人需依次导航通过每段赛道：

| 阶段 | 路段 | 内容 | 分值 |
|------|------|------|------|
| 阶段一 | Section 011 → 012 | 穿越滚动球区域（二选一策略） | 10~15 分 |
| | | 策略 A：避开滚球安全通过 | 10 分 |
| | | 策略 B：触碰滚球且保持不摔倒 | 15 分 |
| 阶段二 | Section 012 → 013 | 穿越随机地形到达终点"中国结" | 5 分 |
| 阶段三 | 终点 | 终点庆祝动作 | 5 分 |

导航航点顺序：`(0, 7.5)` → `(0, 24.3)` → `(0, 32.3)`

---

## 🛠 技术栈

| 技术 | 用途 |
|------|------|
| **Python 3.10** | 开发语言 |
| **MotrixSim** | 机器人仿真平台（物理引擎） |
| **SKRL** | 强化学习训练框架 |
| **JAX / PyTorch** | 训练后端（支持切换） |
| **uv** | Python 包管理器与 monorepo 工作区 |
| **TensorBoard** | 训练监控与可视化 |

---

## 🧪 可用环境列表

项目注册了多种比赛环境，覆盖不同训练粒度和赛道分段：

| 环境名 | 说明 | 适用场景 |
|--------|------|----------|
| `vbot_navigation_flat` | 平地导航 | 基础导航训练 |
| `vbot_navigation_stairs` | 楼梯地形导航 | 上下坡/楼梯训练 |
| `vbot_navigation_section01` | 赛道第 1 段 | 分段训练 |
| `vbot_navigation_section02` | 赛道第 2 段 | 分段训练 |
| `vbot_navigation_section03` | 赛道第 3 段 | 分段训练 |
| `vbot_navigation_section011_012` | 第 1-2 段合并（无滚球/终点逻辑） | 地形穿越专项训练 |
| `vbot_navigation_long_course` | 长赛道 | 长距离导航训练 |
| `vbot_navigation_full` | 完整赛道（三段全含，满分 25 分） | 完整比赛模拟 |
| `vbot-flat-terrain-walk` | 行走 locomotion | 基础行走训练 |

已训练完成的策略文件位于 `checkpoints/` 目录，包括 `best_agent.pickle`、`best_agent011.pickle` ~ `best_agent013.pickle` 等按赛段保存的最优模型。

---

## 🚀 环境配置与运行

### 1) 克隆仓库

```bash
git clone https://github.com/Logic-TARS/vbot-competition.git
cd vbot-competition
git lfs pull
```

### 2) 安装依赖

本项目使用 `uv` 进行依赖管理，采用 monorepo 工作区结构（`motrix_envs` + `motrix_rl` 两个子包）。

```bash
# JAX 后端（默认推荐）
uv sync --all-packages --extra skrl-jax

# 或 PyTorch 后端
uv sync --all-packages --extra skrl-torch
```

### 3) 查看比赛环境（可视化）

```bash
uv run scripts/view.py --env vbot_navigation_section001
```

### 4) 训练模型

```bash
# 训练单个赛段
uv run scripts/train.py --env vbot_navigation_section001

# 训练完整赛道
uv run scripts/train.py --env vbot_navigation_full

# 查看训练曲线
uv run tensorboard --logdir runs/
```

训练脚本支持多种参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-envs` | 并行环境数 | 2048 |
| `--train-backend` | 训练后端 (`jax`/`torch`) | 自动检测 |
| `--policy` | 加载预训练策略（用于课程学习） | None |
| `--checkpoint-interval` | 保存间隔（timesteps） | 1000 |
| `--seed` / `--rand-seed` | 随机种子 | 固定 / 随机 |
| `--env-cfg` | 环境配置覆盖（如 `--env-cfg curriculum_from_001=True`） | — |

### 5) 推理/回放

```bash
# 自动加载最新训练好的策略
uv run scripts/play.py --env vbot_navigation_section001

# 指定策略文件
uv run scripts/play.py --env vbot_navigation_section001 --policy checkpoints/best_agent.pickle
```

---

## 📁 仓库结构

```
├── motrix_envs/              # 仿真环境包
│   └── src/motrix_envs/
│       ├── navigation/vbot/  # VBot 导航环境（核心比赛代码）
│       ├── locomotion/vbot/  # VBot 行走环境
│       └── basic/            # 基础示例环境（cartpole, hopper 等）
├── motrix_rl/                # RL 训练框架包
│   └── src/motrix_rl/skrl/
│       ├── jax/train/ppo.py  # PPO-JAX 训练器
│       └── torch/train/ppo.py# PPO-PyTorch 训练器
├── scripts/                  # 训练/推理/可视化脚本
│   ├── train.py
│   ├── play.py
│   └── view.py
├── checkpoints/              # 训练好的策略模型
├── docs/                     # 比赛文档
├── logs/                     # 训练日志
├── archives/                 # 历史归档
├── pyproject.toml            # 项目配置（uv workspace）
└── uv.lock
```

---

## 📝 亮点说明

- **奖励函数塑形**：稠密奖励引导 + 阶段性任务奖励（checkpoint 检查点奖励、终点庆祝奖励），加速收敛
- **多阶段课程学习**：支持分段训练 → 合并训练 → 完整赛道的递进式课程学习（通过 `--policy` 加载预训练策略）
- **双后端支持**：JAX 和 PyTorch 两种训练后端可自由切换
- **PPO 算法优化**：针对导航任务调整 PPO 超参数，优化策略稳定性与收敛效率

---

## 👤 作者信息

- **GitHub**: [Logic-TARS](https://github.com/Logic-TARS)
