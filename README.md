# 🏆 谋先飞机器人比赛（VBot Competition）项目展示

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![RL](https://img.shields.io/badge/RL-SKRL-FF6F00)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-2E8B57)
![Status](https://img.shields.io/badge/Portfolio-Ready-success)

> 本仓库用于展示我在 **谋先飞机器人比赛（VBot Competition）** 中的强化学习算法与工程实践成果。  
> 项目基于 **MotrixSim** 仿真平台与 **SKRL** 训练框架，聚焦四足机器人导航任务。

---

## 🌟 项目亮点与个人贡献（Project Highlights & Personal Contributions）

- **算法设计与优化（Algorithm Design）**
  - 面向比赛任务设计与迭代强化学习策略（如 PPO 训练流程优化）。
  - 通过奖励函数与训练流程调优，提升策略稳定性与收敛效率。
  - 最终结果：**[在此处填写性能提升指标]**。

- **自定义强化学习环境（Custom RL Environments）**
  - 在比赛场景中完成导航任务环境的定制与扩展。
  - 面向赛道目标、障碍与阶段任务进行环境配置和训练适配。

- **竞赛成果（Competition Result）**
  - 比赛名次/奖项：**[在此处填写您的具体名次]**。
  - 关键成绩说明：**[在此处填写关键结果，如完成率、稳定性、收敛轮次等]**。

---

## 🛠 技术栈（Tech Stack）

- **Python**
- **SKRL**（强化学习训练框架）
- **MotrixSim**（机器人仿真平台）

---

## 🚀 环境配置与运行（Setup and Run）

### 1) 克隆仓库

```bash
git clone https://github.com/Logic-TARS/vbot-competition.git
cd vbot-competition
git lfs pull
```

### 2) 安装依赖

> 本项目使用 `uv` 进行依赖管理。

```bash
uv sync --all-packages --extra skrl-torch
# 或者使用 JAX 后端：
# uv sync --all-packages --extra skrl-jax
```

### 3) 查看比赛环境（可视化）

```bash
uv run scripts/view.py --env vbot_navigation_section001
```

### 4) 训练模型

```bash
uv run scripts/train.py --env vbot_navigation_section001
uv run tensorboard --logdir runs/
```

### 5) 推理/回放

```bash
uv run scripts/play.py --env vbot_navigation_section001
```

---

## 📁 仓库整理说明

- 比赛文档已统一归档至：`docs/competition/`
- 训练日志已统一归档至：`logs/`
- 历史压缩包已归档至：`archives/`

---

## 👤 作者信息（可按需补充）

- **GitHub**: [Logic-TARS](https://github.com/Logic-TARS)
- **Email / LinkedIn / 个人主页**: [在此处填写您的联系方式]
