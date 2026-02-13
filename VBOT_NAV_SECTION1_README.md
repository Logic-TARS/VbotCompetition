# VBot Navigation Section 1 Competition Environment

This document describes the newly implemented `VBotNavSection1Env` environment for the Motphys "Obstacle Navigation - Section 1" competition.

## Overview

The VBot Navigation Section 1 environment simulates a robot dog competition where the agent must:
1. Start from a random position in the START zone
2. Navigate through complex terrain (pits and slopes)
3. Collect bonus points by visiting smiley faces and red envelopes (hongbaos)
4. Reach the finish zone (2026 platform)
5. Perform a celebration action at the finish

## Competition Rules

### Base Score (20 points)
- Robot starts from START platform with random initial position
- Must navigate through complex terrain (pits + slopes) to reach 2026 platform
- Any body part entering the 2026 zone scores points
- Must not fall or go out of bounds

### Bonus Points (20 points total)
1. **Smiley Face Zones** (3 smileys × 4 points = 12 points max)
   - Entering a smiley zone triggers scoring
   - Same smiley can only be scored once
   - Need to visit all 3 different smileys for full points

2. **Hongbao (Red Envelope) Zones** (3 hongbaos × 2 points = 6 points max)
   - Hovering hongbaos above "GO" characters on slope
   - Reaching hongbao position triggers scoring
   - Each hongbao can only be scored once

3. **Celebration Action** (+2 points)
   - After reaching 2026 platform, perform a celebration
   - Stay still for 1 second at finish zone
   - Elegant and obvious celebration movement

**Theoretical Maximum Score: 20 + 12 + 6 + 2 = 40 points**

### Penalty/Failure Conditions
- Non-random start position (cheating) → All section points deducted
- Out of bounds → All section points deducted (-10 reward penalty)
- Falling (roll/pitch > 45°) → All section points deducted (-10 reward penalty)

## Environment Configuration

### Registration
- **Name**: `vbot_nav_section1`
- **Config Class**: `VBotNavSection1EnvCfg`
- **Environment Class**: `VBotNavSection1Env`

### Key Parameters

```python
from motrix_envs.navigation.vbot import VBotNavSection1EnvCfg

cfg = VBotNavSection1EnvCfg()

# Episode settings
cfg.max_episode_seconds = 70.0  # 70 seconds
cfg.max_episode_steps = 7000    # 7000 steps @ 100Hz

# Terrain zones (coordinates in meters)
cfg.start_zone_center = [0.0, -2.0]
cfg.start_zone_radius = 1.5

cfg.smiley_positions = [
    [-2.0, 0.5],   # Left smiley
    [0.0, 1.0],    # Center smiley
    [2.0, 0.5],    # Right smiley
]
cfg.smiley_radius = 0.8

cfg.hongbao_positions = [
    [-1.5, 4.5],   # Left hongbao
    [0.0, 5.0],    # Center hongbao  
    [1.5, 4.5],    # Right hongbao
]
cfg.hongbao_radius = 0.6

cfg.finish_zone_center = [0.0, 8.0]
cfg.finish_zone_radius = 1.5

# Boundaries (x: ±5m, y: -3.5m to 9m)
cfg.boundary_x_min = -5.0
cfg.boundary_x_max = 5.0
cfg.boundary_y_min = -3.5
cfg.boundary_y_max = 9.0
```

## Observation Space

**Total Dimensions: 75**

### Base Observations (54 dimensions)
- Linear velocity (3): Body linear velocity in body frame
- Gyroscope (3): Angular velocity in body frame
- Projected gravity (3): Gravity vector in body frame
- Joint angles (12): Relative to default stance
- Joint velocities (12): Angular velocities of joints
- Last actions (12): Previous action commands
- Velocity commands (3): Desired [vx, vy, yaw_rate]
- Position error (2): Distance to finish zone [x, y]
- Heading error (1): Angle difference to finish
- Distance (1): Normalized distance to finish
- Reached flag (1): Whether finish zone is reached
- Stop ready flag (1): Whether robot is stable at finish

### Competition-Specific Observations (21 dimensions)
- Smiley relative positions (6): [x, y] for each of 3 smileys, normalized by 5m
- Hongbao relative positions (6): [x, y] for each of 3 hongbaos, normalized by 5m
- Trigger flags (6): Binary flags for each smiley (3) and hongbao (3)
- Finish relative position (2): [x, y] to finish zone, normalized by 5m
- Finish flag (1): Binary flag if finish zone is triggered

## Action Space

**Dimensions: 12** (same as base VBot environment)

Joint position targets for 12 actuators:
- FR_hip, FR_thigh, FR_calf (Front Right leg)
- FL_hip, FL_thigh, FL_calf (Front Left leg)
- RR_hip, RR_thigh, RR_calf (Rear Right leg)
- RL_hip, RL_thigh, RL_calf (Rear Left leg)

Actions are in range [-1, 1] and scaled by `action_scale = 0.25`.

## Reward Function

The reward function is designed to encourage robust navigation with bonus collection:

### 1. Alive Bonus (+0.01 per step)
- Encourages survival and stability
- Penalizes early termination

### 2. Waypoint Progress Rewards (Dense)
- Active waypoint system: smiley1 → smiley2 → smiley3 → hongbao1 → hongbao2 → hongbao3 → finish
- Progress reward: `+0.5 * (last_distance - current_distance)` to active waypoint
- Encourages moving toward next objective
- Clipped to [-0.1, 0.5] per step

### 3. Sparse Trigger Rewards
- **Smiley triggered**: +4.0 per smiley (first time only)
- **Hongbao triggered**: +2.0 per hongbao (first time only)
- **Finish triggered**: +20.0 (first time only)
- **Celebration completed**: +2.0 (stay still at finish for 1 second)

### 4. Stability Penalties
- **Orientation penalty**: -0.05 * (|roll| + |pitch|)
- **Action rate penalty**: -0.01 * ||action_t - action_{t-1}||²
- **Torque penalty**: -1e-5 * ||torques||²

### 5. Termination Penalties
- **Fall or out-of-bounds**: -10.0

## Termination Conditions

The episode terminates when:

1. **Base Contact** (fall detection)
   - Base body contacts ground (sensor value > 0.1)
   
2. **Attitude Failure** (excessive tilt)
   - Roll or pitch > 45 degrees
   
3. **Out of Bounds**
   - Robot position outside boundary limits:
     - X: [-5, 5] meters
     - Y: [-3.5, 9] meters

4. **Timeout**
   - Episode exceeds 70 seconds (7000 steps @ 100Hz)

**Grace Period**: First 20 frames (0.2 seconds) have no termination to prevent spawn-death from physics initialization jitter.

## Usage Example

### Training

```python
from motrix_envs.navigation.vbot import VBotNavSection1Env, VBotNavSection1EnvCfg

# Create configuration
cfg = VBotNavSection1EnvCfg()

# Customize if needed
cfg.max_episode_seconds = 60.0  # Shorter episodes for faster training

# Create environment with multiple parallel instances
env = VBotNavSection1Env(cfg, num_envs=10)

# Training loop
obs = env.reset()
for step in range(1000):
    action = policy(obs)  # Your RL policy
    obs, reward, done, info = env.step(action)
    
    # Check trigger states
    if info.get("smiley_0_rewarded", False):
        print("Smiley 1 collected!")
```

### Evaluation

```python
# Single environment for evaluation
env = VBotNavSection1Env(cfg, num_envs=1)

obs = env.reset()
total_reward = 0
done = False

while not done:
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    # Log collection events
    for i in range(3):
        if info.get(f"smiley_{i}_rewarded", False):
            print(f"Collected smiley {i+1}!")
        if info.get(f"hongbao_{i}_rewarded", False):
            print(f"Collected hongbao {i+1}!")
    
    if info.get("finish_rewarded", False):
        print("Reached finish zone!")
    
    if info.get("celebration_rewarded", False):
        print("Celebration completed!")

print(f"Total reward: {total_reward}")
```

## Training Strategy

### Curriculum Learning Phases

**Phase 1: Basic Navigation (20 points baseline)**
- Focus: START → Finish direct path
- Disable bonus rewards temporarily
- Train for robust locomotion and reaching finish

**Phase 2: Smiley Collection (32 points)**
- Enable smiley rewards
- Train S-shaped path through pit terrain
- Encourage exploration of smiley zones

**Phase 3: Hongbao Collection (38 points)**
- Enable hongbao rewards
- Train slope navigation
- Balance between smileys and hongbaos

**Phase 4: Celebration Action (40 points)**
- Enable celebration reward
- Train to stop at finish and stay still
- Fine-tune final behavior

### Key Design Principles

1. **Robustness > Speed**: One fall = 0 points, so survival is highest priority
2. **Path Planning**: Non-linear path required to collect all bonuses
3. **Generalization**: Random start position prevents overfitting to single path
4. **Exploration**: Waypoint system guides agent through all zones
5. **Stability**: Penalties discourage aggressive movements that risk falling

## Files

- **Configuration**: `motrix_envs/src/motrix_envs/navigation/vbot/cfg.py` (class `VBotNavSection1EnvCfg`)
- **Environment**: `motrix_envs/src/motrix_envs/navigation/vbot/vbot_nav_section1_np.py` (class `VBotNavSection1Env`)
- **Scene XML**: `motrix_envs/src/motrix_envs/navigation/vbot/xmls/scene_section01.xml`
- **Collision XML**: `motrix_envs/src/motrix_envs/navigation/vbot/xmls/0126_C_section01.xml`
- **Exports**: `motrix_envs/src/motrix_envs/navigation/vbot/__init__.py`

## Compatibility

This environment preserves the existing `VBotSection001Env` and all other configurations. It can be used alongside existing environments without conflicts:

```python
# Old environment still works
from motrix_envs.navigation.vbot import VBotSection001Env, VBotSection001EnvCfg

# New competition environment
from motrix_envs.navigation.vbot import VBotNavSection1Env, VBotNavSection1EnvCfg
```

## Next Steps

1. Install dependencies: `uv sync --all-packages --extra skrl-jax` or `--extra skrl-torch`
2. Test environment creation: `python test_vbot_nav_section1.py`
3. Train initial policy: `uv run scripts/train.py --env vbot_nav_section1`
4. Visualize behavior: `uv run scripts/view.py --env vbot_nav_section1`
5. Evaluate trained policy: `uv run scripts/play.py --env vbot_nav_section1 --policy path/to/policy.pt`

## Notes

- Zone coordinates are estimated based on terrain geometry analysis
- May need fine-tuning based on actual terrain visualization
- Reward scaling can be adjusted for better training performance
- Grace period prevents spawn-death during physics initialization
- NaN/Inf guards prevent CUDA crashes from invalid observations
