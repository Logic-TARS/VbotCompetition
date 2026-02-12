# VBot Immediate Termination Bug Fix - Summary

## Problem Statement
VBot robots were "dying at birth" during training with `vbot_navigation_section001` environment:
- **Symptoms**: Robots "disappeared in mid-air" in videos
- **Metrics**: 
  - Total reward (mean): ~0.29 (extremely low)
  - Total reward (max): ~1.7-1.9 (should be much higher)
  - Episode length: 1-5 steps (robots died immediately)
  - Survival rate: <10%

## Root Causes (4 Bugs Identified)

### Bug 1: Reset Position Generation Ignores Curriculum Learning
**Location**: `motrix_envs/src/motrix_envs/navigation/vbot/vbot_section001_np.py` L815-828

**Problem**: 
- `reset()` used polar coordinate generation to spawn robots on outer circle (radius 3.0m)
- This **completely ignored** `cfg.init_state.pos = [0.0, 0.6, 0.5]` (curriculum learning Phase 1)
- Target was at `[0.0, 0.0]`, so robots spawned ~3m away from target
- `boundary_radius = 3.5` meant robots started very close to boundary

**Fix**:
- Replace polar coordinate logic with `cfg.init_state.pos` as base position
- Add small XY randomization using `pos_randomization_range` (±0.3m)
- Lower initial height from 0.5m → 0.35m to reduce fall impact
- Increase `boundary_radius` from 3.5 → 5.0 in config

### Bug 2: Base Contact Sensor Triggers Immediately on Landing
**Location**: `motrix_envs/src/motrix_envs/navigation/vbot/vbot_section001_np.py` L508-520

**Problem**:
- `base_contact` threshold was 0.01 (extremely sensitive)
- No grace period - terminated immediately on landing
- Robot falls from 0.5m height, base touches ground → instant termination

**Fix**:
- Increase threshold from 0.01 → 0.1 (10x more tolerant)
- Add 50-step grace period (~0.5 seconds at 50Hz)
- Only check termination if `current_steps > GRACE_STEPS`
- Add "steps" key to info dict for tracking

### Bug 3: Hard-Coded -10.0 Penalty on Landing
**Location**: `motrix_envs/src/motrix_envs/navigation/vbot/vbot_section001_np.py` L800-803

**Problem**:
- Hard-coded `-10.0` penalty when `orientation_penalty > 0.5` (~45° tilt)
- Robot falling from 0.5m easily tilts >45° on landing → instant -10 reward
- Created extreme reward spikes that disrupted learning

**Fix**:
- Replace with progressive penalty: `clip((orientation_penalty - 0.5) * 5.0, 0.0, 3.0)`
- Penalty scales smoothly: 0° → 0, 45° → 0, 60° → 0.75, 80° → 1.5, 100° → 2.5, max 3.0
- Much gentler learning signal

### Bug 4: Missing Info Dict Keys
**Location**: `motrix_envs/src/motrix_envs/navigation/vbot/vbot_section001_np.py` L1037-1044

**Problem**:
- `reset()` info dict missing "last_distance" key
- `reset()` info dict missing "steps" key
- Caused `_compute_reward()` to use fallback: `info.get("last_distance", distance_to_target)`
- Grace period mechanism couldn't work without "steps"

**Fix**:
- Add `"last_distance": distance_to_target.copy()` to info dict
- Add `"steps": np.zeros(num_envs, dtype=np.int32)` to info dict

## Changes Made

### File 1: `motrix_envs/src/motrix_envs/navigation/vbot/vbot_section001_np.py`

#### Change 1: `reset()` method (L823-837)
```python
# OLD: Polar coordinate generation (Bug 1)
robot_init_xy = np.zeros((num_envs, 2), dtype=np.float32)
for i in range(num_envs):
    theta = np.random.uniform(0, 2 * np.pi)
    radius = cfg.arena_outer_radius + np.random.uniform(-0.1, 0.1)
    robot_init_xy[i, 0] = radius * np.cos(theta)
    robot_init_xy[i, 1] = radius * np.sin(theta)
robot_init_xy += np.array(cfg.arena_center, dtype=np.float32)
robot_init_pos = np.column_stack([robot_init_xy, np.full(num_envs, 0.5)])

# NEW: Use cfg.init_state.pos with small randomization
base_pos = np.array(cfg.init_state.pos, dtype=np.float32)
robot_init_pos = np.tile(base_pos, (num_envs, 1))

if hasattr(cfg.init_state, 'pos_randomization_range'):
    pr = cfg.init_state.pos_randomization_range
    xy_noise = np.random.uniform(
        [pr[0], pr[1]], [pr[2], pr[3]], (num_envs, 2)
    ).astype(np.float32)
    robot_init_pos[:, :2] += xy_noise

robot_init_pos[:, 2] = 0.35  # Lower height
```

#### Change 2: `_compute_terminated()` method (L508-524)
```python
# OLD: No grace period, 0.01 threshold (Bug 2)
base_contact = (base_contact_value > 0.01).flatten()[:num_envs]
terminated = np.logical_or(terminated, base_contact)

# NEW: Grace period + 0.1 threshold
GRACE_STEPS = 50
current_steps = state.info.get("steps", np.zeros(num_envs, dtype=np.int32))
past_grace = current_steps > GRACE_STEPS

base_contact = (base_contact_value > 0.1).flatten()[:num_envs]
terminated = np.logical_or(terminated, base_contact & past_grace)
```

#### Change 3: `_compute_reward()` method (L810-811)
```python
# OLD: Hard-coded -10.0 (Bug 3)
reward = np.where(orientation_penalty > 0.5, reward - 10.0, reward)

# NEW: Progressive penalty
extreme_tilt_penalty = np.clip((orientation_penalty - 0.5) * 5.0, 0.0, 3.0)
reward = reward - extreme_tilt_penalty
```

#### Change 4: `reset()` info dict (L1053-1054)
```python
# OLD: Missing keys (Bug 4)
info = {
    "pose_commands": pose_commands,
    "last_actions": ...,
    "current_actions": ...,
    "filtered_actions": ...,
    "ever_reached": ...,
    "min_distance": distance_to_target.copy(),
}

# NEW: Added missing keys
info = {
    "pose_commands": pose_commands,
    "last_actions": ...,
    "current_actions": ...,
    "filtered_actions": ...,
    "ever_reached": ...,
    "min_distance": distance_to_target.copy(),
    "last_distance": distance_to_target.copy(),  # ✅ Bug 4 fix
    "steps": np.zeros(num_envs, dtype=np.int32),  # ✅ Bug 4 fix
}
```

### File 2: `motrix_envs/src/motrix_envs/navigation/vbot/cfg.py`

#### Change 1: Boundary radius (L395)
```python
# OLD
boundary_radius: float = 3.5

# NEW
boundary_radius: float = 5.0  # ⬆️ Fix Bug 1: 3.5 → 5.0
```

#### Change 2: Initial height (L411)
```python
# OLD
pos = [0.0, 0.6, 0.5]

# NEW
pos = [0.0, 0.6, 0.35]  # ⬇️ Fix Bug 1: 0.5 → 0.35
```

## Expected Impact

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Episode survival steps | 1-5 | 100+ |
| Total reward (mean) | ~0.29 | >2.0 |
| Total reward (max) | ~1.7-1.9 | >10.0 |
| Robot survival rate | <10% | >80% |
| "Mid-air disappearance" | Frequent | Eliminated |

## Verification

### Manual Testing
Run training to observe improvements:
```bash
uv run scripts/train.py --env vbot_navigation_section001
```

Monitor in TensorBoard:
```bash
uv run tensorboard --logdir runs/vbot_navigation_section001
```

### Automated Testing
Run validation test:
```bash
python3 test_immediate_termination_fix.py
```

This test validates:
- ✅ Config changes (boundary_radius, init height)
- ✅ Reset position logic (uses cfg.init_state.pos)
- ✅ Grace period mechanism (50 steps)
- ✅ Base contact threshold (0.1)
- ✅ Progressive penalty (not -10.0)
- ✅ Info dict keys (last_distance, steps)

## Technical Details

### Why These Fixes Work

1. **Bug 1 Fix (Position)**: 
   - Robots now start at curriculum learning position (0.6m from target)
   - Much higher chance of reaching target during exploration
   - Lower fall height reduces landing impact

2. **Bug 2 Fix (Base Contact)**:
   - Grace period allows robot to stabilize after landing
   - Higher threshold prevents false positives from normal ground contact
   - Robots can recover from initial instability

3. **Bug 3 Fix (Penalty)**:
   - Smooth penalty gradient helps learning
   - No sudden reward drops that confuse the policy
   - Robot can explore tilt ranges without extreme punishment

4. **Bug 4 Fix (Info Keys)**:
   - Reward function correctly calculates distance progress
   - Grace period mechanism functions properly
   - Consistent info dict structure across reset/step

### Compatibility

All changes are **backward compatible**:
- `hasattr()` checks protect against missing config attributes
- Graceful fallbacks for old configs
- No breaking changes to API

## Related Issues

This fix addresses the core "immediate termination" issue that was preventing any meaningful training. Related improvements that were already in place:
- Recovery tilt threshold (80°) for extreme falls
- Force initial motion to break zero-velocity trap
- Reward reshaping with dominant positive rewards

## Author
GitHub Copilot

## Date
2026-02-12

## Commit References
- Main fix: `d459f00` - Fix all 4 bugs causing immediate termination
- Test: `501e072` - Add validation test for immediate termination fix
