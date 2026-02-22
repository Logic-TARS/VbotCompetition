# Ping Pong Ball Bouncing

Train a single-arm robotic manipulator to control a paddle for continuous ball bouncing, maintaining the ball at a target height and position.

```{video} /_static/videos/bounce_ball.mp4
:poster: _static/images/poster/bounce_ball.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

Bounce Ball is a single-arm robotic manipulation task using a 6-DOF Peitian AIR4-560 industrial robotic arm to control the position of an end-effector paddle. The agent controls the position changes of the arm's 6 joints as actions, making the ping pong ball bounce continuously on the paddle and keeping it as close as possible to the target height and target horizontal position.

---

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (6,), float32)` |
| **Dimension** | 6                               |

The joints correspond as follows:

| Index | Action Meaning (Joint Position Change)  | Min Value | Max Value | Corresponding XML Name |
| ----: | --------------------------------------- | :-------: | :-------: | :--------------------: |
|     0 | Joint1 (Base Rotation) Position Change  |    -1     |     1     |        `Joint1`        |
|     1 | Joint2 (Upper Arm) Position Change      |    -1     |     1     |        `Joint2`        |
|     2 | Joint3 (Forearm) Position Change        |    -1     |     1     |        `Joint3`        |
|     3 | Joint4 (Wrist Rotation) Position Change |    -1     |     1     |        `Joint4`        |
|     4 | Joint5 (Wrist Pitch) Position Change    |    -1     |     1     |        `Joint5`        |
|     5 | Joint6 (Wrist Rotation) Position Change |    -1     |     1     |        `Joint6`        |

---

## Observation Space

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (25,), float32)` |
| **Dimension** | 25                               |

The observation space consists of the following parts (in order):

| Part        | Content Description                             | Dimension | Remarks                                                                          |
| ----------- | ----------------------------------------------- | --------- | -------------------------------------------------------------------------------- |
| **dof_pos** | Position information for each degree of freedom | 13        | First 6 are arm joints, last 7 are ball's free joint (3 position + 4 quaternion) |
| **dof_vel** | Velocity information for each degree of freedom | 12        | Velocity is derivative of position                                               |

| Index | Observation                 | Min Value | Max Value | XML Name    | Type (Unit)               |
| ----- | --------------------------- | --------- | --------- | ----------- | ------------------------- |
| 0-5   | Arm Joint Angles            | -Inf      | Inf       | Joint1-6    | Angle (rad)               |
| 6     | Ball x-coordinate           | -Inf      | Inf       | ball_x      | Position (m)              |
| 7     | Ball y-coordinate           | -Inf      | Inf       | ball_y      | Position (m)              |
| 8     | Ball z-coordinate           | -Inf      | Inf       | ball_z      | Position (m)              |
| 9-12  | Ball Orientation Quaternion | -Inf      | Inf       | ball_qw/xyz | Quaternion (w,x,y,z)      |
| 13-24 | Joint and Ball Velocities   | -Inf      | Inf       | -           | Velocity/Angular Velocity |

---

## Reward Function Design

The reward function consists of the following components:

```python
# Position Control Reward: Keep ball above paddle center
# Controlled Upward Velocity Reward: Reward moderate upward velocity when ball is well-positioned
# Height Accuracy Reward: Ball is close to target height
# Consecutive Bounces Reward: Reward consecutive successful bounces
# Total Reward = Weighted combination of all components
```

---

## Initial State

-   **Arm Initial Position**: [0, 40, 110, 0, -60, 0] degrees, with random noise
-   **Ball Initial Position**: Above paddle center, with random noise
-   **Ball Initial Velocity**: [0.0, 0.0, 0.0] m/s

---

## Episode Termination Conditions

-   **Ball Falls**: Ball z-coordinate < 0.05m (near ground)
-   **Ball Too High**: Ball z-coordinate > target height + 1.0m (lost control)
-   **Horizontal Deviation Too Far**: Absolute value of ball x-coordinate > 1.5m

---

## Usage Guide

### 1. Environment Preview

```bash
uv run scripts/view.py --env bounce_ball
```

### 2. Start Training

```bash
uv run scripts/train.py --env bounce_ball
```

### 3. View Training Progress

```bash
uv run tensorboard --logdir runs/bounce_ball
```

### 4. Test Training Results

```bash
uv run scripts/play.py --env bounce_ball
```

---

## Expected Training Results

1. Consecutive Bouncing: Capable of achieving 3 or more consecutive bounces
2. Position Control: Ball's horizontal position (x-coordinate) stable within target position ± 0.05m range
3. Height Control: Ball's height stable within target height 0.8 ± 0.1m range
4. Velocity Control: Ball's upward velocity maintained within reasonable range (0.1-1.5 m/s)
5. Stable Control: Capable of maintaining stable bouncing for 20 seconds without dropping
