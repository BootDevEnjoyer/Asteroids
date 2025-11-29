# Analysis: Counter-Clockwise Spiral Behavior

This document explains why the neural network AI enemy consistently exhibits counter-clockwise spiral motion toward the player, even after resetting and retraining from scratch.

## Observed Behavior

- The AI enemy approaches the player in a counter-clockwise spiral pattern
- This behavior persists across all training phases (1, 2, 3)
- Resetting the AI brain and retraining produces the same spiral direction
- The spiral is consistent regardless of initial spawn position

## System Overview

### State Space (5 features)

The `GameStateCollector.collect_state()` method produces:

| Index | Feature | Formula | Range |
|-------|---------|---------|-------|
| 0 | `normalized_angle` | $\arctan2(\Delta y, \Delta x) / \pi$ | [-1, 1] |
| 1 | `normalized_distance` | $d / 800$ | [0, 1] |
| 2 | `normalized_speed` | $v / (2 \cdot v_{max})$ | [0, 1] |
| 3 | `alignment` | $\hat{d} \cdot \hat{v}$ | [-1, 1] |
| 4 | `angular_velocity` | $(\theta_{target} - \theta_{current}) / \pi$ | [-1, 1] |

### Action Space (1 output)

The policy network outputs a single scalar:
- Range: $[-1, 1]$ (via `Tanh` activation)
- Interpretation: Angular adjustment from direct path to player

### Action Application

```python
angle_to_player = atan2(player.y - enemy.y, player.x - enemy.x)
adjustment = network_output * pi * 0.5  # Scale to [-pi/2, pi/2]
target_angle = angle_to_player + adjustment
```

## Mathematical Analysis

### The Missing Information Problem

Let:
- $\theta_p$ = absolute angle to player in world coordinates
- $\theta_e$ = enemy's current heading (direction of movement)
- $\theta_r = \theta_p - \theta_e$ = relative angle (player's position relative to enemy's heading)

The state space provides $\theta_p$ (feature 0) but **not** $\theta_e$ or $\theta_r$.

The `alignment` feature provides:

$$\text{alignment} = \cos(\theta_r)$$

This is the **magnitude** of alignment, not the **sign**. Critically:

$$\cos(+30°) = \cos(-30°) = 0.866$$

The network cannot distinguish "player is 30 degrees to my left" from "player is 30 degrees to my right."

### Why the Network Cannot Learn Optimal Behavior

For optimal pursuit, the network should output:

$$\text{adjustment} = -\frac{\theta_r}{\pi/2}$$

This requires knowing $\theta_r$. But the network only has:
- $\theta_p$ (absolute angle to player)
- $\cos(\theta_r)$ (unsigned alignment)

Without $\theta_e$, the network cannot compute $\theta_r = \theta_p - \theta_e$.

### The Emergent Fixed-Offset Strategy

Since the network cannot determine turn direction, it converges to a simpler strategy:

$$\text{adjustment} = c \quad \text{(constant bias)}$$

This produces:

$$\theta_{target} = \theta_p + c \cdot \frac{\pi}{2}$$

The enemy always aims at an angle $c \cdot \frac{\pi}{2}$ offset from the direct path.

### Geometric Proof of Spiral Trajectory

Given:
- Player at origin $P = (0, 0)$
- Enemy at position $E = (r \cos\phi, r \sin\phi)$ in polar coordinates
- Enemy velocity magnitude $= v$

With a constant angular offset $c$:

$$\text{heading} = \theta_p + c \cdot \frac{\pi}{2} = \phi + \pi + c \cdot \frac{\pi}{2}$$

(since $\theta_p$ points toward player $= \phi + \pi$)

The velocity components:

$$\frac{dx}{dt} = v \cos\left(\phi + \pi + c \cdot \frac{\pi}{2}\right)$$

$$\frac{dy}{dt} = v \sin\left(\phi + \pi + c \cdot \frac{\pi}{2}\right)$$

Converting to polar coordinates:

$$\frac{dr}{dt} = \frac{dx}{dt} \cos\phi + \frac{dy}{dt} \sin\phi = v \cos\left(\pi + c \cdot \frac{\pi}{2}\right) = -v \cos\left(c \cdot \frac{\pi}{2}\right)$$

$$r \frac{d\phi}{dt} = -\frac{dx}{dt} \sin\phi + \frac{dy}{dt} \cos\phi = v \sin\left(\pi + c \cdot \frac{\pi}{2}\right) = -v \sin\left(c \cdot \frac{\pi}{2}\right)$$

Therefore:

$$\frac{d\phi}{dt} = -\frac{v \sin(c \cdot \pi/2)}{r}$$

For any non-zero $c$:
- $\frac{dr}{dt} < 0$ (enemy approaches player, since $|c| < 1$ means $\cos(c\pi/2) > 0$)
- $\frac{d\phi}{dt} \neq 0$ (angular motion exists)
- Sign of $\frac{d\phi}{dt}$ determined by sign of $c$

This is exactly a **logarithmic spiral**:

$$r = r_0 \cdot e^{k\phi}$$

where:

$$k = \frac{\cos(c \cdot \pi/2)}{\sin(c \cdot \pi/2)} = \cot\left(c \cdot \frac{\pi}{2}\right)$$

### Why Counter-Clockwise Specifically?

The consistent counter-clockwise direction implies $c > 0$ (positive bias).

Sources of positive bias:

1. **PyTorch Weight Initialization**
   - `nn.Linear` uses Kaiming uniform initialization
- Final layer bias initialized from $\mathcal{U}\left(-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}\right)$
   - With $n = 32$, bias $\sim \text{Uniform}(-0.177, 0.177)$
   - Any non-zero initialization creates directional preference

2. **ReLU Activation Asymmetry**
   - ReLU zeros negative activations: $\text{ReLU}(x) = \max(0, x)$
   - This breaks symmetry in gradient flow
   - Positive pathways may dominate due to initialization

3. **Coordinate System Convention**
   - Pygame: Y-axis increases downward
   - `atan2`: Returns angle in standard mathematical convention
   - Positive angles = counter-clockwise in world space

4. **Reward Function Symmetry**
   - The reward function is symmetric with respect to turn direction
   - Both clockwise and counter-clockwise spirals receive equal reward
   - Network converges to whichever direction the initialization biases toward

### Quantitative Verification

From debug logs, typical network outputs hover around $+0.1$ to $+0.3$.

For $c = 0.2$:

$$\text{Offset angle} = 0.2 \times 90° = 18°$$

$$\text{Spiral tightness} = \cot(18°) \approx 3.08$$

This matches observed behavior: a moderately tight counter-clockwise spiral.

## Solution

Add the **signed relative angle** to the state space:

```python
relative_angle = angle_to_player - enemy.current_angle
# Wrap to [-pi, pi]
while relative_angle > pi:
    relative_angle -= 2 * pi
while relative_angle < -pi:
    relative_angle += 2 * pi
normalized_relative = relative_angle / pi  # [-1, 1]
```

Replace feature 4 (`angular_velocity`) with `normalized_relative`.

This provides the network with:
- **Sign**: Positive = player is to the left, negative = player is to the right
- **Magnitude**: How far off-course the enemy is

The optimal policy becomes learnable:

$$\text{adjustment} = -k \cdot \theta_r \quad \text{for some } k > 0$$