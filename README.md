# Asteroids AI

A classic Asteroids game featuring neural network enemies that learn to hunt the player through reinforcement learning.

---

> **Note:** This is a personal learning project exploring reinforcement learning in games.
> It is based upon the boot.dev Asteroids project and extended further.
> During AI training modes, the player ship follows predetermined patterns while the neural network enemies learn pursuit behaviors.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Launch the game
uv run python main.py
```

Use the menu to select a mode:
- **Watch AI Learn** - See neural networks train in real-time
- **Watch Trained AI** - Observe learned behaviors
- **Play Classic** - Traditional gameplay with mixed enemies

## Controls

| Key | Action |
|-----|--------|
| Arrow Keys | Move and rotate |
| Space | Shoot |
| ESC | Return to menu / Exit |
| Y | Restart (on game over) |
| M | Menu (on game over) |

## Headless Training

For faster, unattended training sessions:

```bash
# Default headless training (5x speed)
uv run python run_training.py

# Maximum speed training
uv run python run_training.py --speed 10.0

# Training with graphics for monitoring
uv run python run_training.py --graphics
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

Dependencies (managed by uv):
- pygame
- torch
- numpy

## Project Structure

```
asteroids/
    ai/         # Neural network and training logic
    core/       # Game constants and shared utilities
    entities/   # Player, enemies, asteroids, projectiles
    ui/         # Menu and visual components
main.py         # Game entry point
run_training.py # Headless training script
pyproject.toml  # Project dependencies
```

---

Models are saved to `models/` and training logs to `logs/`.
