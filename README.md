# Asteroids Game with Neural Network AI

A modern implementation of the classic Asteroids arcade game featuring intelligent AI enemies that learn pursuit behavior through reinforcement learning. The AI system uses neural networks to evolve from random movement to sophisticated hunting strategies through continuous training.

## Overview

This project demonstrates practical reinforcement learning implementation in a real-time game environment. AI enemies train through a progressive learning system that advances through multiple phases of increasing complexity, developing from basic target approach to advanced interception strategies.

## Features

- **Adaptive AI System**: Neural network enemies that improve through reinforcement learning
- **Progressive Training Phases**: Three-stage learning progression with configurable success thresholds
- **Real-time Learning**: AI trains during gameplay with live performance metrics
- **Flexible Training Modes**: Visual observation mode and high-speed headless training
- **Automatic State Management**: Model persistence with automatic saving and backup systems
- **Performance Monitoring**: Comprehensive training metrics and success rate tracking

## Quick Start

### Playing the Game
```bash
python main.py
```

### AI Training
```bash
# Basic training with visualization
python run_training.py

# Accelerated training with graphics
python run_training.py --speed 3.0

# High-performance headless training
python run_training.py --headless --speed 10.0
```

## Installation

### Requirements
- Python 3.7 or higher
- PyTorch for neural network implementation
- Pygame for game rendering and input handling
- NumPy for numerical computations

### Setup
```bash
git clone <repository-url>
cd Asteroids
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Training System

### Learning Phases
The AI progresses through three distinct learning phases:

**Phase 1: Stationary Target Learning**
- Objective: Learn basic approach behavior toward stationary targets
- Success Threshold: 70% success rate over 30 episodes
- Expected Duration: 30-60 minutes

**Phase 2: Moving Target Interception**
- Objective: Develop interception skills for moving targets
- Success Threshold: 60% success rate over 30 episodes
- Expected Duration: 1-3 hours

**Phase 3: Advanced Hunting Patterns**
- Objective: Master complex pursuit strategies and evasion anticipation
- Success Threshold: 50% success rate over 30 episodes
- Expected Duration: 3-8 hours

### Neural Network Architecture
- **Input Layer**: 5-dimensional state vector (angle, distance, speed, movement alignment, angular velocity)
- **Hidden Layers**: Multi-layer perceptron with progressive dimensionality reduction (64→64→32→1)
- **Output Layer**: Target angle adjustment value constrained to [-1, 1]
- **Training Method**: Policy gradient with value function baseline
- **Optimizer**: Adam with adaptive learning rate scheduling

## Game Controls

| Control | Action |
|---------|--------|
| Arrow Keys | Ship movement and rotation |
| Spacebar | Fire projectiles |
| T | Toggle training mode (during gameplay) |
| ESC | Exit game |
| Y | Restart game (when game over) |

## Technical Architecture

### Core Components
- `main.py` - Main game loop and entry point
- `run_training.py` - Dedicated training script with configuration options
- `ai_brain.py` - Neural network implementation and training logic
- `enemy.py` - AI enemy behaviors and neural network integration
- `player.py` - Player ship mechanics and input handling
- `asteroid.py` - Asteroid physics and collision systems
- `constants.py` - Game configuration and hyperparameters

### Generated Files
- `ai_enemy_brain.pth` - Primary trained model weights
- `ai_enemy_brain_backup.pth` - Backup model for recovery
- `training_log.json` - Detailed episode performance data
- `ai_brain_phase_X_milestone.pth` - Phase completion checkpoints

## Training Performance

### Speed Benchmarks
| Mode | Episodes/Hour | Memory Usage | Use Case |
|------|---------------|--------------|----------|
| Graphics (1x) | ~120 | 50MB | Initial observation |
| Graphics (3x) | ~360 | 60MB | Active monitoring |
| Headless (5x) | ~1200 | 40MB | Background training |
| Headless (10x) | ~2000 | 45MB | Maximum throughput |

### Command Line Options
```bash
# Training script options
python run_training.py [options]
  --speed FLOAT     Training speed multiplier (default: 1.0)
  --headless        Disable graphics for maximum performance
  --help            Show all available options

# Main game options
python main.py [options]
  --auto-train      Enable automatic training mode
  --speed FLOAT     Training speed multiplier
  --headless        Run without graphics display
```

## Development Notes

### Recent Improvements
The AI system underwent significant architectural improvements resulting in a success rate increase from 0% to 48.1% through:
- Enhanced state space representation with movement alignment metrics
- Improved action space design with bounded angle normalization
- Dense reward shaping with behavioral incentives
- Stabilized gradient computation and learning convergence

### Dependencies
- **Pygame 2.6.1**: Game engine and rendering
- **PyTorch 2.7.1**: Neural network framework
- **NumPy 2.3.1**: Numerical computation support

## Documentation

For detailed technical information about the training system, neural network architecture, and performance optimization strategies, see `TRAINING_README.md`.

## License

This project is available for educational and research purposes. Please refer to the license file for specific terms and conditions.