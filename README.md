# 🚀 Asteroids with AI Training System

A modern take on the classic Asteroids game featuring **advanced neural network enemies** that learn to hunt the player through reinforcement learning. Watch AI evolve from random movement to sophisticated hunting strategies.

## ✨ Key Features

- **🤖 Self-Learning AI Enemies**: Neural networks that train in real-time through 3 progressive phases
- **🎮 Classic Asteroids Gameplay**: Enhanced with modern graphics and visual effects  
- **⚡ Automated Training System**: Leave it running overnight - AI gets smarter continuously
- **📊 Live Training Metrics**: Watch success rates, phases, and learning progress in real-time
- **🔧 Flexible Training Modes**: Graphics mode for observation, headless mode for maximum speed

## 🎯 Quick Start

### Play the Game
```bash
python main.py
```

### Train AI (Automatic)
```bash
# Watch AI learn (recommended first time)
python run_training.py

# Fast training with graphics  
python run_training.py --speed 3.0

# Maximum speed overnight training
python run_training.py --headless --speed 10.0
```

## 🧠 How the AI Works

The AI progresses through **3 learning phases**:

1. **🟢 Phase 1**: Stationary target - learns basic approach (70% success threshold)
2. **🟠 Phase 2**: Moving target - learns interception (60% success threshold)  
3. **🔴 Phase 3**: Complex patterns - masters advanced hunting (50% success threshold)

**Recent Performance Breakthrough**: After fixing fundamental learning flaws, AI success rate jumped from **0% to 48.1%** in just 27 episodes!

## 🎮 Controls

- **Arrow Keys**: Move/rotate ship • **Spacebar**: Shoot • **ESC**: Exit

## 📦 Installation

```bash
git clone <repository-url>
cd Asteroids
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🔧 Technical Highlights

- **Neural Network**: 5-input policy gradient network with experience replay
- **Real-time Learning**: AI trains during gameplay with live reward shaping
- **Advanced State Space**: Includes movement alignment and angular velocity for efficient learning
- **Auto-saving System**: Preserves training progress with automatic backups

## 📁 Generated Files

- `ai_enemy_brain.pth` - Trained AI model
- `training_log.json` - Episode-by-episode performance data
- `ai_enemy_brain_backup.pth` - Safety backup

## 📚 Documentation

- **[TRAINING_README.md](TRAINING_README.md)** - Comprehensive AI training guide with technical deep-dive
- **Training Performance**: ~2000 episodes/hour (headless mode)
- **Expected Timeline**: 30 min - 8 hours depending on phase complexity

---

**🎯 Pro Tip**: Start with `python run_training.py` to watch the AI learn, then switch to headless mode for overnight training sessions!

Built with Python 3.7+ • Pygame 2.6.1 • PyTorch