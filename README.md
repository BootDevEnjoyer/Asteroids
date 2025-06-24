# A Cursor Experiment - Asteroids Game

A modern take on the classic Asteroids arcade game built with Python and Pygame. Features enhanced graphics, enemy AI, bouncing bullets, and visual effects.

## Features

- **Classic Asteroids Gameplay**: Navigate your ship through an asteroid field, destroying asteroids and avoiding collisions
- **Enhanced Ship Design**: Detailed ship graphics with thrust particle effects and multiple visual elements
- **Smart Enemies**: Two types of enemies with different behaviors:
  - **Chasers**: Follow the player when in detection range with trail effects
  - **Shooters**: Move like asteroids but shoot red projectiles at the player
- **Bouncing Bullets**: Player shots bounce off screen edges up to 3 times with visual feedback
- **Animated Starfield Background**: Multi-layer parallax starfield with twinkling effects
- **Scoring System**: Earn points for destroying asteroids and enemies
- **Game Over & Restart**: Full game restart functionality

## Controls

- **Arrow Keys**: Move and rotate the ship
  - ↑ Up: Thrust forward
  - ↓ Down: Thrust backward
  - ← Left: Rotate left
  - → Right: Rotate right
- **Spacebar**: Shoot
- **ESC**: Exit game (or exit from game over screen)
- **Y**: Restart game (on game over screen)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Asteroids.git
cd Asteroids
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Game

```bash
python main.py
```

## Gameplay

- Destroy asteroids to earn points (larger asteroids = more points)
- Avoid collisions with asteroids and enemies
- Watch out for red enemy shots
- Use bouncing bullets strategically to hit hard-to-reach targets
- Survive as long as possible to achieve a high score

## Technical Details

- Built with Python 3 and Pygame
- Object-oriented design with modular components
- Smooth 60 FPS gameplay
- Screen wrapping for seamless movement
- Collision detection with optimized radius checking

## Files Structure

- `main.py` - Main game loop and state management
- `player.py` - Player ship with controls and visual effects
- `asteroid.py` - Asteroid behavior and splitting mechanics
- `shot.py` - Bullet physics and bouncing system
- `enemy.py` - Enemy AI and shooting mechanics
- `circleshape.py` - Base class for all circular game objects
- `constants.py` - Game configuration and parameters

## Requirements

- Python 3.7+
- Pygame 2.6.1

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.