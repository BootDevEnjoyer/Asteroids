"""Game session state container and management functions."""

import time
import pygame
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from asteroids.core.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from asteroids.entities.player import Player
from asteroids.entities.asteroid import Asteroid
from asteroids.entities.asteroidfield import AsteroidField
from asteroids.entities.shot import Shot
from asteroids.entities.enemy import Enemy, ShooterEnemy, NeuralEnemy, EnemyShot, EnemySpawner
from asteroids.ui.starfield import Starfield
from asteroids.ai.brain import AIMetricsDisplay, get_global_brain


def _default_session_stats() -> Dict[str, Any]:
    """Create default session statistics dictionary."""
    return {
        'start_time': time.time(),
        'total_games': 0,
        'ai_victories': 0,
        'model_saves': 0,
        'auto_restarts': 0,
        'best_reward': 0.0,
        'last_save_time': time.time()
    }


@dataclass
class GameContext:
    """
    Container for all active game session state.
    
    Encapsulates sprite groups, game objects, and session tracking
    that were previously scattered as local variables in main().
    """
    # Sprite groups
    updatable: pygame.sprite.Group
    drawable: pygame.sprite.Group
    asteroid_group: pygame.sprite.Group
    shot_group: pygame.sprite.Group
    enemy_group: pygame.sprite.Group
    enemy_shot_group: pygame.sprite.Group

    # Game objects
    player: Player
    asteroid_field: AsteroidField
    enemy_spawner: EnemySpawner
    starfield: Starfield
    ai_display: AIMetricsDisplay

    # Session tracking
    session_stats: Dict[str, Any] = field(default_factory=_default_session_stats)
    score: int = 0
    training_time: float = 0.0
    auto_restart_timer: float = 0.0
    game_over: bool = False
    training_mode: bool = False


def create_sprite_groups() -> tuple:
    """
    Create fresh sprite groups and configure class containers.
    
    Returns:
        Tuple of (updatable, drawable, asteroid_group, shot_group, 
                  enemy_group, enemy_shot_group)
    """
    updatable = pygame.sprite.Group()
    drawable = pygame.sprite.Group()
    asteroid_group = pygame.sprite.Group()
    shot_group = pygame.sprite.Group()
    enemy_group = pygame.sprite.Group()
    enemy_shot_group = pygame.sprite.Group()

    # Configure sprite class containers
    Player.containers = (updatable, drawable)  # type: ignore
    Asteroid.containers = (updatable, drawable, asteroid_group)  # type: ignore
    AsteroidField.containers = (updatable,)  # type: ignore
    Shot.containers = (updatable, drawable, shot_group)  # type: ignore
    Enemy.containers = (updatable, drawable, enemy_group)  # type: ignore
    ShooterEnemy.containers = (updatable, drawable, enemy_group)  # type: ignore
    NeuralEnemy.containers = (updatable, drawable, enemy_group)  # type: ignore
    EnemyShot.containers = (updatable, drawable, enemy_shot_group)  # type: ignore

    return updatable, drawable, asteroid_group, shot_group, enemy_group, enemy_shot_group


def create_game_context(
    enemy_type: str = "neural",
    preserve_session_stats: Optional[Dict[str, Any]] = None,
    training_mode: bool = False
) -> GameContext:
    """
    Create a new game context with fresh game objects.
    
    Args:
        enemy_type: Type of enemies to spawn ("neural", "mixed", "none")
        preserve_session_stats: Optional existing stats to preserve across restarts
        training_mode: Whether AI-controlled training behaviors are active
        
    Returns:
        Fully initialized GameContext
    """
    # Create sprite groups
    groups = create_sprite_groups()
    updatable, drawable, asteroid_group, shot_group, enemy_group, enemy_shot_group = groups

    # Create player at screen center
    player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    # Create field spawners (disable asteroids during AI training)
    asteroid_field = AsteroidField(enabled=not training_mode)
    enemy_spawner = EnemySpawner(enemy_group, enemy_type=enemy_type, training_mode=training_mode)

    # Create visual components
    starfield = Starfield(num_layers=3, stars_per_layer=75)
    ai_display = AIMetricsDisplay()

    # Initialize or preserve session stats
    if preserve_session_stats is not None:
        session_stats = preserve_session_stats
    else:
        session_stats = _default_session_stats()
        # Sync with AI brain for accurate initial display
        ai_brain = get_global_brain()
        session_stats['total_games'] = ai_brain.training_steps
        session_stats['ai_victories'] = len([r for r in ai_brain.success_history if r > 0])
        session_stats['best_reward'] = ai_brain.best_episode_reward

    return GameContext(
        updatable=updatable,
        drawable=drawable,
        asteroid_group=asteroid_group,
        shot_group=shot_group,
        enemy_group=enemy_group,
        enemy_shot_group=enemy_shot_group,
        player=player,
        asteroid_field=asteroid_field,
        enemy_spawner=enemy_spawner,
        starfield=starfield,
        ai_display=ai_display,
        session_stats=session_stats,
        score=0,
        training_time=0.0,
        auto_restart_timer=0.0,
        game_over=False,
        training_mode=training_mode,
    )


def reset_game_context(ctx: GameContext, enemy_type: Optional[str] = None) -> GameContext:
    """
    Reset game context for a new round while preserving session stats.
    
    Args:
        ctx: Existing context to reset
        enemy_type: Override enemy type, or None to preserve from existing context
        
    Returns:
        Fresh GameContext with preserved session statistics
    """
    # Preserve enemy type from existing spawner if not overridden
    actual_enemy_type = enemy_type if enemy_type else ctx.enemy_spawner.enemy_type
    
    return create_game_context(
        enemy_type=actual_enemy_type,
        preserve_session_stats=ctx.session_stats,
        training_mode=ctx.training_mode,
    )

