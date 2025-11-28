import pygame
import math
import time
import argparse
import random
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from asteroids.core.constants import *
from asteroids.ai.brain import save_global_brain, get_global_brain, reset_global_brain
from asteroids.ai.debug_rl import debug_episode_end, debug_dump
from asteroids.ui.menu import MenuScreen
from asteroids.core.game_context import GameContext, create_game_context, reset_game_context


class GameState(Enum):
    """Defines the distinct states the game can be in."""
    MENU = auto()
    RL_TRAINING = auto()
    RL_SHOWCASE = auto()
    CLASSIC_PLAY = auto()
    GAME_OVER = auto()


@dataclass
class GameConfig:
    """Configuration for a specific game mode, derived from GameState."""
    enemy_type: str
    training_enabled: bool
    player_controlled: bool
    auto_restart: bool
    show_metrics: bool
    speed_multiplier: float

    @classmethod
    def for_state(cls, state: GameState, speed: float = 1.0) -> "GameConfig":
        """Factory method to create config matching a game state."""
        configs = {
            GameState.MENU: cls(
                enemy_type="none",
                training_enabled=False,
                player_controlled=True,
                auto_restart=False,
                show_metrics=False,
                speed_multiplier=1.0
            ),
            GameState.RL_TRAINING: cls(
                enemy_type="neural",
                training_enabled=True,
                player_controlled=False,
                auto_restart=True,
                show_metrics=True,
                speed_multiplier=speed
            ),
            GameState.RL_SHOWCASE: cls(
                enemy_type="neural",
                training_enabled=False,
                player_controlled=False,
                auto_restart=False,
                show_metrics=True,
                speed_multiplier=1.0
            ),
            GameState.CLASSIC_PLAY: cls(
                enemy_type="mixed",
                training_enabled=False,
                player_controlled=True,
                auto_restart=False,
                show_metrics=False,
                speed_multiplier=1.0
            ),
            GameState.GAME_OVER: cls(
                enemy_type="none",
                training_enabled=False,
                player_controlled=False,
                auto_restart=False,
                show_metrics=False,
                speed_multiplier=1.0
            ),
        }
        return configs[state]


def draw_game_over_screen(screen, font, score, show_menu_option=True):
    """Render the game over screen with options."""
    screen.fill("black")
    
    game_over_font = pygame.font.Font(None, 72)
    game_over_text = game_over_font.render("Game Over!", True, "white")
    game_over_rect = game_over_text.get_rect()
    game_over_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)
    
    score_font = pygame.font.Font(None, 48)
    score_text = score_font.render(f"Final Score: {score}", True, "white")
    score_rect = score_text.get_rect()
    score_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)
    
    restart_font = pygame.font.Font(None, 36)
    restart_text = restart_font.render("Press Y to Restart", True, "green")
    restart_rect = restart_text.get_rect()
    restart_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)
    
    option_font = pygame.font.Font(None, 36)
    
    if show_menu_option:
        menu_text = option_font.render("Press M for Menu", True, "cyan")
        menu_rect = menu_text.get_rect()
        menu_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80)
        screen.blit(menu_text, menu_rect)
        y_offset = 120
    else:
        y_offset = 80
    
    exit_text = option_font.render("Press ESC to Exit", True, "white")
    exit_rect = exit_text.get_rect()
    exit_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + y_offset)
    
    screen.blit(game_over_text, game_over_rect)
    screen.blit(score_text, score_rect)
    screen.blit(restart_text, restart_rect)
    screen.blit(exit_text, exit_rect)
    pygame.display.flip()

def draw_auto_training_overlay(screen, font, session_stats):
    """render training progress overlay for automatic mode"""
    # training session info panel dimensions
    panel_width = 400
    panel_height = 220
    panel_x = 10
    panel_y = 10
    
    # semi-transparent background
    panel_surface = pygame.Surface((panel_width, panel_height))
    panel_surface.set_alpha(200)
    panel_surface.fill((10, 40, 10))
    screen.blit(panel_surface, (panel_x, panel_y))
    
    # green border for training mode
    pygame.draw.rect(screen, (0, 255, 0), 
                    (panel_x, panel_y, panel_width, panel_height), 3)
    
    # title display
    title_font = pygame.font.Font(None, 32)
    title_text = title_font.render("AUTOMATIC TRAINING MODE", True, (0, 255, 0))
    screen.blit(title_text, (panel_x + 10, panel_y + 8))
    
    # session statistics calculation
    small_font = pygame.font.Font(None, 24)
    y_offset = panel_y + 45
    
    session_time = time.time() - session_stats['start_time']
    hours = int(session_time // 3600)
    minutes = int((session_time % 3600) // 60)
    seconds = int(session_time % 60)
    
    stats_lines = [
        f"Session Time: {hours:02d}:{minutes:02d}:{seconds:02d}",
        f"Total Games: {session_stats['total_games']}",
        f"AI Victories: {session_stats['ai_victories']}",
        f"Current Win Rate: {session_stats['ai_victories']/max(1,session_stats['total_games']):.1%}",
        f"Model Saves: {session_stats['model_saves']}",
        f"Auto-Restarts: {session_stats['auto_restarts']}",
        f"Best Episode Reward: {session_stats['best_reward']:.1f}",
        f"Games per Hour: {session_stats['total_games']/(session_time/3600) if session_time > 60 else 0:.1f}"
    ]
    
    for line in stats_lines:
        text = small_font.render(line, True, "white")
        screen.blit(text, (panel_x + 15, y_offset))
        y_offset += 22

def main(auto_training=False, training_speed=1.0, headless=False):
    pygame.init()

    clock = pygame.time.Clock()
    dt = 0
    font = pygame.font.Font(None, 36)
    
    # Create screen
    if not headless:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Asteroids")
    else:
        screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    # Determine initial state based on CLI arguments
    if headless or auto_training:
        # CLI mode: skip menu, go directly to training
        current_state = GameState.RL_TRAINING
        print("Starting Asteroids AI Training!")
        print(f"Auto Training: {auto_training}")
        print(f"Training Speed: {training_speed}x")
        print(f"Headless Mode: {headless}")
    else:
        # Interactive mode: start with menu
        current_state = GameState.MENU
        print("Starting Asteroids - Use menu to select mode")
    
    print(f"Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    
    # Create config for current state
    config = GameConfig.for_state(current_state, training_speed)
    
    # Initialize menu (always created for potential return-to-menu)
    menu = MenuScreen()
    
    # Game context (created when entering a game state)
    ctx: Optional[GameContext] = None
    
    # Previous state for tracking transitions
    previous_state: Optional[GameState] = None
    
    # Track the last game mode for restart functionality
    last_game_mode: GameState = GameState.RL_TRAINING
    
    # Get AI brain reference
    ai_brain = get_global_brain()
    print(f"AI Brain loaded - Phase: {ai_brain.training_phase}, Episodes: {ai_brain.training_steps}")

    running = True
    while running:
        # Handle state transitions
        if current_state != previous_state:
            previous_state = current_state
            config = GameConfig.for_state(current_state, training_speed)
            
            # Create game context when entering a game state
            if current_state in (GameState.RL_TRAINING, GameState.RL_SHOWCASE, GameState.CLASSIC_PLAY):
                training_mode = not config.player_controlled
                ctx = create_game_context(enemy_type=config.enemy_type, training_mode=training_mode)
                ai_brain.set_session_stats(ctx.session_stats)
                last_game_mode = current_state
                
                # Set window title based on state
                if not headless:
                    titles = {
                        GameState.RL_TRAINING: "Asteroids - AI Training Mode",
                        GameState.RL_SHOWCASE: "Asteroids - AI Showcase",
                        GameState.CLASSIC_PLAY: "Asteroids - Classic Mode",
                    }
                    pygame.display.set_caption(titles.get(current_state, "Asteroids"))
                
                print(f"Entered {current_state.name} mode")
        
        # Process events
        next_state: Optional[GameState] = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Saving AI brain before exit...")
                save_global_brain()
                debug_dump()
                if ctx:
                    ctx.session_stats['model_saves'] += 1
                    print(f"Training session complete!")
                    print(f"Total games: {ctx.session_stats['total_games']}")
                running = False
                break
            
            # State-specific event handling
            if current_state == GameState.MENU:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break
                # Handle menu button clicks
                selected = menu.handle_event(event)
                if selected:
                    if selected == "RESET_AI":
                        # Archive and reset AI model without changing state
                        ai_brain, archived = reset_global_brain()
                        print(f"AI model reset complete. Ready for fresh training!")
                    else:
                        next_state = GameState[selected]
            
            elif current_state == GameState.GAME_OVER:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("Saving AI brain before exit...")
                        save_global_brain()
                        debug_dump()
                        running = False
                        break
                    elif event.key == pygame.K_y and ctx:
                        # Restart current game mode
                        ctx = reset_game_context(ctx)
                        ctx.game_over = False
                        ctx.session_stats['auto_restarts'] += 1
                        next_state = last_game_mode
                        print("Restarting game!")
                    elif event.key == pygame.K_m:
                        next_state = GameState.MENU
                        ctx = None
            
            elif current_state in (GameState.RL_TRAINING, GameState.RL_SHOWCASE, GameState.CLASSIC_PLAY):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        next_state = GameState.MENU
                        print("Returning to menu...")
        
        if not running:
            break
        
        # Apply state transition
        if next_state:
            current_state = next_state
            continue
        
        # State-specific update and render
        if current_state == GameState.MENU:
            menu.update(dt)
            if not headless:
                menu.draw(screen)
                pygame.display.flip()
        
        elif current_state == GameState.GAME_OVER:
            # Check if the game mode we came from had auto_restart enabled
            game_mode_config = GameConfig.for_state(last_game_mode, training_speed)
            
            if game_mode_config.auto_restart and ctx:
                ctx.auto_restart_timer += dt
                if ctx.auto_restart_timer >= 2.0:
                    print(f"Auto-restart #{ctx.session_stats['auto_restarts'] + 1}")
                    ctx = reset_game_context(ctx)
                    ctx.session_stats['auto_restarts'] += 1
                    # Return to the game state we came from
                    current_state = last_game_mode
                    continue
            
            if not headless and ctx:
                draw_game_over_screen(screen, font, ctx.score, show_menu_option=not game_mode_config.auto_restart)
        
        elif current_state in (GameState.RL_TRAINING, GameState.RL_SHOWCASE, GameState.CLASSIC_PLAY) and ctx:
            # Active gameplay update
            if not ctx.game_over:
                screen.fill("black")
                
                # Render starfield
                if not headless:
                    ctx.starfield.update(dt, ctx.player.velocity if hasattr(ctx.player, 'velocity') else None)
                    ctx.starfield.draw(screen)
                    ctx.starfield.add_twinkle_effect(screen)
                
                # Training mode: control player movement based on AI training phase
                if not config.player_controlled:
                    ctx.training_time += dt
                    
                    current_phase = 1
                    for enemy in ctx.enemy_group:
                        if hasattr(enemy, 'brain'):
                            current_phase = enemy.brain.training_phase
                            break
                    
                    # Override player controls based on training phase
                    if current_phase == 1:
                        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
                        ctx.player.position = pygame.Vector2(center_x, center_y)
                        ctx.player.velocity = pygame.Vector2(0, 0)
                    elif current_phase == 2:
                        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
                        radius = 100
                        speed = 0.8 * config.speed_multiplier
                        angle = ctx.training_time * speed
                        ctx.player.position = pygame.Vector2(
                            center_x + math.cos(angle) * radius,
                            center_y + math.sin(angle) * radius
                        )
                        ctx.player.velocity = pygame.Vector2(
                            -math.sin(angle) * radius * speed,
                            math.cos(angle) * radius * speed
                        )
                    elif current_phase == 3:
                        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
                        pattern = int(ctx.training_time / 10) % 4
                        
                        if pattern == 0:
                            t = ctx.training_time * config.speed_multiplier
                            scale = 150
                            ctx.player.position = pygame.Vector2(
                                center_x + math.sin(t) * scale,
                                center_y + math.sin(t * 2) * scale * 0.5
                            )
                        elif pattern == 1:
                            t = ctx.training_time * config.speed_multiplier * 0.5
                            radius = 50 + (t * 10) % 100
                            ctx.player.position = pygame.Vector2(
                                center_x + math.cos(t * 2) * radius,
                                center_y + math.sin(t * 2) * radius
                            )
                        elif pattern == 2:
                            if not hasattr(ctx.player, 'random_target') or ctx.player.random_target is None:
                                ctx.player.random_target = pygame.Vector2(center_x, center_y)
                            direction = ctx.player.random_target - ctx.player.position
                            if direction.length() < 20 or random.random() < 0.01:
                                ctx.player.random_target = pygame.Vector2(
                                    random.uniform(100, SCREEN_WIDTH - 100),
                                    random.uniform(100, SCREEN_HEIGHT - 100)
                                )
                            if direction.length() > 0:
                                direction = direction.normalize()
                                ctx.player.velocity = direction * 100 * config.speed_multiplier
                                ctx.player.position += ctx.player.velocity * dt
                        else:
                            if not hasattr(ctx.player, 'bounce_velocity') or ctx.player.bounce_velocity is None:
                                ctx.player.bounce_velocity = pygame.Vector2(100, 80) * config.speed_multiplier
                            ctx.player.position += ctx.player.bounce_velocity * dt
                            if ctx.player.position.x < 50 or ctx.player.position.x > SCREEN_WIDTH - 50:
                                ctx.player.bounce_velocity.x *= -1
                            if ctx.player.position.y < 50 or ctx.player.position.y > SCREEN_HEIGHT - 50:
                                ctx.player.bounce_velocity.y *= -1
                            ctx.player.velocity = ctx.player.bounce_velocity
                
                effective_dt = dt * config.speed_multiplier
                ctx.enemy_spawner.update(effective_dt, ctx.player, ctx.asteroid_group)
                
                # Selective updating based on training phase
                if not config.player_controlled and current_phase < 3:
                    for obj in ctx.updatable:
                        if obj != ctx.player:
                            obj.update(effective_dt)
                else:
                    ctx.updatable.update(effective_dt)
                
                # Collision detection: asteroids vs player
                for asteroid in ctx.asteroid_group:
                    if ctx.player.check_collision(asteroid):
                        print("Game over! Hit asteroid")
                        ctx.game_over = True
                        ctx.player.killed_by_ai = False
                        if config.auto_restart:
                            ctx.session_stats['total_games'] += 1
                        break
                    for shot in ctx.shot_group:
                        if shot.check_collision(asteroid):
                            shot.kill()
                            ctx.score += 2 if asteroid.radius <= ASTEROID_MIN_RADIUS else 1
                            asteroid.split()
                
                # Collision detection: enemies vs player
                if not ctx.game_over:
                    for enemy in ctx.enemy_group:
                        if ctx.player.check_collision(enemy):
                            for ai_enemy in ctx.enemy_group:
                                if hasattr(ai_enemy, 'active'):
                                    ai_enemy.active = False
                            if hasattr(enemy, 'brain'):
                                enemy.episode_reward += 500.0
                                enemy.brain.store_reward(500.0)
                                debug_episode_end(enemy.enemy_id, enemy.episode_reward, True, enemy.episode_length)
                                enemy.brain.end_episode(enemy.episode_reward, success=True)
                                enemy.episode_ended = True
                                ctx.session_stats['best_reward'] = max(ctx.session_stats['best_reward'], enemy.episode_reward)
                                save_global_brain()
                                debug_dump()
                                ctx.session_stats['model_saves'] += 1
                                print(f"AI Enemy caught player! Final reward: {enemy.episode_reward:.1f}")
                            print("Game over! Enemy caught you!")
                            ctx.game_over = True
                            ctx.player.killed_by_ai = True
                            if config.auto_restart:
                                ctx.session_stats['total_games'] += 1
                                ctx.session_stats['ai_victories'] += 1
                            break
                        for shot in ctx.shot_group:
                            if shot.check_collision(enemy):
                                shot.kill()
                                enemy.kill()
                                ctx.score += 5
                                break
                
                # Collision detection: enemy shots vs player
                if not ctx.game_over:
                    for enemy_shot in ctx.enemy_shot_group:
                        if ctx.player.check_collision(enemy_shot):
                            print("Game over! Hit by enemy shot!")
                            ctx.game_over = True
                            ctx.player.killed_by_ai = True
                            enemy_shot.kill()
                            if config.auto_restart:
                                ctx.session_stats['total_games'] += 1
                                ctx.session_stats['ai_victories'] += 1
                            break
                
                # Periodic auto-save
                if config.training_enabled and time.time() - ctx.session_stats['last_save_time'] > 300:
                    print("Periodic auto-save...")
                    save_global_brain()
                    ctx.session_stats['model_saves'] += 1
                    ctx.session_stats['last_save_time'] = time.time()
                
                # Render game
                if not ctx.game_over and not headless:
                    for sprite in ctx.drawable:
                        sprite.draw(screen)
                    
                    score_text = font.render(f"Score: {ctx.score}", True, "white")
                    score_rect = score_text.get_rect()
                    score_rect.centerx = SCREEN_WIDTH // 2
                    score_rect.y = 20
                    screen.blit(score_text, score_rect)
                    
                    # Mode status display
                    if not config.player_controlled:
                        current_phase = 1
                        for enemy in ctx.enemy_group:
                            if hasattr(enemy, 'brain'):
                                current_phase = enemy.brain.training_phase
                                break
                        phase_names = {1: "Stationary Target", 2: "Slow Moving Target", 3: "Advanced Patterns"}
                        phase_name = phase_names.get(current_phase, "Unknown")
                        training_text = font.render(f"TRAINING MODE - Phase {current_phase}: {phase_name}", True, "yellow")
                        training_rect = training_text.get_rect()
                        training_rect.centerx = SCREEN_WIDTH // 2
                        training_rect.y = 60
                        screen.blit(training_text, training_rect)
                    else:
                        mode_text = font.render("CLASSIC MODE - Press ESC for menu", True, "white")
                        mode_rect = mode_text.get_rect()
                        mode_rect.centerx = SCREEN_WIDTH // 2
                        mode_rect.y = 60
                        screen.blit(mode_text, mode_rect)
                    
                    # Training overlay
                    if config.show_metrics:
                        draw_auto_training_overlay(screen, font, ctx.session_stats)
                        ctx.ai_display.draw_metrics(screen, ai_brain)
                    
                    pygame.display.flip()
            else:
                # Game over - transition to GAME_OVER state
                current_state = GameState.GAME_OVER
                continue
        
        target_fps = int(60 * config.speed_multiplier)
        dt = clock.tick(target_fps) / 1000

def parse_arguments():
    """parse command line arguments for different training modes"""
    parser = argparse.ArgumentParser(description='Asteroids AI Training Game')
    parser.add_argument('--auto-train', action='store_true', 
                       help='Enable automatic training mode (no user intervention required)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Training speed multiplier (default: 1.0, max recommended: 5.0)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no graphics, maximum speed)')
    return parser.parse_args()
        
if __name__ == "__main__":
    args = parse_arguments()
    
    # validate and constrain arguments
    if args.speed < 0.1 or args.speed > 20.0:
        print("Warning: Training speed should be between 0.1 and 20.0")
        args.speed = max(0.1, min(20.0, args.speed))
    
    if args.headless:
        args.auto_train = True  # headless mode requires auto-training
        print("Headless mode enabled - forcing auto-training mode")
    
    try:
        main(auto_training=args.auto_train, training_speed=args.speed, headless=args.headless)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving AI brain...")
        save_global_brain()
        print("Training session ended safely")