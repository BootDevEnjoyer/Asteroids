import pygame
import math
import time
import argparse
import random
from constants import *
from player import *
from asteroid import *
from asteroidfield import *
from starfield import *
from enemy import *
from ai_brain import AIMetricsDisplay, save_global_brain, get_global_brain

current_training_mode = False

def draw_game_over_screen(screen, font, score):
    """Draw the Game Over screen with big white text on black background"""
    screen.fill("black")
    
    # Create large font for Game Over text
    game_over_font = pygame.font.Font(None, 72)
    game_over_text = game_over_font.render("Game Over!", True, "white")
    game_over_rect = game_over_text.get_rect()
    game_over_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)
    
    # Create medium font for score display
    score_font = pygame.font.Font(None, 48)
    score_text = score_font.render(f"Final Score: {score}", True, "white")
    score_rect = score_text.get_rect()
    score_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)
    
    # Create smaller font for restart instruction
    restart_font = pygame.font.Font(None, 36)
    restart_text = restart_font.render("Press Y to Restart", True, "green")
    restart_rect = restart_text.get_rect()
    restart_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)
    
    # Create smaller font for exit instruction
    exit_font = pygame.font.Font(None, 36)
    exit_text = exit_font.render("Press ESC to Exit", True, "white")
    exit_rect = exit_text.get_rect()
    exit_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80)
    
    screen.blit(game_over_text, game_over_rect)
    screen.blit(score_text, score_rect)
    screen.blit(restart_text, restart_rect)
    screen.blit(exit_text, exit_rect)
    pygame.display.flip()

def draw_auto_training_overlay(screen, font, session_stats):
    """Draw training progress overlay for automatic mode"""
    # Training session info panel
    panel_width = 400
    panel_height = 220
    panel_x = 10
    panel_y = 10
    
    # Semi-transparent background
    panel_surface = pygame.Surface((panel_width, panel_height))
    panel_surface.set_alpha(200)
    panel_surface.fill((10, 40, 10))
    screen.blit(panel_surface, (panel_x, panel_y))
    
    # Green border for training mode
    pygame.draw.rect(screen, (0, 255, 0), 
                    (panel_x, panel_y, panel_width, panel_height), 3)
    
    # Title
    title_font = pygame.font.Font(None, 32)
    title_text = title_font.render("🤖 AUTOMATIC TRAINING MODE", True, (0, 255, 0))
    screen.blit(title_text, (panel_x + 10, panel_y + 8))
    
    # Session statistics
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

def reset_game():
    """Reset all game state for a fresh start"""
    # Clear all sprite groups
    updatable = pygame.sprite.Group()
    drawable = pygame.sprite.Group()
    asteroid_group = pygame.sprite.Group()
    shot_group = pygame.sprite.Group()
    enemy_group = pygame.sprite.Group()
    enemy_shot_group = pygame.sprite.Group()
    
    # Reassign containers
    Player.containers = (updatable, drawable) # type: ignore
    Asteroid.containers = (updatable, drawable, asteroid_group) # type: ignore
    AsteroidField.containers = (updatable) # type: ignore
    Shot.containers = (updatable, drawable, shot_group) # type: ignore
    Enemy.containers = (updatable, drawable, enemy_group) # type: ignore
    ShooterEnemy.containers = (updatable, drawable, enemy_group) # type: ignore
    NeuralEnemy.containers = (updatable, drawable, enemy_group) # type: ignore
    EnemyShot.containers = (updatable, drawable, enemy_shot_group) # type: ignore
    
    # Create new player and asteroid field
    x = SCREEN_WIDTH / 2
    y = SCREEN_HEIGHT / 2
    player = Player(x, y)
    asteroid_field = AsteroidField()
    enemy_spawner = EnemySpawner(enemy_group)
    
    return updatable, drawable, asteroid_group, shot_group, enemy_group, enemy_shot_group, player, asteroid_field, enemy_spawner

def main(auto_training=False, training_speed=1.0, headless=False):
    pygame.init()

    # Training session statistics
    session_stats = {
        'start_time': time.time(),
        'total_games': 0,
        'ai_victories': 0,
        'model_saves': 0,
        'auto_restarts': 0,
        'best_reward': 0.0,
        'last_save_time': time.time()
    }

    # Scoring
    font = pygame.font.Font(None, 36)
    score = 0
    game_over = False

    # Training mode variables
    training_mode = auto_training
    current_training_mode = training_mode
    training_time = 0.0   # Track time for player movement patterns
    auto_restart_timer = 0.0  # Timer for automatic restart
    
    # Initialize game objects
    updatable, drawable, asteroid_group, shot_group, enemy_group, enemy_shot_group, player, asteroid_field, enemy_spawner = reset_game()

    # Initialize starfield background
    starfield = Starfield(num_layers=3, stars_per_layer=75)

    ai_display = AIMetricsDisplay()

    clock = pygame.time.Clock()
    dt = 0
    
    # Create screen (unless headless mode)
    if not headless:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Asteroids - AI Training Mode" if auto_training else "Asteroids")
    else:
        # Create minimal surface for headless mode
        screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    print("Starting Asteroids AI Training!")
    print(f"Auto Training: {auto_training}")
    print(f"Training Speed: {training_speed}x")
    print(f"Headless Mode: {headless}")
    print(f"Screen width: {SCREEN_WIDTH}")
    print(f"Screen height: {SCREEN_HEIGHT}")

    # Get AI brain reference and pass session stats
    ai_brain = get_global_brain()
    ai_brain.set_session_stats(session_stats)  # Connect the brain to the session stats
    
    # Initialize session stats from AI brain data for immediate display
    session_stats['total_games'] = ai_brain.training_steps
    session_stats['ai_victories'] = len([r for r in ai_brain.success_history if r > 0])
    session_stats['best_reward'] = ai_brain.best_episode_reward
    
    print(f"AI Brain loaded - Phase: {ai_brain.training_phase}, Episodes: {ai_brain.training_steps}")

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Saving AI brain before exit...")
                save_global_brain()
                session_stats['model_saves'] += 1
                print(f"Training session complete!")
                print(f"Total games: {session_stats['total_games']}")
                print(f"AI victories: {session_stats['ai_victories']} ({session_stats['ai_victories']/max(1,session_stats['total_games']):.1%})")
                return
            if event.type == pygame.KEYDOWN and not auto_training:
                if event.key == pygame.K_t and not game_over:
                    training_mode = not training_mode
                    current_training_mode = training_mode
                    mode_text = "Training Mode" if training_mode else "Normal Mode"
                    print(f"Switched to {mode_text}")
            if game_over and event.type == pygame.KEYDOWN and not auto_training:
                if event.key == pygame.K_ESCAPE:
                    print("Saving AI brain before exit...")
                    save_global_brain()
                    session_stats['model_saves'] += 1
                    return
                elif event.key == pygame.K_y:
                    # Manual restart
                    print("Restarting game!")
                    game_over = False
                    score = 0
                    training_time = 0.0
                    auto_restart_timer = 0.0
                    session_stats['total_games'] += 1
                    session_stats['auto_restarts'] += 1
                    updatable, drawable, asteroid_group, shot_group, enemy_group, enemy_shot_group, player, asteroid_field, enemy_spawner = reset_game()
                    starfield = Starfield(num_layers=3, stars_per_layer=75)
        
        # Automatic restart logic for training mode
        if auto_training and game_over:
            auto_restart_timer += dt
            if auto_restart_timer >= 2.0:  # Wait 2 seconds then restart
                print(f"🔄 Auto-restart #{session_stats['auto_restarts'] + 1}")
                game_over = False
                score = 0
                training_time = 0.0
                auto_restart_timer = 0.0
                session_stats['auto_restarts'] += 1
                
                # Don't double-count total_games and ai_victories - they're updated immediately when game ends
                print(f"🔄 Game ended. Stats: {session_stats['ai_victories']}/{session_stats['total_games']} AI victories")
                
                updatable, drawable, asteroid_group, shot_group, enemy_group, enemy_shot_group, player, asteroid_field, enemy_spawner = reset_game()
                starfield = Starfield(num_layers=3, stars_per_layer=75)
        
        if not game_over:
            # Normal game logic
            screen.fill("black")
            
            # Update and draw starfield first (background) - only if not headless
            if not headless:
                starfield.update(dt, player.velocity if hasattr(player, 'velocity') else None)
                starfield.draw(screen)
                starfield.add_twinkle_effect(screen)

            # Training mode: control player movement based on AI training phase
            if training_mode:
                training_time += dt
                
                # Get current AI training phase from any neural enemy
                current_phase = 1
                for enemy in enemy_group:
                    if hasattr(enemy, 'brain'):
                        current_phase = enemy.brain.training_phase
                        break
                
                # Override player controls based on training phase
                if current_phase == 1:
                    # Phase 1: Stationary player in center
                    center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
                    player.position = pygame.Vector2(center_x, center_y)
                    player.velocity = pygame.Vector2(0, 0)
                    
                elif current_phase == 2:
                    # Phase 2: Slow circular movement
                    center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
                    radius = 100
                    speed = 0.8 * training_speed  # Apply training speed multiplier
                    
                    angle = training_time * speed
                    player.position = pygame.Vector2(
                        center_x + math.cos(angle) * radius,
                        center_y + math.sin(angle) * radius
                    )
                    # Set velocity to match movement for AI state
                    player.velocity = pygame.Vector2(
                        -math.sin(angle) * radius * speed,
                        math.cos(angle) * radius * speed
                    )
                    
                elif current_phase == 3:
                    # Phase 3: Random movement patterns for advanced training
                    center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
                    
                    # More complex movement patterns
                    pattern = int(training_time / 10) % 4  # Change pattern every 10 seconds
                    
                    if pattern == 0:
                        # Figure-8 pattern
                        t = training_time * training_speed
                        scale = 150
                        player.position = pygame.Vector2(
                            center_x + math.sin(t) * scale,
                            center_y + math.sin(t * 2) * scale * 0.5
                        )
                    elif pattern == 1:
                        # Spiral pattern
                        t = training_time * training_speed * 0.5
                        radius = 50 + (t * 10) % 100
                        player.position = pygame.Vector2(
                            center_x + math.cos(t * 2) * radius,
                            center_y + math.sin(t * 2) * radius
                        )
                    elif pattern == 2:
                        # Random walk with bounds
                        if not hasattr(player, 'random_target'):
                            player.random_target = pygame.Vector2(center_x, center_y)
                        
                        # Move toward random target
                        direction = player.random_target - player.position
                        if direction.length() < 20 or random.random() < 0.01:
                            # Pick new random target
                            player.random_target = pygame.Vector2(
                                random.uniform(100, SCREEN_WIDTH - 100),
                                random.uniform(100, SCREEN_HEIGHT - 100)
                            )
                        
                        if direction.length() > 0:
                            direction = direction.normalize()
                            player.velocity = direction * 100 * training_speed
                            player.position += player.velocity * dt
                    else:
                        # Bouncing pattern
                        if not hasattr(player, 'bounce_velocity'):
                            player.bounce_velocity = pygame.Vector2(100, 80) * training_speed
                        
                        player.position += player.bounce_velocity * dt
                        
                        # Bounce off walls
                        if player.position.x < 50 or player.position.x > SCREEN_WIDTH - 50:
                            player.bounce_velocity.x *= -1
                        if player.position.y < 50 or player.position.y > SCREEN_HEIGHT - 50:
                            player.bounce_velocity.y *= -1
                        
                        player.velocity = player.bounce_velocity
                    
            # Update enemy spawner
            enemy_spawner.update(dt * training_speed, player, asteroid_group)
            
            # Update all objects (but player movement may be overridden in training mode)
            if training_mode and current_phase < 3:
                # In training mode phases 1-2, update everything except player movement
                for obj in updatable:
                    if obj != player:
                        obj.update(dt * training_speed)
            else:
                # Normal update or phase 3
                updatable.update(dt * training_speed)
            
            # Check asteroid collisions
            for asteroid in asteroid_group:
                if player.check_collision(asteroid):
                    print("Game over! Hit asteroid")
                    game_over = True
                    player.killed_by_ai = False  # Not an AI victory
                    
                    # Immediately update session stats for live display
                    if auto_training:
                        session_stats['total_games'] += 1
                        print(f"📊 Live Update - Game #{session_stats['total_games']} (Player hit asteroid)")
                    break

                for shot in shot_group:
                    if shot.check_collision(asteroid):
                        shot.kill()

                        if asteroid.radius <= ASTEROID_MIN_RADIUS:
                            score += 2
                        else:
                            score += 1
                        asteroid.split()
            
            # Check enemy collisions with player
            if not game_over:
                for enemy in enemy_group:
                    if player.check_collision(enemy):
                        # ADD THIS: Immediately disable all AI enemies when direct collision occurs
                        for ai_enemy in enemy_group:
                            if hasattr(ai_enemy, 'active'):  # Only disable AI enemies
                                ai_enemy.active = False
                        
                        # MAJOR: Give huge reward to AI enemies for catching player!
                        if hasattr(enemy, 'brain'):
                            enemy.episode_reward += 500.0  # MASSIVE success reward!
                            enemy.brain.store_reward(500.0)
                            enemy.brain.end_episode(enemy.episode_reward, success=True)
                            
                            # Update session stats
                            session_stats['best_reward'] = max(session_stats['best_reward'], enemy.episode_reward)
                            
                            save_global_brain()  # Save the shared brain!
                            session_stats['model_saves'] += 1
                            print(f"🎯 AI Enemy caught player! Final reward: {enemy.episode_reward:.1f}")
                            print("Global AI brain saved after successful hunt!")
                        
                        print("Game over! Enemy caught you!")
                        game_over = True
                        player.killed_by_ai = True  # Mark as AI victory
                        
                        # Immediately update session stats for live display
                        if auto_training:
                            session_stats['total_games'] += 1
                            session_stats['ai_victories'] += 1
                            print(f"🎯 Live Update - AI Victory! Total: {session_stats['ai_victories']}/{session_stats['total_games']}")
                        break
                        
                    # Enemies can be shot by player
                    for shot in shot_group:
                        if shot.check_collision(enemy):
                            shot.kill()
                            enemy.kill()
                            score += 5  # Higher score for defeating enemies
                            break
            
            # Check enemy shot collisions with player
            if not game_over:
                for enemy_shot in enemy_shot_group:
                    if player.check_collision(enemy_shot):
                        print("Game over! Hit by enemy shot!")
                        game_over = True
                        player.killed_by_ai = True  # Mark as AI victory
                        enemy_shot.kill()
                        
                        # Immediately update session stats for live display
                        if auto_training:
                            session_stats['total_games'] += 1
                            session_stats['ai_victories'] += 1
                            print(f"🎯 Live Update - AI Victory! Total: {session_stats['ai_victories']}/{session_stats['total_games']}")
                        break
            
            # Auto-save periodically during training
            if auto_training and time.time() - session_stats['last_save_time'] > 300:  # Every 5 minutes
                print("🔄 Periodic auto-save...")
                save_global_brain()
                session_stats['model_saves'] += 1
                session_stats['last_save_time'] = time.time()
                
            if not game_over and not headless:
                for sprite in drawable:
                    sprite.draw(screen)

                # Draw Score
                score_text = font.render(f"Score: {score}", True, "white")
                score_rect = score_text.get_rect()
                score_rect.centerx = SCREEN_WIDTH // 2
                score_rect.y = 20
                screen.blit(score_text, score_rect)
                
                # Draw training mode info
                if training_mode:
                    # Get current phase
                    current_phase = 1
                    for enemy in enemy_group:
                        if hasattr(enemy, 'brain'):
                            current_phase = enemy.brain.training_phase
                            break
                    
                    phase_names = {1: "Stationary Target", 2: "Slow Moving Target", 3: "Advanced Patterns"}
                    phase_name = phase_names.get(current_phase, "Unknown")
                    
                    training_text = font.render(f"🎓 TRAINING MODE - Phase {current_phase}: {phase_name}", True, "yellow")
                    training_rect = training_text.get_rect()
                    training_rect.centerx = SCREEN_WIDTH // 2
                    training_rect.y = 60
                    screen.blit(training_text, training_rect)
                    
                    if not auto_training:
                        # Instructions for manual mode
                        instruction_font = pygame.font.Font(None, 24)
                        instruction_text = instruction_font.render("Press 'T' to toggle Training Mode", True, "gray")
                        instruction_rect = instruction_text.get_rect()
                        instruction_rect.centerx = SCREEN_WIDTH // 2
                        instruction_rect.y = 90
                        screen.blit(instruction_text, instruction_rect)
                else:
                    mode_text = font.render("🎮 NORMAL MODE", True, "white")
                    mode_rect = mode_text.get_rect()
                    mode_rect.centerx = SCREEN_WIDTH // 2
                    mode_rect.y = 60
                    screen.blit(mode_text, mode_rect)
                
                # Draw session statistics overlay for auto-training
                if auto_training:
                    # Sync session stats with AI brain data for live updates
                    session_stats['total_games'] = ai_brain.training_steps
                    session_stats['ai_victories'] = len([r for r in ai_brain.success_history if r > 0])
                    session_stats['best_reward'] = ai_brain.best_episode_reward
                    
                    draw_auto_training_overlay(screen, font, session_stats)
                
                # Always draw AI metrics using the global brain
                ai_brain = get_global_brain()
                if ai_brain:
                    ai_display.draw_metrics(screen, ai_brain)

                pygame.display.flip()
        else:
            # Game Over screen (only show in manual mode)
            if not auto_training and not headless:
                draw_game_over_screen(screen, font, score)

        # Adjust frame rate based on training speed
        target_fps = int(60 * training_speed)
        dt = clock.tick(target_fps)/1000

def parse_arguments():
    """Parse command line arguments for different training modes"""
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
    
    # Validate arguments
    if args.speed < 0.1 or args.speed > 10.0:
        print("Warning: Training speed should be between 0.1 and 10.0")
        args.speed = max(0.1, min(10.0, args.speed))
    
    if args.headless:
        args.auto_train = True  # Headless implies auto-training
        print("Headless mode enabled - forcing auto-training mode")
    
    try:
        main(auto_training=args.auto_train, training_speed=args.speed, headless=args.headless)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        print("Saving AI brain...")
        save_global_brain()
        print("Training session ended safely")