import pygame
from constants import *
from player import *
from asteroid import *
from asteroidfield import *
from starfield import *

def draw_game_over_screen(screen, font, score):
    """Draw the Game Over screen with big white text on black background"""
    screen.fill("black")
    
    # Create large font for Game Over text
    game_over_font = pygame.font.Font(None, 72)
    game_over_text = game_over_font.render("Game Over!", True, "white")
    game_over_rect = game_over_text.get_rect()
    game_over_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80)
    
    # Create medium font for score display
    score_font = pygame.font.Font(None, 48)
    score_text = score_font.render(f"Final Score: {score}", True, "white")
    score_rect = score_text.get_rect()
    score_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    
    # Create smaller font for exit instruction
    exit_font = pygame.font.Font(None, 36)
    exit_text = exit_font.render("Press ESC to Exit", True, "white")
    exit_rect = exit_text.get_rect()
    exit_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80)
    
    screen.blit(game_over_text, game_over_rect)
    screen.blit(score_text, score_rect)
    screen.blit(exit_text, exit_rect)
    pygame.display.flip()

def main():
    pygame.init()

    # Scoring
    font = pygame.font.Font(None, 36)
    score = 0
    game_over = False

    updatable = pygame.sprite.Group()
    drawable = pygame.sprite.Group()
    asteroid_group = pygame.sprite.Group()
    shot_group = pygame.sprite.Group()

    Player.containers = (updatable, drawable) # type: ignore
    Asteroid.containers = (updatable, drawable, asteroid_group) # type: ignore
    AsteroidField.containers = (updatable) # type: ignore
    Shot.containers = (updatable, drawable, shot_group) # type: ignore

    

    # Initialize starfield background
    starfield = Starfield(num_layers=3, stars_per_layer=150)
    
    # Draw player and asteroid field
    x = SCREEN_WIDTH / 2
    y = SCREEN_HEIGHT / 2
    player = Player(x, y)
    asteroid_field = AsteroidField()

    clock = pygame.time.Clock()
    dt = 0
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        
    print("Starting Asteroids!")
    print(f"Screen width: {SCREEN_WIDTH}")
    print(f"Screen height: {SCREEN_HEIGHT}")

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
        
        if not game_over:
            # Normal game logic
            screen.fill("black")
            
            # Update and draw starfield first (background)
            starfield.update(dt, player.velocity if hasattr(player, 'velocity') else None)
            starfield.draw(screen)
            starfield.add_twinkle_effect(screen)

            updatable.update(dt)
            for asteroid in asteroid_group:
                if player.check_collision(asteroid):
                    print("Game over!")
                    game_over = True
                    break

                for shot in shot_group:
                    if shot.check_collision(asteroid):
                        shot.kill()

                        if asteroid.radius <= ASTEROID_MIN_RADIUS:
                            score += 2
                        else:
                            score += 1
                        asteroid.split()

            if not game_over:
                for sprite in drawable:
                    sprite.draw(screen)

                # Draw Score
                score_text = font.render(f"Score: {score}", True, "white")
                score_rect = score_text.get_rect()
                score_rect.centerx = SCREEN_WIDTH // 2
                score_rect.y = 20
                screen.blit(score_text, score_rect)

                pygame.display.flip()
        else:
            # Game Over screen
            draw_game_over_screen(screen, font, score)

        dt = clock.tick(60)/1000
        
        
if __name__ == "__main__":
    main()