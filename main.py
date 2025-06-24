import pygame
from constants import *
from player import *
from asteroid import *
from asteroidfield import *

def main():
    pygame.init()

    updatable = pygame.sprite.Group()
    drawable = pygame.sprite.Group()
    asteroid_group = pygame.sprite.Group()
    shot_group = pygame.sprite.Group()

    Player.containers = (updatable, drawable) # type: ignore
    Asteroid.containers = (updatable, drawable, asteroid_group) # type: ignore
    AsteroidField.containers = (updatable) # type: ignore
    Shot.containers = (updatable, drawable, shot_group) # type: ignore

    

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
        screen.fill("black")

        updatable.update(dt)
        for asteroid in asteroid_group:
            if player.check_collision(asteroid):
                print("Game over!")
                return

            for shot in shot_group:
                if shot.check_collision(asteroid):
                    shot.kill()
                    asteroid.split()

        for sprite in drawable:
            sprite.draw(screen)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        dt = clock.tick(60)/1000
        
        
if __name__ == "__main__":
    main()