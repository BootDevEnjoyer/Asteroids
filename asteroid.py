from circleshape import *
from constants import *
import random
import pygame

class Asteroid(CircleShape):
    def __init__(self, x, y, radius):
        super().__init__(x, y, radius)
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))

    def draw(self, screen):
        pygame.draw.circle(screen, "white", self.position, self.radius, 2)

    def update(self, dt):
        self.position += self.velocity * dt
        # Handle screen bouncing instead of wrapping
        self.bounce_off_edges()
    
    def bounce_off_edges(self):
        """Make asteroid bounce off screen edges"""
        # Bounce off left or right edges
        if (self.position.x - self.radius <= 0 and self.velocity.x < 0) or \
           (self.position.x + self.radius >= SCREEN_WIDTH and self.velocity.x > 0):
            self.velocity.x = -self.velocity.x
            # Keep asteroid within bounds
            if self.position.x - self.radius <= 0:
                self.position.x = self.radius
            else:
                self.position.x = SCREEN_WIDTH - self.radius
                
        # Bounce off top or bottom edges
        if (self.position.y - self.radius <= 0 and self.velocity.y < 0) or \
           (self.position.y + self.radius >= SCREEN_HEIGHT and self.velocity.y > 0):
            self.velocity.y = -self.velocity.y
            # Keep asteroid within bounds
            if self.position.y - self.radius <= 0:
                self.position.y = self.radius
            else:
                self.position.y = SCREEN_HEIGHT - self.radius
    
    def split(self):
        self.kill()
        if self.radius <= ASTEROID_MIN_RADIUS:
            return
        
        angle = random.uniform(20,50)
        new_vel_1 = self.velocity.rotate(angle)
        new_vel_2 = self.velocity.rotate(-angle)

        new_radius = self.radius - ASTEROID_MIN_RADIUS
        new_asteroid_1 = Asteroid(self.position.x, self.position.y, new_radius) # type: ignore
        new_asteroid_1.velocity = new_vel_1 * 1.2
        new_asteroid_2 = Asteroid(self.position.x, self.position.y, new_radius) # type: ignore 
        new_asteroid_2.velocity = new_vel_2 * 1.2

