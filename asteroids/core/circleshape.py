"""Base classes for circular game objects with collision detection."""

import pygame
from asteroids.core.constants import SCREEN_WIDTH, SCREEN_HEIGHT

class CircleShape(pygame.sprite.Sprite):
    # base class for circular game objects
    containers = None
    
    def __init__(self, x: float, y: float, radius: float) -> None:
        # sprite group management for pygame collision systems
        if hasattr(self, "containers") and self.containers:
            super().__init__(self.containers)
        else:
            super().__init__()

        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0, 0)
        self.radius = radius

    def draw(self, screen: pygame.Surface) -> None:
        # render object to screen
        raise NotImplementedError("Subclasses must implement draw method")

    def update(self, dt: float) -> None:
        # update object state based on delta time
        raise NotImplementedError("Subclasses must implement update method")
    
    def check_collision(self, other: 'CircleShape') -> bool:
        # check collision using distance between centers vs sum of radii
        distance = pygame.Vector2.distance_to(self.position, other.position)
        return distance <= self.radius + other.radius

    def wrap_position(self) -> None:
        # teleport to opposite edge when leaving screen
        if self.position.x > SCREEN_WIDTH:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = SCREEN_WIDTH
            
        if self.position.y > SCREEN_HEIGHT:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = SCREEN_HEIGHT

