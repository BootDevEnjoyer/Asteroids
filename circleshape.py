import pygame
from constants import SCREEN_WIDTH, SCREEN_HEIGHT
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Base class for game objects
class CircleShape(pygame.sprite.Sprite):
    containers = None
    
    def __init__(self, x: float, y: float, radius: float) -> None:
        # we will be using this later
        if hasattr(self, "containers") and self.containers:
            super().__init__(self.containers)
        else:
            super().__init__()

        self.position: pygame.Vector2 = pygame.Vector2(x, y)
        self.velocity: pygame.Vector2 = pygame.Vector2(0, 0)
        self.radius: float = radius

    def draw(self, screen: pygame.Surface) -> None:
        # sub-classes must override
        pass

    def update(self, dt: float) -> None:
        # sub-classes must override
        pass
    
    def check_collision(self, other: 'CircleShape') -> bool:
        distance = pygame.Vector2.distance_to(self.position, other.position)
        return distance <= self.radius + other.radius

    def wrap_position(self) -> None:
        """Wrap object position around screen edges"""
        # Wrap horizontally
        if self.position.x > SCREEN_WIDTH:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = SCREEN_WIDTH
            
        # Wrap vertically  
        if self.position.y > SCREEN_HEIGHT:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = SCREEN_HEIGHT

