"""Asteroid game objects with irregular shapes and collision physics."""

import random
import pygame
import math
from asteroids.core.circleshape import CircleShape
from asteroids.core.constants import *

class Asteroid(CircleShape):
    """Irregular asteroid with jagged shape and bouncing physics."""

    def __init__(self, x, y, radius):
        super().__init__(x, y, radius)
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-50, 50)
        self.shape_points = self.generate_jagged_shape()

    def generate_jagged_shape(self):
        """Generate irregular polygon points for asteroid shape."""
        points = []
        num_points = random.randint(8, 12)

        for i in range(num_points):
            angle = (360 / num_points) * i + random.uniform(-15, 15)
            point_radius = self.radius * random.uniform(0.7, 1.3)
            x = math.cos(math.radians(angle)) * point_radius
            y = math.sin(math.radians(angle)) * point_radius
            points.append(pygame.Vector2(x, y))

        return points

    def get_rotated_points(self):
        """Get the asteroid's shape points rotated by current rotation."""
        rotated_points = []
        for point in self.shape_points:
            rotated_point = point.rotate(self.rotation)
            world_point = self.position + rotated_point
            rotated_points.append(world_point)
        return rotated_points

    def draw(self, screen):
        points = self.get_rotated_points()
        if len(points) >= 3:
            pygame.draw.polygon(screen, "white", points, 2)
            if len(points) >= 6:
                detail_points = points[::2]
                if len(detail_points) >= 3:
                    pygame.draw.polygon(screen, "gray", detail_points, 1)

    def update(self, dt):
        self.position += self.velocity * dt
        self.rotation += self.rotation_speed * dt
        self.bounce_off_edges()
    
    def bounce_off_edges(self):
        """Make asteroid bounce off screen edges."""
        if (self.position.x - self.radius <= 0 and self.velocity.x < 0) or (self.position.x + self.radius >= SCREEN_WIDTH and self.velocity.x > 0): # type: ignore
            self.velocity.x = -self.velocity.x
            self.rotation_speed += random.uniform(-20, 20)
            if self.position.x - self.radius <= 0: # type: ignore
                self.position.x = self.radius # type: ignore
            else:
                self.position.x = SCREEN_WIDTH - self.radius # type: ignore

        if (self.position.y - self.radius <= 0 and self.velocity.y < 0) or (self.position.y + self.radius >= SCREEN_HEIGHT and self.velocity.y > 0): # type: ignore
            self.velocity.y = -self.velocity.y
            self.rotation_speed += random.uniform(-20, 20)
            if self.position.y - self.radius <= 0: # type: ignore
                self.position.y = self.radius # type: ignore
            else:
                self.position.y = SCREEN_HEIGHT - self.radius # type: ignore
    
    def split(self):
        """Destroy current asteroid and spawn two smaller fragments."""
        self.kill()
        if self.radius <= ASTEROID_MIN_RADIUS:
            return

        angle = random.uniform(20, 50)
        new_vel_1 = self.velocity.rotate(angle)
        new_vel_2 = self.velocity.rotate(-angle)

        new_radius = self.radius - ASTEROID_MIN_RADIUS
        new_asteroid_1 = Asteroid(self.position.x, self.position.y, new_radius) # type: ignore
        new_asteroid_1.velocity = new_vel_1 * 1.2
        new_asteroid_1.rotation_speed = self.rotation_speed * random.uniform(1.2, 2.0)

        new_asteroid_2 = Asteroid(self.position.x, self.position.y, new_radius) # type: ignore
        new_asteroid_2.velocity = new_vel_2 * 1.2
        new_asteroid_2.rotation_speed = self.rotation_speed * random.uniform(1.2, 2.0)

