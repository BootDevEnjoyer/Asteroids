from circleshape import *
from constants import *
import random
import pygame
import math

class Asteroid(CircleShape):
    def __init__(self, x, y, radius):
        super().__init__(x, y, radius)
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Rotation properties
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-50, 50)  # degrees per second
        
        # Generate jagged shape
        self.shape_points = self.generate_jagged_shape()

    def generate_jagged_shape(self):
        """Generate irregular polygon points for asteroid shape"""
        points = []
        num_points = random.randint(8, 12)  # Number of vertices
        
        for i in range(num_points):
            # Angle for this point around the circle
            angle = (360 / num_points) * i
            
            # Add some randomness to the angle
            angle += random.uniform(-15, 15)
            
            # Vary the radius for jagged edges (70% to 130% of base radius)
            point_radius = self.radius * random.uniform(0.7, 1.3)
            
            # Calculate point position
            x = math.cos(math.radians(angle)) * point_radius
            y = math.sin(math.radians(angle)) * point_radius
            
            points.append(pygame.Vector2(x, y))
        
        return points

    def get_rotated_points(self):
        """Get the asteroid's shape points rotated by current rotation"""
        rotated_points = []
        
        for point in self.shape_points:
            # Rotate each point by the current rotation
            rotated_point = point.rotate(self.rotation)
            # Translate to asteroid's position
            world_point = self.position + rotated_point
            rotated_points.append(world_point)
            
        return rotated_points

    def draw(self, screen):
        # Get rotated shape points
        points = self.get_rotated_points()
        
        # Draw the jagged asteroid shape
        if len(points) >= 3:
            pygame.draw.polygon(screen, "white", points, 2)
            
            # Add some surface detail - draw smaller internal lines
            if len(points) >= 6:
                # Draw some internal detail lines for rocky texture
                detail_points = points[::2]  # Every other point
                if len(detail_points) >= 3:
                    pygame.draw.polygon(screen, "gray", detail_points, 1)

    def update(self, dt):
        # Update position
        self.position += self.velocity * dt
        
        # Update rotation
        self.rotation += self.rotation_speed * dt
        
        # Handle screen bouncing instead of wrapping
        self.bounce_off_edges()
    
    def bounce_off_edges(self):
        """Make asteroid bounce off screen edges"""
        # Bounce off left or right edges
        if (self.position.x - self.radius <= 0 and self.velocity.x < 0) or \
           (self.position.x + self.radius >= SCREEN_WIDTH and self.velocity.x > 0):
            self.velocity.x = -self.velocity.x
            # Add some rotation change when bouncing
            self.rotation_speed += random.uniform(-20, 20)
            # Keep asteroid within bounds
            if self.position.x - self.radius <= 0:
                self.position.x = self.radius
            else:
                self.position.x = SCREEN_WIDTH - self.radius
                
        # Bounce off top or bottom edges
        if (self.position.y - self.radius <= 0 and self.velocity.y < 0) or \
           (self.position.y + self.radius >= SCREEN_HEIGHT and self.velocity.y > 0):
            self.velocity.y = -self.velocity.y
            # Add some rotation change when bouncing
            self.rotation_speed += random.uniform(-20, 20)
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
        # Give split asteroids faster rotation
        new_asteroid_1.rotation_speed = self.rotation_speed * random.uniform(1.2, 2.0)
        
        new_asteroid_2 = Asteroid(self.position.x, self.position.y, new_radius) # type: ignore 
        new_asteroid_2.velocity = new_vel_2 * 1.2
        # Give split asteroids faster rotation
        new_asteroid_2.rotation_speed = self.rotation_speed * random.uniform(1.2, 2.0)

