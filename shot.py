from circleshape import *
from constants import *

class Shot(CircleShape):
    def __init__(self, x, y, velocity):
        super().__init__(x, y, SHOT_RADIUS)
        self.velocity = velocity
        self.bounce_count = 0
        self.max_bounces = 3
        self.speed_reduction = 0.9  # 10% speed reduction per bounce

    def draw(self, screen):
        # Change bullet color based on bounces for visual feedback (friendly colors)
        if self.bounce_count == 0:
            color = "cyan"          # Bright cyan - clearly friendly
        elif self.bounce_count == 1:
            color = "lightblue"     # Light blue - still friendly
        elif self.bounce_count == 2:
            color = "lightgreen"    # Light green - getting weaker
        else:
            color = "white"         # White - final bounce
        
        # Draw main bullet
        pygame.draw.circle(screen, color, self.position, self.radius)
        # Add a subtle glow effect for friendly shots
        pygame.draw.circle(screen, color, self.position, self.radius + 1, 1)

    def update(self, dt):
        self.position += self.velocity * dt
        
        # Check for bouncing off screen edges
        bounced = False
        
        # Bounce off left or right edges
        if (self.position.x - self.radius <= 0 and self.velocity.x < 0) or \
           (self.position.x + self.radius >= SCREEN_WIDTH and self.velocity.x > 0):
            self.velocity.x = -self.velocity.x
            # Keep bullet within bounds
            if self.position.x - self.radius <= 0:
                self.position.x = self.radius
            else:
                self.position.x = SCREEN_WIDTH - self.radius
            bounced = True
            
        # Bounce off top or bottom edges
        if (self.position.y - self.radius <= 0 and self.velocity.y < 0) or \
           (self.position.y + self.radius >= SCREEN_HEIGHT and self.velocity.y > 0):
            self.velocity.y = -self.velocity.y
            # Keep bullet within bounds
            if self.position.y - self.radius <= 0:
                self.position.y = self.radius
            else:
                self.position.y = SCREEN_HEIGHT - self.radius
            bounced = True
        
        # Handle bounce effects
        if bounced:
            self.bounce_count += 1
            # Reduce speed with each bounce
            self.velocity *= self.speed_reduction
            
            # Remove bullet after max bounces
            if self.bounce_count > self.max_bounces:
                self.kill()