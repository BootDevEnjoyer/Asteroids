"""Shot projectile with bouncing mechanics and visual feedback."""

from circleshape import *
from constants import *

class Shot(CircleShape):
    def __init__(self, x, y, velocity):
        super().__init__(x, y, SHOT_RADIUS)
        self.velocity = velocity
        self.bounce_count = 0
        self.max_bounces = 3
        self.speed_reduction = 0.9  # velocity multiplier per bounce

    def draw(self, screen):
        # color transitions based on bounce count
        if self.bounce_count == 0:
            color = "cyan"
        elif self.bounce_count == 1:
            color = "lightblue"
        elif self.bounce_count == 2:
            color = "lightgreen"
        else:
            color = "white"
        
        # render projectile with glow effect
        pygame.draw.circle(screen, color, self.position, self.radius)
        pygame.draw.circle(screen, color, self.position, self.radius + 1, 1)

    def update(self, dt):
        self.position += self.velocity * dt
        
        # screen boundary collision detection
        bounced = False
        
        # horizontal boundary collision
        if (self.position.x - self.radius <= 0 and self.velocity.x < 0) or \
           (self.position.x + self.radius >= SCREEN_WIDTH and self.velocity.x > 0):
            self.velocity.x = -self.velocity.x
            # clamp position within screen bounds
            if self.position.x - self.radius <= 0:
                self.position.x = self.radius
            else:
                self.position.x = SCREEN_WIDTH - self.radius
            bounced = True
            
        # vertical boundary collision
        if (self.position.y - self.radius <= 0 and self.velocity.y < 0) or \
           (self.position.y + self.radius >= SCREEN_HEIGHT and self.velocity.y > 0):
            self.velocity.y = -self.velocity.y
            # clamp position within screen bounds
            if self.position.y - self.radius <= 0:
                self.position.y = self.radius
            else:
                self.position.y = SCREEN_HEIGHT - self.radius
            bounced = True
        
        # process bounce mechanics
        if bounced:
            self.bounce_count += 1
            # apply velocity reduction
            self.velocity *= self.speed_reduction
            
            # destroy after exceeding bounce limit
            if self.bounce_count > self.max_bounces:
                self.kill()