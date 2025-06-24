from circleshape import *
from constants import *
from shot import *

class Player(CircleShape):
    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_RADIUS)
        self.rotation = 0
        # Collision radius is smaller than visual radius for tighter collision detection
        self.collision_radius = PLAYER_RADIUS * 0.7  # 70% of visual radius
        self.shot_cooldown = 0.0

    def triangle(self):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90) * self.radius / 1.5
        a = self.position + forward * self.radius
        b = self.position - forward * self.radius - right
        c = self.position - forward * self.radius + right
        return [a, b, c]

    def draw(self, screen):
        pygame.draw.polygon(screen, "white", self.triangle(), 2) # type: ignore

    def check_collision(self, other):
        """Override collision detection to use smaller collision radius for player"""
        distance = pygame.Vector2.distance_to(self.position, other.position)
        return distance <= self.collision_radius + other.radius

    def rotate(self, dt):
        self.rotation += PLAYER_TURN_SPEED * dt

    def move(self, dt):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        self.position += forward * PLAYER_SPEED * dt

    def update(self, dt):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            self.rotate(-dt)
        if keys[pygame.K_RIGHT]:
            self.rotate(dt)
        if keys[pygame.K_UP]:
            self.move(dt)
        if keys[pygame.K_DOWN]:
            self.move(-dt)

        if keys[pygame.K_SPACE]:
            self.shoot(dt)

        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            exit()

        self.shot_cooldown -= dt
        if self.shot_cooldown < 0:
            self.shot_cooldown = 0.0

    def shoot(self, dt):
        if self.shot_cooldown > 0:
            return

        shot_velocity = pygame.Vector2(0, 1).rotate(self.rotation) * PLAYER_SHOOT_SPEED
        shot = Shot(self.position.x, self.position.y, shot_velocity) # type: ignore

        self.shot_cooldown = PLAYER_SHOOT_COOLDOWN
