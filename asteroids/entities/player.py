import random
import math
import pygame
from asteroids.core.circleshape import CircleShape
from asteroids.core.constants import *
from asteroids.entities.shot import Shot

class ThrustParticle:
    """Individual particle for engine thrust visual effect."""

    def __init__(self, x, y, velocity, life_time=0.5):
        self.position = pygame.Vector2(x, y)
        self.velocity = velocity
        self.max_life = life_time
        self.life = life_time
        self.size = random.uniform(1.5, 3.0)

    def update(self, dt):
        self.position += self.velocity * dt
        self.life -= dt
        self.velocity *= 0.98

    def draw(self, screen):
        if self.life <= 0:
            return

        alpha = max(0, self.life / self.max_life)
        size = self.size * alpha

        if alpha > 0.7:
            color = (255, 255, 255)
        elif alpha > 0.4:
            color = (255, 200, 100)
        else:
            color = (255, 150, 50)

        if size > 1:
            pygame.draw.circle(screen, color, self.position, int(size))

class Player(CircleShape):
    """Player ship with thrust particles and reduced collision radius."""

    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_RADIUS)
        self.rotation = 0
        self.collision_radius = PLAYER_RADIUS * 0.7
        self.shot_cooldown = 0.0
        self.max_energy = PLAYER_MAX_ENERGY
        self.energy = PLAYER_MAX_ENERGY
        self._regen_delay = PLAYER_ENERGY_REGEN_DELAY
        self._regen_timer = 0.0
        self.energy_starved = False

        self.thrust_particles = []
        self.is_thrusting = False

        self.killed_by_ai = False
        self.random_target = None
        self.bounce_velocity = None

    def get_ship_points(self):
        """Generate detailed ship geometry with multiple polygon sections."""
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90)

        nose = self.position + forward * self.radius
        rear_center = self.position - forward * self.radius * 0.8
        wing_left = self.position - forward * self.radius * 0.3 - right * self.radius * 0.6
        wing_right = self.position - forward * self.radius * 0.3 + right * self.radius * 0.6
        rear_left = self.position - forward * self.radius * 0.8 - right * self.radius * 0.3
        rear_right = self.position - forward * self.radius * 0.8 + right * self.radius * 0.3

        return {
            'main_body': [nose, wing_left, rear_left, rear_center, rear_right, wing_right],
            'cockpit': [
                self.position + forward * self.radius * 0.3,
                self.position + forward * self.radius * 0.1 - right * self.radius * 0.15,
                self.position - forward * self.radius * 0.1 - right * self.radius * 0.15,
                self.position - forward * self.radius * 0.1 + right * self.radius * 0.15,
                self.position + forward * self.radius * 0.1 + right * self.radius * 0.15,
            ],
            'engine_left': [rear_left, rear_left - forward * self.radius * 0.2, rear_center],
            'engine_right': [rear_right, rear_right - forward * self.radius * 0.2, rear_center]
        }

    def triangle(self):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90) * self.radius / 1.5
        a = self.position + forward * self.radius
        b = self.position - forward * self.radius - right
        c = self.position - forward * self.radius + right
        return [a, b, c]

    def spawn_thrust_particles(self, dt):
        """Generate thrust particles behind ship during acceleration."""
        if not self.is_thrusting:
            return

        particles_per_frame = 8

        for _ in range(particles_per_frame):
            rear_offset = pygame.Vector2(0, -1).rotate(self.rotation) * self.radius * 0.8
            spread = pygame.Vector2(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)).rotate(self.rotation) * self.radius
            particle_pos = self.position + rear_offset + spread

            base_velocity = pygame.Vector2(0, -1).rotate(self.rotation) * random.uniform(80, 150)
            velocity_spread = pygame.Vector2(random.uniform(-50, 50), random.uniform(-30, 30))
            particle_velocity = base_velocity + velocity_spread

            particle = ThrustParticle(
                particle_pos.x,
                particle_pos.y,
                particle_velocity,
                random.uniform(0.3, 0.7)
            )
            self.thrust_particles.append(particle)

    def update_thrust_particles(self, dt):
        """Update particle states and remove expired particles."""
        for particle in self.thrust_particles[:]:
            particle.update(dt)
            if particle.life <= 0:
                self.thrust_particles.remove(particle)

    def draw(self, screen):
        for particle in self.thrust_particles:
            particle.draw(screen)

        ship_parts = self.get_ship_points()

        pygame.draw.polygon(screen, "white", ship_parts['main_body'], 2)
        pygame.draw.polygon(screen, "cyan", ship_parts['cockpit'], 2)

        if self.is_thrusting:
            pygame.draw.polygon(screen, "yellow", ship_parts['engine_left'], 2)
            pygame.draw.polygon(screen, "yellow", ship_parts['engine_right'], 2)
        else:
            pygame.draw.polygon(screen, "gray", ship_parts['engine_left'], 2)
            pygame.draw.polygon(screen, "gray", ship_parts['engine_right'], 2)

    def check_collision(self, other):
        """Collision detection using reduced radius for improved gameplay."""
        distance = pygame.Vector2.distance_to(self.position, other.position)
        return distance <= self.collision_radius + other.radius

    def rotate(self, dt):
        self.rotation += PLAYER_TURN_SPEED * dt

    def move(self, dt):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        movement = forward * PLAYER_SPEED * dt
        self.position += movement
        self.velocity = forward * PLAYER_SPEED * (1 if dt > 0 else -1)

    def update(self, dt):
        keys = pygame.key.get_pressed()

        self.velocity = pygame.Vector2(0, 0)
        self.is_thrusting = False

        if keys[pygame.K_LEFT]:
            self.rotate(-dt)
        if keys[pygame.K_RIGHT]:
            self.rotate(dt)
        if keys[pygame.K_UP]:
            self.move(dt)
            self.is_thrusting = True
        if keys[pygame.K_DOWN]:
            self.move(-dt)

        if keys[pygame.K_SPACE]:
            self.shoot(dt)

        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            exit()

        self.spawn_thrust_particles(dt)
        self.update_thrust_particles(dt)
        self.wrap_position()

        self.shot_cooldown = max(0.0, self.shot_cooldown - dt)

        if self._regen_timer > 0.0:
            self._regen_timer = max(0.0, self._regen_timer - dt)
        else:
            self.energy = min(self.max_energy, self.energy + PLAYER_ENERGY_REGEN_RATE * dt)
        
        if self.energy > 0.0 and self.energy_starved:
            self.energy_starved = False

    def shoot(self, dt):
        if self.shot_cooldown > 0:
            return
        if self.energy < PLAYER_ENERGY_PER_SHOT:
            self.energy_starved = True
            return

        shot_velocity = pygame.Vector2(0, 1).rotate(self.rotation) * PLAYER_SHOOT_SPEED
        shot = Shot(self.position.x, self.position.y, shot_velocity) # type: ignore

        self.shot_cooldown = PLAYER_SHOOT_COOLDOWN
        self.energy = max(0.0, self.energy - PLAYER_ENERGY_PER_SHOT)
        self._regen_timer = self._regen_delay
        self.energy_starved = self.energy <= 0.0
