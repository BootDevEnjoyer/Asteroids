from circleshape import *
from constants import *
from shot import *
import random
import math

class ThrustParticle:
    """Individual particle for engine thrust effect"""
    def __init__(self, x, y, velocity, life_time=0.5):
        self.position = pygame.Vector2(x, y)
        self.velocity = velocity
        self.max_life = life_time
        self.life = life_time
        self.size = random.uniform(1.5, 3.0)
        
    def update(self, dt):
        self.position += self.velocity * dt
        self.life -= dt
        # Slow down particles over time
        self.velocity *= 0.98
        
    def draw(self, screen):
        if self.life <= 0:
            return
        
        # Fade particles as they age
        alpha = max(0, self.life / self.max_life)
        size = self.size * alpha
        
        # Color gradient from white to orange to red
        if alpha > 0.7:
            color = (255, 255, 255)  # White
        elif alpha > 0.4:
            color = (255, 200, 100)  # Orange
        else:
            color = (255, 150, 50)   # Red-orange
            
        if size > 1:
            pygame.draw.circle(screen, color, self.position, int(size))

class Player(CircleShape):
    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_RADIUS)
        self.rotation = 0
        # Collision radius is smaller than visual radius for tighter collision detection
        self.collision_radius = PLAYER_RADIUS * 0.7  # 70% of visual radius
        self.shot_cooldown = 0.0
        
        # Thrust particle system
        self.thrust_particles = []
        self.is_thrusting = False
        
        # Training mode attributes
        self.killed_by_ai = False  # Track if player was killed by AI
        self.random_target = None  # For random movement patterns
        self.bounce_velocity = None  # For bouncing movement patterns

    def get_ship_points(self):
        """Generate detailed ship design with multiple polygons"""
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90)
        
        # Main ship body (elongated triangle)
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
        """Create thrust particles behind the ship when thrusting"""
        if not self.is_thrusting:
            return
            
        # Spawn multiple particles per frame
        particles_per_frame = 8
        
        for _ in range(particles_per_frame):
            # Position particles at the rear of the ship with some spread
            rear_offset = pygame.Vector2(0, -1).rotate(self.rotation) * self.radius * 0.8
            spread = pygame.Vector2(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)).rotate(self.rotation) * self.radius
            
            particle_pos = self.position + rear_offset + spread
            
            # Particle velocity should be opposite to ship's forward direction
            base_velocity = pygame.Vector2(0, -1).rotate(self.rotation) * random.uniform(80, 150)
            # Add some randomness
            velocity_spread = pygame.Vector2(random.uniform(-50, 50), random.uniform(-30, 30))
            particle_velocity = base_velocity + velocity_spread
            
            # Create particle with random lifetime
            particle = ThrustParticle(
                particle_pos.x, 
                particle_pos.y, 
                particle_velocity,
                random.uniform(0.3, 0.7)
            )
            self.thrust_particles.append(particle)

    def update_thrust_particles(self, dt):
        """Update and clean up thrust particles"""
        # Update all particles
        for particle in self.thrust_particles[:]:
            particle.update(dt)
            if particle.life <= 0:
                self.thrust_particles.remove(particle)

    def draw(self, screen):
        # Draw thrust particles first (behind ship)
        for particle in self.thrust_particles:
            particle.draw(screen)
        
        # Draw detailed ship
        ship_parts = self.get_ship_points()
        
        # Main body
        pygame.draw.polygon(screen, "white", ship_parts['main_body'], 2)
        
        # Cockpit
        pygame.draw.polygon(screen, "cyan", ship_parts['cockpit'], 2)
        
        # Engine sections
        if self.is_thrusting:
            # Draw engines with thrust glow
            pygame.draw.polygon(screen, "yellow", ship_parts['engine_left'], 2)
            pygame.draw.polygon(screen, "yellow", ship_parts['engine_right'], 2)
        else:
            pygame.draw.polygon(screen, "gray", ship_parts['engine_left'], 2)
            pygame.draw.polygon(screen, "gray", ship_parts['engine_right'], 2)

    def check_collision(self, other):
        """Override collision detection to use smaller collision radius for player"""
        distance = pygame.Vector2.distance_to(self.position, other.position)
        return distance <= self.collision_radius + other.radius

    def rotate(self, dt):
        self.rotation += PLAYER_TURN_SPEED * dt

    def move(self, dt):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        # dt can be negative for backward movement
        movement = forward * PLAYER_SPEED * dt
        self.position += movement
        # Track velocity for parallax effects (use sign of dt for direction)
        self.velocity = forward * PLAYER_SPEED * (1 if dt > 0 else -1)

    def update(self, dt):
        keys = pygame.key.get_pressed()
        
        # Reset velocity and thrust state each frame
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

        # Update particle systems
        self.spawn_thrust_particles(dt)
        self.update_thrust_particles(dt)

        # Handle screen wrapping
        self.wrap_position()

        self.shot_cooldown -= dt
        if self.shot_cooldown < 0:
            self.shot_cooldown = 0.0

    def shoot(self, dt):
        if self.shot_cooldown > 0:
            return

        shot_velocity = pygame.Vector2(0, 1).rotate(self.rotation) * PLAYER_SHOOT_SPEED
        shot = Shot(self.position.x, self.position.y, shot_velocity) # type: ignore

        self.shot_cooldown = PLAYER_SHOOT_COOLDOWN
