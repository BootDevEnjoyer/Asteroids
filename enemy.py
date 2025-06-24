import pygame
import math
import random
from circleshape import CircleShape
from constants import *

class EnemyShot(CircleShape):
    def __init__(self, x, y, direction):
        super().__init__(x, y, SHOOTER_SHOT_RADIUS)
        self.velocity = direction * SHOOTER_SHOT_SPEED
        
    def draw(self, screen):
        # Draw enemy shot as red bullet
        pygame.draw.circle(screen, "red", self.position, self.radius)
        # Add slight glow effect
        pygame.draw.circle(screen, (255, 100, 100), self.position, self.radius + 2, 1)
        
    def update(self, dt):
        self.position += self.velocity * dt
        
        # Kill shot if it goes off screen (no wrapping for enemy shots)
        if (self.position.x < -50 or self.position.x > SCREEN_WIDTH + 50 or
            self.position.y < -50 or self.position.y > SCREEN_HEIGHT + 50):
            self.kill()

class Enemy(CircleShape):
    def __init__(self, x, y):
        super().__init__(x, y, ENEMY_RADIUS)
        self.player_target = None
        self.rotation = 0
        self.rotation_speed = random.uniform(-30, 30)  # Slow rotation for menacing effect
        
        # Visual effects
        self.pulse_timer = 0
        self.trail_points = []  # For leaving a trail
        
    def set_target(self, player):
        """Set the player as the target to follow"""
        self.player_target = player
        
    def update(self, dt):
        # Update rotation for visual effect
        self.rotation += self.rotation_speed * dt
        self.pulse_timer += dt
        
        # Follow the player if we have a target
        if self.player_target:
            # Calculate direction to player
            direction_to_player = self.player_target.position - self.position
            distance_to_player = direction_to_player.length()
            
            # Only start following when close enough, otherwise maintain initial velocity
            if distance_to_player > 0 and distance_to_player <= ENEMY_DETECTION_RANGE:
                # Normalize direction and apply faster following speed
                direction_to_player = direction_to_player.normalize()
                target_velocity = direction_to_player * ENEMY_FOLLOW_SPEED
                
                # Gradually transition from initial velocity to following velocity
                transition_factor = min(1.0, (ENEMY_DETECTION_RANGE - distance_to_player) / (ENEMY_DETECTION_RANGE * 0.5))
                self.velocity = self.velocity.lerp(target_velocity, transition_factor * dt * 2)
            # If player is too far, maintain current velocity (initial spawn velocity or previous movement)
        
        # Store previous position before wrapping
        prev_position = pygame.Vector2(self.position)
        
        # Update position
        self.position += self.velocity * dt
        
        # Handle screen wrapping
        self.wrap_position()
        
        # Check if wrapping occurred (position jumped significantly)
        position_jump = (self.position - prev_position).length()
        if position_jump > 100:  # If position jumped more than 100 pixels, wrapping occurred
            self.trail_points.clear()  # Clear trail to prevent lines across screen
        
        # Add trail point (only if we didn't just wrap)
        if len(self.trail_points) == 0 or (self.position - self.trail_points[-1]).length() > 5:
            self.trail_points.append(pygame.Vector2(self.position))
            # Limit trail length
            if len(self.trail_points) > 8:
                self.trail_points.pop(0)
    
    def draw(self, screen):
        # Draw trail first (behind enemy)
        if len(self.trail_points) > 1:
            for i in range(len(self.trail_points) - 1):
                current_point = self.trail_points[i]
                next_point = self.trail_points[i + 1]
                
                # Skip drawing if points are too far apart (would create line across screen)
                distance = (next_point - current_point).length()
                if distance > 50:  # Skip if points are more than 50 pixels apart
                    continue
                
                alpha = (i + 1) / len(self.trail_points)
                trail_color = (int(255 * alpha * 0.3), 0, int(255 * alpha * 0.3))  # Purple trail
                
                try:
                    pygame.draw.line(screen, trail_color, current_point, next_point, 2)
                except:
                    pass  # Skip if points are invalid
        
        # Draw main enemy body with pulsing effect
        pulse = math.sin(self.pulse_timer * 4) * 0.3 + 1.0  # Pulse between 0.7 and 1.3
        current_radius = self.radius * pulse
        
        # Draw outer ring (red - danger!)
        pygame.draw.circle(screen, "red", self.position, int(current_radius + 3), 2)
        
        # Draw main body (dark red)
        pygame.draw.circle(screen, (150, 0, 0), self.position, int(current_radius), 2)
        
        # Draw inner core (bright red, pulsing)
        core_radius = int(current_radius * 0.6)
        pygame.draw.circle(screen, "red", self.position, core_radius)
        
        # Draw spikes/tentacles for menacing look
        num_spikes = 6
        for i in range(num_spikes):
            angle = (360 / num_spikes) * i + self.rotation
            spike_length = current_radius * 1.5
            
            start_pos = self.position + pygame.Vector2(
                math.cos(math.radians(angle)) * current_radius,
                math.sin(math.radians(angle)) * current_radius
            )
            end_pos = self.position + pygame.Vector2(
                math.cos(math.radians(angle)) * spike_length,
                math.sin(math.radians(angle)) * spike_length
            )
            
            pygame.draw.line(screen, "red", start_pos, end_pos, 2)
        
        # Draw warning glow when close to player
        if self.player_target:
            distance_to_player = (self.position - self.player_target.position).length()
            if distance_to_player < ENEMY_RADIUS * 3:
                # Draw warning glow
                glow_radius = int(current_radius * 2)
                glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (255, 0, 0, 30), (glow_radius, glow_radius), glow_radius)
                screen.blit(glow_surface, (self.position.x - glow_radius, self.position.y - glow_radius))


class ShooterEnemy(CircleShape):
    def __init__(self, x, y):
        super().__init__(x, y, SHOOTER_ENEMY_RADIUS)
        self.player_target = None
        self.rotation = 0
        self.rotation_speed = random.uniform(-45, 45)  # Faster rotation for distinction
        self.shoot_cooldown = 0
        
        # Visual effects
        self.pulse_timer = 0
        
        # Give it asteroid-like movement (constant velocity)
        angle = random.uniform(0, 360)
        speed = random.uniform(SHOOTER_ENEMY_SPEED * 0.8, SHOOTER_ENEMY_SPEED * 1.2)
        self.velocity = pygame.Vector2(
            math.cos(math.radians(angle)) * speed,
            math.sin(math.radians(angle)) * speed
        )
        
    def set_target(self, player):
        """Set the player as the target to shoot at"""
        self.player_target = player
        
    def update(self, dt):
        # Update rotation and timers
        self.rotation += self.rotation_speed * dt
        self.pulse_timer += dt
        self.shoot_cooldown -= dt
        
        # Move with constant velocity (like asteroids)
        self.position += self.velocity * dt
        
        # Shoot at player if in range and cooldown is ready
        if (self.player_target and self.shoot_cooldown <= 0):
            distance_to_player = (self.player_target.position - self.position).length()
            if distance_to_player <= SHOOTER_SHOOT_RANGE:
                self.shoot_at_player()
                self.shoot_cooldown = SHOOTER_SHOOT_COOLDOWN
        
        # Handle screen wrapping (like asteroids)
        self.wrap_position()
    
    def shoot_at_player(self):
        """Shoot a projectile toward the player"""
        if self.player_target:
            # Calculate direction to player
            direction = self.player_target.position - self.position
            if direction.length() > 0:
                direction = direction.normalize()
                
                # Create enemy shot (will be added to containers automatically)
                shot = EnemyShot(self.position.x, self.position.y, direction)
    
    def draw(self, screen):
        # Draw main enemy body with different visual style than regular enemy
        pulse = math.sin(self.pulse_timer * 3) * 0.2 + 1.0  # Different pulse rate
        current_radius = self.radius * pulse
        
        # Draw outer ring (orange - different from red enemies)
        pygame.draw.circle(screen, "orange", self.position, int(current_radius + 4), 3)
        
        # Draw main body (dark orange)
        pygame.draw.circle(screen, (200, 100, 0), self.position, int(current_radius), 2)
        
        # Draw inner core (bright orange, pulsing)
        core_radius = int(current_radius * 0.5)
        pygame.draw.circle(screen, "orange", self.position, core_radius)
        
        # Draw cross-hairs/targeting system instead of spikes
        cross_length = current_radius * 1.8
        for angle in [0, 90, 180, 270]:
            angle_rad = math.radians(angle + self.rotation)
            start_pos = self.position + pygame.Vector2(
                math.cos(angle_rad) * current_radius * 0.7,
                math.sin(angle_rad) * current_radius * 0.7
            )
            end_pos = self.position + pygame.Vector2(
                math.cos(angle_rad) * cross_length,
                math.sin(angle_rad) * cross_length
            )
            pygame.draw.line(screen, "orange", start_pos, end_pos, 2)
        
        # Draw shooting indicator when close to player
        if self.player_target:
            distance_to_player = (self.position - self.player_target.position).length()
            if distance_to_player <= SHOOTER_SHOOT_RANGE:
                # Draw targeting line toward player
                direction = (self.player_target.position - self.position).normalize()
                target_end = self.position + direction * min(distance_to_player, 100)
                pygame.draw.line(screen, (255, 200, 0), self.position, target_end, 1)


class EnemySpawner:
    """Manages enemy spawning and lifecycle"""
    def __init__(self, enemy_group):
        self.spawn_timer = 0
        self.enemy_group = enemy_group
        
    def update(self, dt, player):
        self.spawn_timer += dt
        
        # Spawn new enemy if conditions are met
        if (self.spawn_timer >= ENEMY_SPAWN_RATE and 
            len(self.enemy_group) < ENEMY_MAX_COUNT):
            self.spawn_enemy(player)
            self.spawn_timer = 0
    
    def spawn_enemy(self, player):
        """Spawn a new enemy at a random edge of the screen"""
        # Choose enemy type: 70% chance for follower, 30% chance for shooter
        enemy_type = random.choices(["follower", "shooter"], weights=[70, 30])[0]
        
        # Choose random edge: 0=top, 1=right, 2=bottom, 3=left
        edge = random.randint(0, 3)
        
        if enemy_type == "shooter":
            # Shooter enemies spawn anywhere on edge and move like asteroids
            if edge == 0:  # Top
                x = random.uniform(0, SCREEN_WIDTH)
                y = -SHOOTER_ENEMY_RADIUS
            elif edge == 1:  # Right
                x = SCREEN_WIDTH + SHOOTER_ENEMY_RADIUS
                y = random.uniform(0, SCREEN_HEIGHT)
            elif edge == 2:  # Bottom
                x = random.uniform(0, SCREEN_WIDTH)
                y = SCREEN_HEIGHT + SHOOTER_ENEMY_RADIUS
            else:  # Left
                x = -SHOOTER_ENEMY_RADIUS
                y = random.uniform(0, SCREEN_HEIGHT)
            
            enemy = ShooterEnemy(x, y)
            enemy.set_target(player)
            
        else:  # Follower enemy (original type)
            if edge == 0:  # Top
                x = random.uniform(0, SCREEN_WIDTH)
                y = -ENEMY_RADIUS
                # Initial velocity toward center-bottom
                initial_velocity = pygame.Vector2(0, 1)
            elif edge == 1:  # Right
                x = SCREEN_WIDTH + ENEMY_RADIUS
                y = random.uniform(0, SCREEN_HEIGHT)
                # Initial velocity toward center-left
                initial_velocity = pygame.Vector2(-1, 0)
            elif edge == 2:  # Bottom
                x = random.uniform(0, SCREEN_WIDTH)
                y = SCREEN_HEIGHT + ENEMY_RADIUS
                # Initial velocity toward center-top
                initial_velocity = pygame.Vector2(0, -1)
            else:  # Left
                x = -ENEMY_RADIUS
                y = random.uniform(0, SCREEN_HEIGHT)
                # Initial velocity toward center-right
                initial_velocity = pygame.Vector2(1, 0)
            
            enemy = Enemy(x, y)
            enemy.set_target(player)
            
            # Give enemy initial velocity toward center (with some randomness like asteroids)
            speed = random.uniform(ENEMY_SPEED * 0.8, ENEMY_SPEED * 1.2)
            enemy.velocity = initial_velocity * speed
            # Add some angle variation like asteroids do
            enemy.velocity = enemy.velocity.rotate(random.uniform(-20, 20))
        
        return enemy 