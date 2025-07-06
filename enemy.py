import pygame
import math
import random
from circleshape import CircleShape
from constants import *
import torch

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
                # Fix: Clamp the lerp factor to [0, 1]
                lerp_factor = min(1.0, max(0.0, transition_factor * dt * 2))
                self.velocity = self.velocity.lerp(target_velocity, lerp_factor)
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


class NeuralEnemy(CircleShape):
    """AI-controlled enemy using ultra-simplified neural network learning"""
    def __init__(self, x, y):
        super().__init__(x, y, ENEMY_RADIUS)
        
        # Get the global AI brain
        from ai_brain import get_global_brain, GameStateCollector
        self.brain = get_global_brain()
        self.state_collector = GameStateCollector()
        
        # Enemy identification
        self.enemy_id = random.randint(100, 999)
        
        # Movement settings - much simpler!
        self.move_speed = ENEMY_SPEED * 1.2  # Constant speed
        self.current_angle = random.uniform(0, 2 * math.pi)  # Current movement direction
        self.target_angle = self.current_angle  # Target direction from AI
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_distance = 0
        
        # Visual effects
        self.pulse_timer = 0
        self.ai_glow_intensity = 0.5
        
        # Game references
        self.player_target = None
        self.asteroid_group = None
        self.enemy_group = None
        
        # Training phase tracking
        self.last_distance_to_player = 0
        
        # Add this line:
        self.active = True  # Disable AI when collision detected
        
        print(f"Neural AI Enemy #{self.enemy_id} spawned - Phase {self.brain.training_phase} learning")
    
    def set_target(self, player):
        """Set the player as the target to learn to follow"""
        self.player_target = player
        if player:
            self.last_distance_to_player = (self.position - player.position).length()
            self.episode_start_distance = self.last_distance_to_player
    
    def set_game_groups(self, asteroid_group, enemy_group):
        """Set references to game object groups for state collection"""
        self.asteroid_group = asteroid_group
        self.enemy_group = enemy_group
    
    def calculate_reward(self):
        """Reward function that encourages getting closer but doesn't declare success"""
        if not self.player_target:
            return 0.0
        
        current_distance = (self.position - self.player_target.position).length()
        reward = 0.0
        
        # Primary reward: getting closer to player
        distance_improvement = self.last_distance_to_player - current_distance
        reward += distance_improvement * 5.0  # Strong reward for progress
        
        # Calculate efficiency bonus - reward moving towards player efficiently
        if hasattr(self, 'velocity') and self.velocity.length() > 0:
            direction_to_player = (self.player_target.position - self.position)
            if direction_to_player.length() > 0:
                direction_to_player = direction_to_player.normalize()
                velocity_direction = self.velocity.normalize()
                
                # Dot product gives us alignment (-1 to 1)
                alignment = direction_to_player.dot(velocity_direction)
                reward += alignment * 2.0  # Bonus for moving toward player
                
                # Penalty for moving away from player
                if alignment < -0.5:
                    reward -= 3.0
        
        # Distance-based bonuses 
        if current_distance < 30:    
            reward += 30
        elif current_distance < 60:   # Close
            reward += 15.0
        elif current_distance < 100:  # Getting there
            reward += 8.0
        elif current_distance < 150:  # Making progress
            reward += 1.0
        
        # Penalty for being far
        if current_distance > 400:
            reward -= 2.0
        
        # Small time penalty to encourage faster completion
        reward -= 0.2
        
        # Rotation efficiency penalty - discourage excessive turning
        if hasattr(self, 'target_angle') and hasattr(self, 'current_angle'):
            angle_diff = abs(self.target_angle - self.current_angle)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)  # Use shortest rotation
            
            # Penalty for large angle differences (spinning)
            if angle_diff > math.pi / 2:  # > 90 degrees
                reward -= 1.0
            elif angle_diff > math.pi / 4:  # > 45 degrees
                reward -= 0.5
        
        self.last_distance_to_player = current_distance
        return reward
    
    def update(self, dt):
        """Ultra-simplified update with direction-only learning"""
        # Import main module at the beginning to access training mode
        import main
        
        self.pulse_timer += dt
        self.ai_glow_intensity = (math.sin(self.pulse_timer * 4) + 1) * 0.5
        
        # Add this check immediately:
        if not self.active:
            return  # Stop all AI processing
        
        if not self.player_target:
            return
        
        # Get simple state
        current_state = self.state_collector.collect_state(self, self.player_target)
        
        # Get target angle adjustment from neural network
        angle_adjustment = self.brain.get_action(current_state)
        
        # Convert angle adjustment to target direction
        direction_to_player = self.player_target.position - self.position
        if direction_to_player.length() > 0:
            angle_to_player = math.atan2(direction_to_player.y, direction_to_player.x)
            adjustment_radians = angle_adjustment.item() * math.pi * 0.5
            self.target_angle = angle_to_player + adjustment_radians
        else:
            self.target_angle = self.current_angle
        
        # Normalize angles to prevent accumulation
        self.current_angle = self.current_angle % (2 * math.pi)
        self.target_angle = self.target_angle % (2 * math.pi)
        
        # Smoothly turn toward target angle
        angle_diff = self.target_angle - self.current_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Turn at reasonable rate
        max_turn_rate = 3.0 * dt
        if abs(angle_diff) > max_turn_rate:
            self.current_angle += max_turn_rate * (1 if angle_diff > 0 else -1)
        else:
            self.current_angle = self.target_angle
        
        # Normalize current angle after adjustment
        self.current_angle = self.current_angle % (2 * math.pi)
        
        # Move at constant speed in current direction
        self.velocity = pygame.Vector2(
            math.cos(self.current_angle) * self.move_speed,
            math.sin(self.current_angle) * self.move_speed
        )
        
        # Update position
        self.position += self.velocity * dt
        
        # **MODIFIED**: Check for training mode before respawning
        # NO WRAPPING - respawn if off-screen (ONLY IN TRAINING MODE)
        if (self.position.x < -50 or self.position.x > SCREEN_WIDTH + 50 or
            self.position.y < -50 or self.position.y > SCREEN_HEIGHT + 50):
            
            # Check if we're in training mode by looking for training_mode global
            if hasattr(main, 'current_training_mode') and main.current_training_mode:
                self.respawn_near_player()
                return
            else:
                # In normal mode: use screen wrapping like other enemies
                self.wrap_position()
        
        # Calculate reward - no success detection here!
        reward = self.calculate_reward()
        self.episode_reward += reward
        self.episode_length += 1
        self.brain.store_reward(reward)
        
        # **MODIFIED**: Episode management (ONLY IN TRAINING MODE)
        current_distance = (self.position - self.player_target.position).length()
        episode_done = False
        
        # Check if we're in training mode for episode management
        if hasattr(main, 'current_training_mode') and main.current_training_mode:
            if self.episode_length >= 200:  # 3.3 seconds at 60fps
                episode_done = True
            
            elif current_distance > 600:
                # Too far away, respawn closer
                self.respawn_near_player()
                episode_done = True
            
            if episode_done:
                # End episode - never with success=True (only collision detection does that)
                self.brain.end_episode(self.episode_reward, success=False)
                
                # Check for phase advancement
                if self.brain.should_advance_phase():
                    self.brain.advance_phase()
                
                # Reset for new episode
                self.episode_reward = 0.0
                self.episode_length = 0
                self.respawn_near_player()
        
        # Debug output
        if self.brain.training_steps % 50 == 0 and self.episode_length == 1:
            print(f"🤖 AI #{self.enemy_id}: Phase {self.brain.training_phase}, Episode {self.brain.training_steps}")
            print(f"   Success: {self.brain.get_success_rate():.1%}, Distance: {current_distance:.0f}px")
            print(f"   Angle: {math.degrees(self.current_angle):.0f}°, Target: {math.degrees(self.target_angle):.0f}°")
    
    def respawn_near_player(self):
        """Respawn at random position near player (no wrapping!)"""
        if not self.player_target:
            return
        
        # Spawn 200-400 pixels away from player
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(200, 400)
        
        spawn_pos = self.player_target.position + pygame.Vector2(
            math.cos(angle) * distance,
            math.sin(angle) * distance
        )
        
        # Clamp to screen bounds with margin
        spawn_pos.x = max(50, min(SCREEN_WIDTH - 50, spawn_pos.x))
        spawn_pos.y = max(50, min(SCREEN_HEIGHT - 50, spawn_pos.y))
        
        self.position = spawn_pos
        self.last_distance_to_player = (self.position - self.player_target.position).length()
        self.episode_start_distance = self.last_distance_to_player
        
        # Reset movement
        self.current_angle = random.uniform(0, 2 * math.pi)
        self.target_angle = self.current_angle
    
    def draw(self, screen):
        """Draw AI enemy with target direction visualization"""
        pulse = math.sin(self.pulse_timer * 3) * 0.2 + 1.0
        current_radius = self.radius * pulse
        
        # Phase-specific colors
        phase_colors = {
            1: (100, 255, 100),   # Green for phase 1
            2: (255, 200, 100),   # Orange for phase 2  
            3: (255, 100, 100)    # Red for phase 3
        }
        main_color = phase_colors.get(self.brain.training_phase, (100, 200, 255))
        
        # AI glow effect
        glow_radius = int(current_radius * (1.3 + self.ai_glow_intensity * 0.3))
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = (*main_color, 25)
        pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)
        screen.blit(glow_surface, (self.position.x - glow_radius, self.position.y - glow_radius))
        
        # Main body
        pygame.draw.circle(screen, main_color, self.position, int(current_radius + 2), 2)
        pygame.draw.circle(screen, (20, 20, 60), self.position, int(current_radius))
        
        # Inner core
        core_radius = int(current_radius * 0.6)
        pygame.draw.circle(screen, main_color, self.position, core_radius)
        
        # TARGET DIRECTION ARROW - this shows where AI wants to go!
        arrow_length = current_radius * 2.5
        arrow_end = self.position + pygame.Vector2(
            math.cos(self.target_angle) * arrow_length,
            math.sin(self.target_angle) * arrow_length
        )
        
        # Draw target direction line
        pygame.draw.line(screen, (255, 255, 100), self.position, arrow_end, 3)
        
        # Draw arrowhead
        arrow_size = 8
        arrowhead_angle1 = self.target_angle + 2.5
        arrowhead_angle2 = self.target_angle - 2.5
        
        arrowhead1 = arrow_end + pygame.Vector2(
            math.cos(arrowhead_angle1) * arrow_size,
            math.sin(arrowhead_angle1) * arrow_size
        )
        arrowhead2 = arrow_end + pygame.Vector2(
            math.cos(arrowhead_angle2) * arrow_size,
            math.sin(arrowhead_angle2) * arrow_size
        )
        
        pygame.draw.polygon(screen, (255, 255, 100), [arrow_end, arrowhead1, arrowhead2])
        
        # Current movement direction (smaller, different color)
        current_end = self.position + pygame.Vector2(
            math.cos(self.current_angle) * (arrow_length * 0.7),
            math.sin(self.current_angle) * (arrow_length * 0.7)
        )
        pygame.draw.line(screen, (150, 150, 255), self.position, current_end, 2)
        
        # AI label with phase info
        font = pygame.font.Font(None, 16)
        ai_text = font.render(f"AI-P{self.brain.training_phase}", True, (255, 255, 255))
        text_rect = ai_text.get_rect()
        text_rect.center = (int(self.position.x), int(self.position.y - current_radius - 20))
        screen.blit(ai_text, text_rect)
    
    def kill(self):
        """Called when enemy is destroyed - end episode"""
        final_penalty = -20.0
        self.brain.store_reward(final_penalty)
        self.brain.end_episode(self.episode_reward + final_penalty, success=False)
        super().kill()

class EnemySpawner:
    """Manages enemy spawning and lifecycle"""
    def __init__(self, enemy_group):
        self.spawn_timer = 0
        self.enemy_group = enemy_group
        
    def update(self, dt, player, asteroid_group=None):
        self.spawn_timer += dt
        
        # Spawn new enemy if conditions are met
        if (self.spawn_timer >= ENEMY_SPAWN_RATE and 
            len(self.enemy_group) < ENEMY_MAX_COUNT):
            self.spawn_enemy(player, asteroid_group)
            self.spawn_timer = 0
    
    def spawn_enemy(self, player, asteroid_group=None):
        """Spawn a new enemy at a random edge of the screen"""
        # Choose enemy type: 40% neural, 30% follower, 30% shooter
        # enemy_type = random.choices(["neural", "follower", "shooter"], weights=[40, 30, 30])[0]
        enemy_type = "neural"
        
        # Choose random edge: 0=top, 1=right, 2=bottom, 3=left
        edge = random.randint(0, 3)
        
        if enemy_type == "neural":
            # Neural enemy - spawn closer to center and give initial velocity toward center
            center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
            
            if edge == 0:  # Top
                x = random.uniform(SCREEN_WIDTH * 0.2, SCREEN_WIDTH * 0.8)
                y = -ENEMY_RADIUS
                initial_velocity = pygame.Vector2(0, 1)  # Move down
            elif edge == 1:  # Right
                x = SCREEN_WIDTH + ENEMY_RADIUS
                y = random.uniform(SCREEN_HEIGHT * 0.2, SCREEN_HEIGHT * 0.8)
                initial_velocity = pygame.Vector2(-1, 0)  # Move left
            elif edge == 2:  # Bottom
                x = random.uniform(SCREEN_WIDTH * 0.2, SCREEN_WIDTH * 0.8)
                y = SCREEN_HEIGHT + ENEMY_RADIUS
                initial_velocity = pygame.Vector2(0, -1)  # Move up
            else:  # Left
                x = -ENEMY_RADIUS
                y = random.uniform(SCREEN_HEIGHT * 0.2, SCREEN_HEIGHT * 0.8)
                initial_velocity = pygame.Vector2(1, 0)  # Move right
            
            enemy = NeuralEnemy(x, y)
            enemy.set_target(player)
            
            # Give initial velocity toward center
            spawn_pos = pygame.Vector2(x, y)
            center_pos = pygame.Vector2(center_x, center_y)
            direction_to_center = (center_pos - spawn_pos).normalize()
            enemy.velocity = direction_to_center * ENEMY_SPEED * 0.5  # Start moving toward center
            
            # Set game group references for state collection
            if asteroid_group:
                enemy.set_game_groups(asteroid_group, self.enemy_group)
            print(f"Neural enemy spawned at ({x:.1f}, {y:.1f}) moving toward center")
            
        elif enemy_type == "shooter":
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