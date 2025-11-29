"""Enemy classes for asteroid game with various AI behaviors."""

import pygame
import math
import random
from asteroids.core.circleshape import CircleShape
from asteroids.core.constants import *
from asteroids.ai.debug_rl import debug_step, debug_episode_end, debug_print

class EnemyShot(CircleShape):
    """Enemy projectile with glow effect."""

    def __init__(self, x, y, direction):
        super().__init__(x, y, SHOOTER_SHOT_RADIUS)
        self.velocity = direction * SHOOTER_SHOT_SPEED

    def draw(self, screen):
        pygame.draw.circle(screen, "red", self.position, self.radius)
        pygame.draw.circle(screen, (255, 100, 100), self.position, self.radius + 2, 1)

    def update(self, dt):
        self.position += self.velocity * dt
        if (self.position.x < -50 or self.position.x > SCREEN_WIDTH + 50 or
            self.position.y < -50 or self.position.y > SCREEN_HEIGHT + 50):
            self.kill()

class Enemy(CircleShape):
    """Basic enemy that follows the player within detection range."""

    def __init__(self, x, y):
        super().__init__(x, y, ENEMY_RADIUS)
        self.player_target = None
        self.rotation = 0
        self.rotation_speed = random.uniform(-30, 30)
        self.pulse_timer = 0
        self.trail_points = []

    def set_target(self, player):
        self.player_target = player

    def update(self, dt):
        self.rotation += self.rotation_speed * dt
        self.pulse_timer += dt

        if self.player_target:
            direction_to_player = self.player_target.position - self.position
            distance_to_player = direction_to_player.length()

            if distance_to_player > 0 and distance_to_player <= ENEMY_DETECTION_RANGE:
                direction_to_player = direction_to_player.normalize()
                target_velocity = direction_to_player * ENEMY_FOLLOW_SPEED
                transition_factor = min(1.0, (ENEMY_DETECTION_RANGE - distance_to_player) / (ENEMY_DETECTION_RANGE * 0.5))
                lerp_factor = min(1.0, max(0.0, transition_factor * dt * 2))
                self.velocity = self.velocity.lerp(target_velocity, lerp_factor)

        prev_position = pygame.Vector2(self.position)
        self.position += self.velocity * dt
        self.wrap_position()

        # Clear trail on screen wrap to avoid visual artifacts
        position_jump = (self.position - prev_position).length()
        if position_jump > 100:
            self.trail_points.clear()

        if len(self.trail_points) == 0 or (self.position - self.trail_points[-1]).length() > 5:
            self.trail_points.append(pygame.Vector2(self.position))
            if len(self.trail_points) > 8:
                self.trail_points.pop(0)
    
    def draw(self, screen):
        if len(self.trail_points) > 1:
            for i in range(len(self.trail_points) - 1):
                current_point = self.trail_points[i]
                next_point = self.trail_points[i + 1]
                distance = (next_point - current_point).length()
                if distance > 50:
                    continue
                alpha = (i + 1) / len(self.trail_points)
                trail_color = (int(255 * alpha * 0.3), 0, int(255 * alpha * 0.3))
                try:
                    pygame.draw.line(screen, trail_color, current_point, next_point, 2)
                except (TypeError, ValueError):
                    pass

        pulse = math.sin(self.pulse_timer * 4) * 0.3 + 1.0
        current_radius = self.radius * pulse

        pygame.draw.circle(screen, "red", self.position, int(current_radius + 3), 2)
        pygame.draw.circle(screen, (150, 0, 0), self.position, int(current_radius), 2)

        core_radius = int(current_radius * 0.6)
        pygame.draw.circle(screen, "red", self.position, core_radius)

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

        if self.player_target:
            distance_to_player = (self.position - self.player_target.position).length()
            if distance_to_player < ENEMY_RADIUS * 3:
                glow_radius = int(current_radius * 2)
                glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (255, 0, 0, 30), (glow_radius, glow_radius), glow_radius)
                screen.blit(glow_surface, (self.position.x - glow_radius, self.position.y - glow_radius))   # type: ignore


class ShooterEnemy(CircleShape):
    """Enemy that drifts like an asteroid and shoots at the player."""

    def __init__(self, x, y):
        super().__init__(x, y, SHOOTER_ENEMY_RADIUS)
        self.player_target = None
        self.rotation = 0
        self.rotation_speed = random.uniform(-45, 45)
        self.shoot_cooldown = 0
        self.pulse_timer = 0

        angle = random.uniform(0, 360)
        speed = random.uniform(SHOOTER_ENEMY_SPEED * 0.8, SHOOTER_ENEMY_SPEED * 1.2)
        self.velocity = pygame.Vector2(
            math.cos(math.radians(angle)) * speed,
            math.sin(math.radians(angle)) * speed
        )

    def set_target(self, player):
        self.player_target = player

    def update(self, dt):
        self.rotation += self.rotation_speed * dt
        self.pulse_timer += dt
        self.shoot_cooldown -= dt
        self.position += self.velocity * dt

        if self.player_target and self.shoot_cooldown <= 0:
            distance_to_player = (self.player_target.position - self.position).length()
            if distance_to_player <= SHOOTER_SHOOT_RANGE:
                self.shoot_at_player()
                self.shoot_cooldown = SHOOTER_SHOOT_COOLDOWN

        self.wrap_position()

    def shoot_at_player(self):
        if self.player_target:
            direction = self.player_target.position - self.position
            if direction.length() > 0:
                direction = direction.normalize()
                shot = EnemyShot(self.position.x, self.position.y, direction) # type: ignore

    def draw(self, screen):
        pulse = math.sin(self.pulse_timer * 3) * 0.2 + 1.0
        current_radius = self.radius * pulse

        pygame.draw.circle(screen, "orange", self.position, int(current_radius + 4), 3)
        pygame.draw.circle(screen, (200, 100, 0), self.position, int(current_radius), 2)

        core_radius = int(current_radius * 0.5)
        pygame.draw.circle(screen, "orange", self.position, core_radius)

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

        if self.player_target:
            distance_to_player = (self.position - self.player_target.position).length()
            if distance_to_player <= SHOOTER_SHOOT_RANGE:
                direction = (self.player_target.position - self.position).normalize()
                target_end = self.position + direction * min(distance_to_player, 100)
                pygame.draw.line(screen, (255, 200, 0), self.position, target_end, 1)


class NeuralEnemy(CircleShape):
    """AI-controlled enemy using neural network learning."""

    def __init__(self, x, y, training_mode: bool = False):
        super().__init__(x, y, ENEMY_RADIUS)
        self.training_mode = training_mode

        from asteroids.ai.brain import get_global_brain, GameStateCollector
        self.brain = get_global_brain()
        self.state_collector = GameStateCollector()

        self.enemy_id = random.randint(100, 999)
        self.move_speed = ENEMY_SPEED * 1.2
        self.current_angle = random.uniform(0, 2 * math.pi)
        self.target_angle = self.current_angle

        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_distance = 0

        self.pulse_timer = 0
        self.ai_glow_intensity = 0.5

        self.player_target = None
        self.asteroid_group = None
        self.enemy_group = None

        self.last_distance_to_player = 0
        self.active = True
        self.episode_ended = False
        self.initial_approach = True  # Skip distance check during approach from spawn

        print(f"Neural AI Enemy #{self.enemy_id} spawned - Phase {self.brain.training_phase} learning")
    
    def set_target(self, player):
        self.player_target = player
        if player:
            self.last_distance_to_player = (self.position - player.position).length()
            self.episode_start_distance = self.last_distance_to_player
            direction_to_player = player.position - self.position
            if direction_to_player.length() > 0:
                self.current_angle = math.atan2(direction_to_player.y, direction_to_player.x)
                self.target_angle = self.current_angle

    def set_game_groups(self, asteroid_group, enemy_group):
        self.asteroid_group = asteroid_group
        self.enemy_group = enemy_group

    def calculate_reward(self):
        """Compute learning reward based on player proximity and movement efficiency."""
        if not self.player_target:
            return 0.0

        current_distance = (self.position - self.player_target.position).length()
        reward = 0.0

        distance_improvement = self.last_distance_to_player - current_distance
        reward += distance_improvement * 5.0

        if hasattr(self, 'velocity') and self.velocity.length() > 0:
            direction_to_player = (self.player_target.position - self.position)
            if direction_to_player.length() > 0:
                direction_to_player = direction_to_player.normalize()
                velocity_direction = self.velocity.normalize()
                alignment = direction_to_player.dot(velocity_direction)
                reward += alignment * 2.0
                if alignment < -0.5:
                    reward -= 3.0

        if current_distance < 30:
            reward += 30
        elif current_distance < 60:
            reward += 15.0
        elif current_distance < 100:
            reward += 8.0
        elif current_distance < 150:
            reward += 1.0

        if current_distance > 400:
            reward -= 2.0

        reward -= 0.2

        if hasattr(self, 'target_angle') and hasattr(self, 'current_angle'):
            angle_diff = abs(self.target_angle - self.current_angle)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            if angle_diff > math.pi / 2:
                reward -= 1.0
            elif angle_diff > math.pi / 4:
                reward -= 0.5

        self.last_distance_to_player = current_distance
        return reward
    
    def update(self, dt):
        self.pulse_timer += dt
        self.ai_glow_intensity = (math.sin(self.pulse_timer * 4) + 1) * 0.5

        if not self.active:
            return
        if not self.player_target:
            return

        current_state = self.state_collector.collect_state(self, self.player_target)
        angle_adjustment = self.brain.get_action(current_state)

        # Treat network output as desired turn direction relative to current heading.
        # Positive = turn left, Negative = turn right.
        turn_offset = angle_adjustment.item() * math.pi * 0.5
        self.target_angle = self.current_angle + turn_offset

        self.current_angle = self.current_angle % (2 * math.pi)
        self.target_angle = self.target_angle % (2 * math.pi)

        angle_diff = self.target_angle - self.current_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        max_turn_rate = 3.0 * dt
        if abs(angle_diff) > max_turn_rate:
            self.current_angle += max_turn_rate * (1 if angle_diff > 0 else -1)
        else:
            self.current_angle = self.target_angle

        self.current_angle = self.current_angle % (2 * math.pi)

        self.velocity = pygame.Vector2(
            math.cos(self.current_angle) * self.move_speed,
            math.sin(self.current_angle) * self.move_speed
        )

        self.position += self.velocity * dt

        if (self.position.x < -50 or self.position.x > SCREEN_WIDTH + 50 or
            self.position.y < -50 or self.position.y > SCREEN_HEIGHT + 50):
            if self.training_mode:
                self.respawn_near_player()
                return
            else:
                self.wrap_position()

        reward = self.calculate_reward()
        self.episode_reward += reward
        self.episode_length += 1
        self.brain.store_reward(reward)

        current_distance = (self.position - self.player_target.position).length()
        debug_step(self.enemy_id, [], angle_adjustment.item(), reward, current_distance)
        episode_done = False

        if self.training_mode:
            # Phase-dependent step limits: earlier phases allow more time to learn basics
            phase_step_limits = {1: 800, 2: 500, 3: 300}
            max_steps = phase_step_limits.get(self.brain.training_phase, 400)

            if self.episode_length >= max_steps:
                episode_done = True
            elif current_distance > 600:
                # Skip distance check during initial approach from spawn point
                if not self.initial_approach:
                    self.respawn_near_player()
                    episode_done = True

            # Mark initial approach complete once enemy gets reasonably close
            if self.initial_approach and current_distance < 400:
                self.initial_approach = False

            if episode_done:
                debug_episode_end(self.enemy_id, self.episode_reward, False, self.episode_length)
                self.brain.end_episode(self.episode_reward, success=False)

                if self.brain.should_advance_phase():
                    self.brain.advance_phase()

                self.episode_reward = 0.0
                self.episode_length = 0
                self.respawn_near_player()

        if self.brain.training_steps % 50 == 0 and self.episode_length == 1:
            print(f"[AI] Enemy #{self.enemy_id}: Phase {self.brain.training_phase}, Episode {self.brain.training_steps}")
            print(f"   Success: {self.brain.get_success_rate():.1%}, Distance: {current_distance:.0f}px")
            print(f"   Angle: {math.degrees(self.current_angle):.0f}, Target: {math.degrees(self.target_angle):.0f}")
            debug_print()
    
    def respawn_near_player(self):
        """Relocate enemy near player for a new training episode."""
        if not self.player_target:
            return

        angle = random.uniform(0, 2 * math.pi)

        # Earlier phases spawn closer to help the AI learn pursuit basics
        phase_distances = {1: (100, 200), 2: (150, 300), 3: (200, 400)}
        min_dist, max_dist = phase_distances.get(self.brain.training_phase, (200, 400))
        distance = random.uniform(min_dist, max_dist)

        spawn_pos = self.player_target.position + pygame.Vector2(
            math.cos(angle) * distance,
            math.sin(angle) * distance
        )

        spawn_pos.x = max(50, min(SCREEN_WIDTH - 50, spawn_pos.x))
        spawn_pos.y = max(50, min(SCREEN_HEIGHT - 50, spawn_pos.y))

        self.position = spawn_pos
        self.last_distance_to_player = (self.position - self.player_target.position).length()
        self.episode_start_distance = self.last_distance_to_player

        direction_to_player = self.player_target.position - self.position
        if direction_to_player.length() > 0:
            self.current_angle = math.atan2(direction_to_player.y, direction_to_player.x)
        self.target_angle = self.current_angle

        self.episode_ended = False
        self.initial_approach = False  
    
    def draw(self, screen):
        pulse = math.sin(self.pulse_timer * 3) * 0.2 + 1.0
        current_radius = self.radius * pulse

        phase_colors = {1: (100, 255, 100), 2: (255, 200, 100), 3: (255, 100, 100)}
        main_color = phase_colors.get(self.brain.training_phase, (100, 200, 255))

        glow_radius = int(current_radius * (1.3 + self.ai_glow_intensity * 0.3))
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = (*main_color, 25)
        pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)
        screen.blit(glow_surface, (self.position.x - glow_radius, self.position.y - glow_radius))

        pygame.draw.circle(screen, main_color, self.position, int(current_radius + 2), 2)
        pygame.draw.circle(screen, (20, 20, 60), self.position, int(current_radius))

        core_radius = int(current_radius * 0.6)
        pygame.draw.circle(screen, main_color, self.position, core_radius)

        arrow_length = current_radius * 2.5
        arrow_end = self.position + pygame.Vector2(
            math.cos(self.target_angle) * arrow_length,
            math.sin(self.target_angle) * arrow_length
        )

        pygame.draw.line(screen, (255, 255, 100), self.position, arrow_end, 3)

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

        current_end = self.position + pygame.Vector2(
            math.cos(self.current_angle) * (arrow_length * 0.7),
            math.sin(self.current_angle) * (arrow_length * 0.7)
        )
        pygame.draw.line(screen, (150, 150, 255), self.position, current_end, 2)

        font = pygame.font.Font(None, 16)
        ai_text = font.render(f"AI-P{self.brain.training_phase}", True, (255, 255, 255))
        text_rect = ai_text.get_rect()
        text_rect.center = (int(self.position.x), int(self.position.y - current_radius - 20))
        screen.blit(ai_text, text_rect)
    
    def kill(self):
        # Prevent double episode-end on game reset
        if not self.episode_ended:
            final_penalty = -20.0
            self.brain.store_reward(final_penalty)
            self.brain.end_episode(self.episode_reward + final_penalty, success=False)
            self.episode_ended = True
        super().kill()

class EnemySpawner:
    """Manages enemy spawning and lifecycle."""
    
    def __init__(self, enemy_group, enemy_type: str = "neural", training_mode: bool = False):
        """
        Initialize the enemy spawner.
        
        Args:
            enemy_group: Sprite group for spawned enemies
            enemy_type: Type of enemies to spawn
                - "neural": AI-controlled neural network enemies
                - "shooter": Enemies that shoot at player
                - "follower": Enemies that chase the player
                - "mixed": Random mix of shooter and follower
                - "none": No enemies spawned
            training_mode: Whether AI-controlled training behaviors are active
        """
        self.spawn_timer = 0
        self.enemy_group = enemy_group
        self.enemy_type = enemy_type
        self.training_mode = training_mode
        
    def update(self, dt, player, asteroid_group=None):
        self.spawn_timer += dt

        if self.enemy_type == "none":
            return

        # Limit to 1 enemy during training to prevent replay buffer corruption
        max_enemies = 1 if self.training_mode else ENEMY_MAX_COUNT

        if (self.spawn_timer >= ENEMY_SPAWN_RATE and
            len(self.enemy_group) < max_enemies):
            self.spawn_enemy(player, asteroid_group)
            self.spawn_timer = 0

    def spawn_enemy(self, player, asteroid_group=None):
        if self.enemy_type == "mixed":
            enemy_type = random.choice(["follower", "shooter"])
        else:
            enemy_type = self.enemy_type

        edge = random.randint(0, 3)

        if enemy_type == "neural":
            center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2

            if edge == 0:
                x = random.uniform(SCREEN_WIDTH * 0.2, SCREEN_WIDTH * 0.8)
                y = -ENEMY_RADIUS
                initial_velocity = pygame.Vector2(0, 1)
            elif edge == 1:
                x = SCREEN_WIDTH + ENEMY_RADIUS
                y = random.uniform(SCREEN_HEIGHT * 0.2, SCREEN_HEIGHT * 0.8)
                initial_velocity = pygame.Vector2(-1, 0)
            elif edge == 2:
                x = random.uniform(SCREEN_WIDTH * 0.2, SCREEN_WIDTH * 0.8)
                y = SCREEN_HEIGHT + ENEMY_RADIUS
                initial_velocity = pygame.Vector2(0, -1)
            else:
                x = -ENEMY_RADIUS
                y = random.uniform(SCREEN_HEIGHT * 0.2, SCREEN_HEIGHT * 0.8)
                initial_velocity = pygame.Vector2(1, 0)
            
            enemy = NeuralEnemy(x, y, training_mode=self.training_mode)
            enemy.set_target(player)

            spawn_pos = pygame.Vector2(x, y)
            center_pos = pygame.Vector2(center_x, center_y)
            direction_to_center = (center_pos - spawn_pos).normalize()
            enemy.velocity = direction_to_center * ENEMY_SPEED * 0.5

            if asteroid_group:
                enemy.set_game_groups(asteroid_group, self.enemy_group)
            print(f"Neural enemy spawned at ({x:.1f}, {y:.1f}) moving toward center")

        elif enemy_type == "shooter":
            if edge == 0:
                x = random.uniform(0, SCREEN_WIDTH)
                y = -SHOOTER_ENEMY_RADIUS
            elif edge == 1:
                x = SCREEN_WIDTH + SHOOTER_ENEMY_RADIUS
                y = random.uniform(0, SCREEN_HEIGHT)
            elif edge == 2:
                x = random.uniform(0, SCREEN_WIDTH)
                y = SCREEN_HEIGHT + SHOOTER_ENEMY_RADIUS
            else:
                x = -SHOOTER_ENEMY_RADIUS
                y = random.uniform(0, SCREEN_HEIGHT)
            
            enemy = ShooterEnemy(x, y)
            enemy.set_target(player)

        else:
            if edge == 0:
                x = random.uniform(0, SCREEN_WIDTH)
                y = -ENEMY_RADIUS
                initial_velocity = pygame.Vector2(0, 1)
            elif edge == 1:
                x = SCREEN_WIDTH + ENEMY_RADIUS
                y = random.uniform(0, SCREEN_HEIGHT)
                initial_velocity = pygame.Vector2(-1, 0)
            elif edge == 2:
                x = random.uniform(0, SCREEN_WIDTH)
                y = SCREEN_HEIGHT + ENEMY_RADIUS
                initial_velocity = pygame.Vector2(0, -1)
            else:
                x = -ENEMY_RADIUS
                y = random.uniform(0, SCREEN_HEIGHT)
                initial_velocity = pygame.Vector2(1, 0)
            
            enemy = Enemy(x, y)
            enemy.set_target(player)

            speed = random.uniform(ENEMY_SPEED * 0.8, ENEMY_SPEED * 1.2)
            enemy.velocity = initial_velocity * speed
            enemy.velocity = enemy.velocity.rotate(random.uniform(-20, 20))

        return enemy 