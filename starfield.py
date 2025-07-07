import pygame
import random
import math
from constants import SCREEN_WIDTH, SCREEN_HEIGHT

class Star:
    """individual star with position, speed, and visual properties"""
    def __init__(self, x, y, layer):
        self.x = x
        self.y = y
        self.layer = layer  # depth layer for parallax effect
        
        # configure properties based on depth layer
        if layer == 0:  # background stars
            self.speed = 2
            self.size = 1
            self.brightness = 80
            self.color = (self.brightness, self.brightness, self.brightness)
        elif layer == 1:  # middle layer stars
            self.speed = 6
            self.size = random.choice([1, 2])
            self.brightness = 150
            self.color = (self.brightness, self.brightness, self.brightness)
        else:  # foreground stars
            self.speed = 12
            self.size = random.choice([2, 3])
            self.brightness = 255
            self.color = (self.brightness, self.brightness, self.brightness)
        
        # add color variation for visual interest
        if random.random() < 0.1:
            if random.random() < 0.5:
                self.color = (self.brightness, self.brightness // 2, self.brightness // 2)  # reddish tint
            else:
                self.color = (self.brightness // 2, self.brightness // 2, self.brightness)  # bluish tint

class Starfield:
    """manages a parallax starfield with multiple depth layers"""
    
    def __init__(self, num_layers=3, stars_per_layer=150):
        self.num_layers = num_layers
        self.stars_per_layer = stars_per_layer
        self.stars = []
        
        # generate stars for each depth layer
        for layer in range(num_layers):
            layer_stars = []
            for _ in range(stars_per_layer):
                x = random.randint(0, SCREEN_WIDTH)
                y = random.randint(0, SCREEN_HEIGHT)
                star = Star(x, y, layer)
                layer_stars.append(star)
            self.stars.append(layer_stars)
    
    def update(self, dt, player_velocity=None):
        """update star positions with parallax movement"""
        # base drift speed for natural movement
        base_speed_x = 1
        base_speed_y = 0.5
        
        # apply player movement influence for enhanced parallax
        player_influence_x = 0
        player_influence_y = 0
        if player_velocity:
            player_influence_x = player_velocity.x * 0.02
            player_influence_y = player_velocity.y * 0.02
        
        for layer_idx, layer_stars in enumerate(self.stars):
            for star in layer_stars:
                # calculate movement based on layer speed and influences
                move_x = (base_speed_x + player_influence_x) * star.speed * dt
                move_y = (base_speed_y + player_influence_y) * star.speed * dt
                
                star.x += move_x
                star.y += move_y
                
                # wrap stars around screen boundaries
                if star.x > SCREEN_WIDTH + 10:
                    star.x = -10
                    star.y = random.randint(0, SCREEN_HEIGHT)
                elif star.x < -10:
                    star.x = SCREEN_WIDTH + 10
                    star.y = random.randint(0, SCREEN_HEIGHT)
                    
                if star.y > SCREEN_HEIGHT + 10:
                    star.y = -10
                    star.x = random.randint(0, SCREEN_WIDTH)
                elif star.y < -10:
                    star.y = SCREEN_HEIGHT + 10
                    star.x = random.randint(0, SCREEN_WIDTH)
    
    def draw(self, screen):
        """render all stars from background to foreground"""
        for layer_stars in self.stars:
            for star in layer_stars:
                if star.size == 1:
                    # render single pixel star
                    pygame.draw.circle(screen, star.color, (int(star.x), int(star.y)), 1)
                else:
                    # render larger stars with glow effect
                    pygame.draw.circle(screen, star.color, (int(star.x), int(star.y)), star.size)
                    
                    # add subtle glow for depth
                    if star.size > 1:
                        glow_color = tuple(c // 3 for c in star.color)
                        pygame.draw.circle(screen, glow_color, (int(star.x), int(star.y)), star.size + 1)
    
    def add_twinkle_effect(self, screen):
        """add occasional twinkling to random stars"""
        if random.random() < 0.05:  # 5% chance per frame
            # select random star from any layer
            layer = random.choice(self.stars)
            if layer:
                star = random.choice(layer)
                # render brief bright flash
                twinkle_color = (255, 255, 255)
                pygame.draw.circle(screen, twinkle_color, (int(star.x), int(star.y)), star.size + 2, 1) 