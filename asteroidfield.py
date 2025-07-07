"""Asteroid spawning system for procedural enemy generation."""

import pygame
import random
from asteroid import Asteroid
from constants import *


class AsteroidField(pygame.sprite.Sprite):
    # manages procedural asteroid spawning from screen edges
    edges = [
        [
            pygame.Vector2(1, 0),
            lambda y: pygame.Vector2(0, y * SCREEN_HEIGHT),
        ],
        [
            pygame.Vector2(-1, 0),
            lambda y: pygame.Vector2(
                SCREEN_WIDTH, y * SCREEN_HEIGHT
            ),
        ],
        [
            pygame.Vector2(0, 1),
            lambda x: pygame.Vector2(x * SCREEN_WIDTH, 0),
        ],
        [
            pygame.Vector2(0, -1),
            lambda x: pygame.Vector2(
                x * SCREEN_WIDTH, SCREEN_HEIGHT
            ),
        ],
    ]

    def __init__(self) -> None:
        # initialize spawn timer for asteroid generation
        pygame.sprite.Sprite.__init__(self, self.containers) # type: ignore
        self.spawn_timer = 0.0

    def spawn(self, radius: float, position: pygame.Vector2, velocity: pygame.Vector2) -> None:
        # create new asteroid with specified properties
        asteroid = Asteroid(position.x, position.y, radius)
        asteroid.velocity = velocity

    def update(self, dt: float) -> None:
        # handle asteroid spawning based on timer intervals
        self.spawn_timer += dt
        if self.spawn_timer > ASTEROID_SPAWN_RATE:
            self.spawn_timer = 0

            # select random edge and generate spawn parameters
            edge = random.choice(self.edges)
            speed = random.randint(100, 150)
            velocity = edge[0] * speed
            velocity = velocity.rotate(random.randint(-30, 30))
            position = edge[1](random.uniform(0, 1))
            kind = random.randint(1, ASTEROID_KINDS)
            self.spawn(ASTEROID_MIN_RADIUS * kind, position, velocity)