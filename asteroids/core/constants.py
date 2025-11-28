SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900

ASTEROID_MIN_RADIUS = 20
ASTEROID_KINDS = 3
ASTEROID_SPAWN_RATE = 9999
ASTEROID_MAX_RADIUS = ASTEROID_MIN_RADIUS * ASTEROID_KINDS

PLAYER_RADIUS = 20
PLAYER_TURN_SPEED = 200
PLAYER_SPEED = 200
PLAYER_SHOOT_SPEED = 500
PLAYER_SHOOT_COOLDOWN = 0.1 # seconds

SHOT_RADIUS = 5

# Enemy constants
ENEMY_RADIUS = 15
ENEMY_SPEED = 50
ENEMY_FOLLOW_SPEED = ENEMY_SPEED * 2  # Faster speed when following player
ENEMY_MAX_COUNT = 2  # Even fewer enemies for clearer observation
ENEMY_SPAWN_RATE = 3.0  # seconds between spawns (much slower for clear observation)
ENEMY_DETECTION_RANGE = 600  # How close to player before enemy starts following

# Shooter enemy constants
SHOOTER_ENEMY_RADIUS = 18
SHOOTER_ENEMY_SPEED = 80
SHOOTER_SHOOT_COOLDOWN = 2.0  # seconds between shots
SHOOTER_SHOOT_RANGE = 400  # How close to player before shooting
SHOOTER_SHOT_SPEED = 200
SHOOTER_SHOT_RADIUS = 4
