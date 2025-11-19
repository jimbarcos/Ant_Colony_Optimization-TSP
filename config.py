import pygame

# Screen settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors - Modern Color Palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)

# Modern UI Colors
BG_COLOR = (240, 240, 240)  # Greyish white background
PANEL_BG = (41, 50, 65)  # Dark blue-gray panel
CITY_COLOR = (255, 107, 107)  # Coral red for cities
CITY_OUTLINE = (220, 20, 60)  # Crimson outline
BEST_PATH_COLOR = (46, 213, 115)  # Emerald green
PHEROMONE_COLOR = (52, 152, 219)  # Dodger blue
ANT_COLOR = (44, 62, 80)  # Dark slate
GRID_COLOR = (220, 220, 220)
TEXT_COLOR = (52, 73, 94)  # Dark text
ACCENT_COLOR = (155, 89, 182)  # Purple accent

# ACO Defaults
DEFAULT_NUM_ANTS = 50
DEFAULT_ALPHA = 1.0  # Pheromone importance
DEFAULT_BETA = 2.0   # Distance importance
DEFAULT_EVAPORATION_RATE = 0.5
DEFAULT_Q = 100.0    # Pheromone deposit factor
DEFAULT_NUM_CITIES = 8
DEFAULT_GRID_SPACING = 100
DEFAULT_ANIMATION_SPEED = 200.0
ITERATION_CUTOFF = 30 # Iterations without improvement to consider converged

# UI Settings
UI_WIDTH = 300
