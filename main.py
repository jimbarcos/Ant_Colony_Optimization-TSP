import pygame
import pygame_gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config import *
from aco import ACO
import math

class AntSprite(pygame.sprite.Sprite):
    def __init__(self, image, start_pos, speed=5.0):
        super().__init__()
        self.original_image = image
        self.image = image
        self.rect = self.image.get_rect(center=start_pos)
        self.pos = np.array(start_pos, dtype=float)
        self.target_pos = None
        self.speed = speed
        self.path = []
        self.current_path_index = 0
        self.finished = False

    def set_path(self, path_coords):
        self.path = path_coords
        self.current_path_index = 0
        if len(self.path) > 1:
            self.target_pos = np.array(self.path[1], dtype=float)
            self.update_rotation()

    def update_rotation(self):
        if self.target_pos is not None:
            direction = self.target_pos - self.pos
            angle = math.degrees(math.atan2(-direction[1], direction[0])) - 90
            self.image = pygame.transform.rotate(self.original_image, angle)
            self.rect = self.image.get_rect(center=self.rect.center)

    def update(self):
        if self.finished or not self.path:
            return

        direction = self.target_pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = self.target_pos
            self.current_path_index += 1
            if self.current_path_index >= len(self.path) - 1:
                self.finished = True
            else:
                self.target_pos = np.array(self.path[self.current_path_index + 1], dtype=float)
                self.update_rotation()
        else:
            normalized_dir = direction / distance
            self.pos += normalized_dir * self.speed
        
        self.rect.center = self.pos

class TSPVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("TSP - Ant Colony Optimization")
        self.clock = pygame.time.Clock()
        
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Assets
        self.create_assets()

        # Simulation State
        self.running_simulation = False
        self.animating = False
        self.show_grid = True
        self.show_labels = False
        self.show_dist_overlay = True
        self.grid_spacing = DEFAULT_GRID_SPACING
        
        self.visible_best_tour = None
        self.visible_best_distance = float('inf')
        self.visible_last_improvement_iter = 0
        self.visible_iteration = 0
        self.converged = False
        self.show_overlays = True
        self.distance_history = []

        self.aco = ACO(
            DEFAULT_NUM_CITIES, 
            DEFAULT_NUM_ANTS, 
            DEFAULT_ALPHA, 
            DEFAULT_BETA, 
            DEFAULT_EVAPORATION_RATE, 
            DEFAULT_Q,
            SCREEN_WIDTH - UI_WIDTH, 
            SCREEN_HEIGHT,
            grid_spacing=DEFAULT_GRID_SPACING # Grid on by default
        )
        
        self.ant_sprites = pygame.sprite.Group()
        self.setup_ui()

    def handle_resize(self, width, height):
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.manager.set_window_resolution((width, height))
        
        # Update UI Panel position
        panel_rect = pygame.Rect(width - UI_WIDTH, 0, UI_WIDTH, height)
        self.panel.set_relative_position(panel_rect.topleft)
        self.panel.set_dimensions(panel_rect.size)
        
        # Update ACO bounds for future generations
        self.aco.width = width - UI_WIDTH
        self.aco.height = height

    def create_assets(self):
        # City Icon (Modern pin/marker style)
        self.city_img = pygame.Surface((24, 28), pygame.SRCALPHA)
        # Main circle
        pygame.draw.circle(self.city_img, CITY_COLOR, (12, 10), 10)
        pygame.draw.circle(self.city_img, CITY_OUTLINE, (12, 10), 10, 2)
        # Inner dot
        pygame.draw.circle(self.city_img, WHITE, (12, 10), 4)
        # Pin point
        pygame.draw.polygon(self.city_img, CITY_COLOR, [(12, 20), (8, 15), (16, 15)])
        pygame.draw.polygon(self.city_img, CITY_OUTLINE, [(12, 20), (8, 15), (16, 15)], 2)
        
        # Ant Icon (Modern bug style with gradient effect)
        self.ant_img = pygame.Surface((20, 20), pygame.SRCALPHA)
        # Body with gradient effect
        pygame.draw.circle(self.ant_img, ANT_COLOR, (10, 6), 4) # Head
        pygame.draw.circle(self.ant_img, (60, 80, 100), (10, 6), 4, 1) # Head outline
        pygame.draw.ellipse(self.ant_img, ANT_COLOR, (7, 10, 6, 8)) # Thorax
        pygame.draw.circle(self.ant_img, ANT_COLOR, (10, 16), 5) # Abdomen
        pygame.draw.circle(self.ant_img, (60, 80, 100), (10, 16), 5, 1) # Abdomen outline
        # Antennae
        pygame.draw.line(self.ant_img, ANT_COLOR, (10, 6), (7, 2), 2)
        pygame.draw.line(self.ant_img, ANT_COLOR, (10, 6), (13, 2), 2)
        # Legs (6 legs)
        pygame.draw.line(self.ant_img, ANT_COLOR, (8, 12), (3, 10), 2)
        pygame.draw.line(self.ant_img, ANT_COLOR, (12, 12), (17, 10), 2)
        pygame.draw.line(self.ant_img, ANT_COLOR, (8, 15), (3, 17), 2)
        pygame.draw.line(self.ant_img, ANT_COLOR, (12, 15), (17, 17), 2)
        pygame.draw.line(self.ant_img, ANT_COLOR, (8, 18), (5, 22), 2)
        pygame.draw.line(self.ant_img, ANT_COLOR, (12, 18), (15, 22), 2)

    def setup_ui(self):
        width, height = self.screen.get_size()
        panel_rect = pygame.Rect(width - UI_WIDTH, 0, UI_WIDTH, height)
        self.panel = pygame_gui.elements.UIPanel(
            relative_rect=panel_rect,
            manager=self.manager
        )

        # Title
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 10, UI_WIDTH - 20, 30),
            text="Configuration",
            manager=self.manager,
            container=self.panel
        )

        # Controls
        y_offset = 50
        
        # Num Cities
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Number of Cities:",
            manager=self.manager,
            container=self.panel
        )
        self.slider_cities = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, y_offset + 25, UI_WIDTH - 20, 20),
            start_value=DEFAULT_NUM_CITIES,
            value_range=(5, 100),
            manager=self.manager,
            container=self.panel
        )
        self.label_cities_val = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset + 45, UI_WIDTH - 20, 20),
            text=str(DEFAULT_NUM_CITIES),
            manager=self.manager,
            container=self.panel
        )
        y_offset += 80

        # Num Ants
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Number of Ants:",
            manager=self.manager,
            container=self.panel
        )
        self.slider_ants = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, y_offset + 25, UI_WIDTH - 20, 20),
            start_value=DEFAULT_NUM_ANTS,
            value_range=(1, 100),
            manager=self.manager,
            container=self.panel
        )
        self.label_ants_val = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset + 45, UI_WIDTH - 20, 20),
            text=str(DEFAULT_NUM_ANTS),
            manager=self.manager,
            container=self.panel
        )
        y_offset += 80

        # Alpha
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Alpha (Pheromone):",
            manager=self.manager,
            container=self.panel
        )
        self.slider_alpha = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, y_offset + 25, UI_WIDTH - 20, 20),
            start_value=DEFAULT_ALPHA,
            value_range=(0.1, 5.0),
            manager=self.manager,
            container=self.panel
        )
        self.label_alpha_val = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset + 45, UI_WIDTH - 20, 20),
            text=f"{DEFAULT_ALPHA:.2f}",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 80

        # Beta
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Beta (Distance):",
            manager=self.manager,
            container=self.panel
        )
        self.slider_beta = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, y_offset + 25, UI_WIDTH - 20, 20),
            start_value=DEFAULT_BETA,
            value_range=(0.1, 10.0),
            manager=self.manager,
            container=self.panel
        )
        self.label_beta_val = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset + 45, UI_WIDTH - 20, 20),
            text=f"{DEFAULT_BETA:.2f}",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 80

        # Evaporation
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Evaporation Rate:",
            manager=self.manager,
            container=self.panel
        )
        self.slider_evap = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, y_offset + 25, UI_WIDTH - 20, 20),
            start_value=DEFAULT_EVAPORATION_RATE,
            value_range=(0.01, 0.99),
            manager=self.manager,
            container=self.panel
        )
        self.label_evap_val = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset + 45, UI_WIDTH - 20, 20),
            text=f"{DEFAULT_EVAPORATION_RATE:.2f}",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 80

        # Animation Speed
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Animation Speed:",
            manager=self.manager,
            container=self.panel
        )
        self.slider_speed = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, y_offset + 25, UI_WIDTH - 20, 20),
            start_value=DEFAULT_ANIMATION_SPEED,
            value_range=(10.0, 5000.0),
            manager=self.manager,
            container=self.panel
        )
        self.label_speed_val = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset + 45, UI_WIDTH - 20, 20),
            text=f"{DEFAULT_ANIMATION_SPEED:.1f}",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 80

        # Buttons
        self.btn_start = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 40),
            text="Start/Pause",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 50
        
        self.btn_reset = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 40),
            text="Reset / Generate New",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 50

        self.btn_reset_pheromones = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 40),
            text="Reset Sim (Keep Map)",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 50

        # Toggles
        self.chk_labels = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 30),
            text="Toggle City Labels",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 40

        self.btn_toggle_overlays = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 30),
            text="Toggle Overlays",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 40

        self.btn_show_chart = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 30),
            text="Show Convergence Chart",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 40

        # Stats
        self.label_iteration = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Iteration: 0",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 30
        self.label_best_dist = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Best Distance: N/A",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 30
        self.label_best_found_at = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y_offset, UI_WIDTH - 20, 20),
            text="Best Found At: -",
            manager=self.manager,
            container=self.panel
        )
        y_offset += 40

    def update_ui_labels(self):
        self.label_cities_val.set_text(str(int(self.slider_cities.get_current_value())))
        self.label_ants_val.set_text(str(int(self.slider_ants.get_current_value())))
        self.label_alpha_val.set_text(f"{self.slider_alpha.get_current_value():.2f}")
        self.label_beta_val.set_text(f"{self.slider_beta.get_current_value():.2f}")
        self.label_evap_val.set_text(f"{self.slider_evap.get_current_value():.2f}")
        self.label_speed_val.set_text(f"{self.slider_speed.get_current_value():.1f}")
        
        self.label_iteration.set_text(f"Iteration: {self.visible_iteration}")
        if self.visible_best_distance != float('inf'):
            converged_text = ""
            if self.converged:
                converged_text = " (Converged)"
            self.label_best_dist.set_text(f"Best Dist: {self.visible_best_distance:.1f}{converged_text}")
            self.label_best_found_at.set_text(f"Best Found At: {self.visible_last_improvement_iter}")
        else:
            self.label_best_dist.set_text("Best Distance: N/A")
            self.label_best_found_at.set_text("Best Found At: -")

    def apply_settings(self):
        # Only apply settings that don't require a full reset if simulation is running
        # But for simplicity, we might just update the ACO parameters
        self.aco.num_ants = int(self.slider_ants.get_current_value())
        self.aco.alpha = self.slider_alpha.get_current_value()
        self.aco.beta = self.slider_beta.get_current_value()
        self.aco.evaporation_rate = self.slider_evap.get_current_value()
        
        # Changing cities requires reset
        new_cities = int(self.slider_cities.get_current_value())
        
        # Check if grid mode changed implicitly by user action (handled in button event)
        # Here we just check if we need to reset
        if new_cities != self.aco.num_cities:
            self.aco.num_cities = new_cities
            self.aco.reset()
            self.visible_best_tour = None
            self.visible_best_distance = float('inf')
            self.visible_last_improvement_iter = 0
            self.visible_iteration = 0
            self.converged = False
            self.running_simulation = False
            self.animating = False
            self.ant_sprites.empty()
            self.distance_history = []

    def toggle_grid(self):
        self.show_grid = not self.show_grid
        if self.show_grid:
            self.aco.grid_spacing = self.grid_spacing
        else:
            self.aco.grid_spacing = None
        self.aco.reset()
        self.visible_best_tour = None
        self.visible_best_distance = float('inf')
        self.visible_last_improvement_iter = 0
        self.visible_iteration = 0
        self.converged = False
        self.running_simulation = False
        self.animating = False
        self.ant_sprites.empty()
        self.distance_history = []

    def draw_grid(self):
        if self.show_grid:
            width, height = self.screen.get_size()
            for x in range(0, width - UI_WIDTH, self.grid_spacing):
                pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, height), 1)
            for y in range(0, height, self.grid_spacing):
                pygame.draw.line(self.screen, GRID_COLOR, (0, y), (width - UI_WIDTH, y), 1)
            # Add grid dots at intersections for better visual
            for x in range(0, width - UI_WIDTH, self.grid_spacing):
                for y in range(0, height, self.grid_spacing):
                    pygame.draw.circle(self.screen, (200, 200, 200), (x, y), 2)

    def draw_legend(self):
        # Modern Legend Box with shadow effect
        width, height = self.screen.get_size()
        sim_width = width - UI_WIDTH
        legend_width = 220
        legend_height = 130
        margin = 10
        
        legend_rect = pygame.Rect(sim_width - legend_width - margin, height - legend_height - margin, legend_width, legend_height)
        
        # Shadow
        shadow = pygame.Surface((legend_rect.width + 4, legend_rect.height + 4))
        shadow.set_alpha(30)
        shadow.fill(BLACK)
        self.screen.blit(shadow, (legend_rect.x + 4, legend_rect.y + 4))
        
        # Background with gradient effect
        s = pygame.Surface((legend_rect.width, legend_rect.height))
        s.set_alpha(240)
        s.fill(WHITE)
        self.screen.blit(s, legend_rect.topleft)
        pygame.draw.rect(self.screen, ACCENT_COLOR, legend_rect, 3, border_radius=8)
        
        # Title
        font_title = pygame.font.SysFont('Arial', 18, bold=True)
        font_item = pygame.font.SysFont('Arial', 16)
        
        title = font_title.render("Legend", True, TEXT_COLOR)
        self.screen.blit(title, (legend_rect.x + 10, legend_rect.y + 8))
        
        # Items
        x = legend_rect.x + 15
        y = legend_rect.y + 35
        
        # City
        city_rect = self.city_img.get_rect(center=(x + 12, y + 10))
        self.screen.blit(self.city_img, city_rect)
        text = font_item.render("City", True, TEXT_COLOR)
        self.screen.blit(text, (x + 35, y - 5))
        y += 28
        
        # Ant
        ant_rect = self.ant_img.get_rect(center=(x + 12, y + 5))
        self.screen.blit(self.ant_img, ant_rect)
        text = font_item.render("Ant", True, TEXT_COLOR)
        self.screen.blit(text, (x + 35, y - 5))
        y += 28
        
        # Best Path
        pygame.draw.line(self.screen, BEST_PATH_COLOR, (x, y ), (x + 25, y), 4)
        text = font_item.render("Best Path", True, TEXT_COLOR)
        self.screen.blit(text, (x + 35, y-10))
        y += 28
        
        # Pheromone
        pygame.draw.line(self.screen, PHEROMONE_COLOR, (x, y-5), (x + 25, y-5), 3)
        text = font_item.render("Pheromone", True, TEXT_COLOR)
        self.screen.blit(text, (x + 35, y - 15))

    def draw_aco(self):
        self.draw_grid()
        width, height = self.screen.get_size()

        # Draw Pheromones
        # Create a surface for pheromones to handle alpha blending efficiently
        pheromone_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        max_pheromone = np.max(self.aco.pheromones)
        if max_pheromone > 0 and not self.converged:
            for i in range(self.aco.num_cities):
                for j in range(i + 1, self.aco.num_cities):
                    strength = self.aco.pheromones[i][j] / max_pheromone
                    if strength > 0.05: # Threshold to avoid drawing everything
                        # Scale alpha: 0-255 with gradient
                        alpha = int(180 * strength)
                        # Color: Blue gradient
                        color = (PHEROMONE_COLOR[0], PHEROMONE_COLOR[1], PHEROMONE_COLOR[2], alpha)
                        start_pos = self.aco.cities[i]
                        end_pos = self.aco.cities[j]
                        
                        # Draw thicker lines for stronger pheromones
                        width = max(1, int(3 * strength))
                        pygame.draw.line(pheromone_surface, color, start_pos, end_pos, width)
        
        self.screen.blit(pheromone_surface, (0,0))

        # Draw Best Path with glow effect
        if self.visible_best_tour:
            tour = self.visible_best_tour
            # Draw glow (outer layer)
            for i in range(len(tour)):
                start_pos = self.aco.cities[tour[i]]
                end_pos = self.aco.cities[tour[(i + 1) % len(tour)]]
                # Glow effect
                glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                pygame.draw.line(glow_surface, (46, 213, 115, 100), start_pos, end_pos, 8)
                self.screen.blit(glow_surface, (0, 0))
            # Draw main path
            for i in range(len(tour)):
                start_pos = self.aco.cities[tour[i]]
                end_pos = self.aco.cities[tour[(i + 1) % len(tour)]]
                pygame.draw.line(self.screen, BEST_PATH_COLOR, start_pos, end_pos, 4)

        # Draw Edge Distance Labels (only for best path)
        if self.visible_best_tour and self.show_labels:
            tour = self.visible_best_tour
            font = pygame.font.SysFont('Arial', 12)
            for i in range(len(tour)):
                from_city_idx = tour[i]
                to_city_idx = tour[(i + 1) % len(tour)]
                start_pos = self.aco.cities[from_city_idx]
                end_pos = self.aco.cities[to_city_idx]
                
                # Calculate midpoint
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                
                # Calculate distance
                dist = self.aco.distances[from_city_idx][to_city_idx]
                
                # Draw distance label with background
                text = font.render(f"{int(dist)}", True, TEXT_COLOR)
                text_rect = text.get_rect(center=(int(mid_x), int(mid_y)))
                
                # Background rectangle
                bg_rect = text_rect.inflate(6, 4)
                bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
                bg_surface.set_alpha(200)
                bg_surface.fill(WHITE)
                self.screen.blit(bg_surface, bg_rect.topleft)
                pygame.draw.rect(self.screen, ACCENT_COLOR, bg_rect, 1)
                
                self.screen.blit(text, text_rect)
        
        # Draw Cities
        for idx, city in enumerate(self.aco.cities):
            # Draw City Icon
            rect = self.city_img.get_rect(center=(int(city[0]), int(city[1])))
            self.screen.blit(self.city_img, rect)
            
            if self.show_labels:
                font = pygame.font.SysFont('Arial', 16, bold=True)
                text_str = str(idx)
                
                # Check for start/end if best tour exists
                if self.visible_best_tour and idx == self.visible_best_tour[0]:
                     text_str += " (Start/End)"
                
                text = font.render(text_str, True, WHITE)
                
                # Position text
                text_rect = text.get_rect(midleft=(rect.right + 5, rect.centery))
                
                # Background for readability (pill shape or rect)
                bg_rect = text_rect.inflate(10, 6)
                pygame.draw.rect(self.screen, CITY_OUTLINE, bg_rect, border_radius=10)
                pygame.draw.rect(self.screen, CITY_COLOR, bg_rect.inflate(-2, -2), border_radius=9)
                
                self.screen.blit(text, text_rect)

        # Draw Ants
        self.ant_sprites.draw(self.screen)
        
        # Draw Legend and Stats Panel
        if self.show_overlays:
            self.draw_legend()
            
            # Draw Modern Stats Panel Overlay
            # Create stats panel
            panel_width = 340
            panel_height = 130
            panel_x = 15
            panel_y = 15
            
            # Shadow
            shadow = pygame.Surface((panel_width + 4, panel_height + 4))
            shadow.set_alpha(40)
            shadow.fill(BLACK)
            self.screen.blit(shadow, (panel_x + 4, panel_y + 4))
            
            # Background
            panel_surface = pygame.Surface((panel_width, panel_height))
            panel_surface.set_alpha(230)
            panel_surface.fill(WHITE)
            self.screen.blit(panel_surface, (panel_x, panel_y))
            pygame.draw.rect(self.screen, ACCENT_COLOR, (panel_x, panel_y, panel_width, panel_height), 3, border_radius=10)
            
            # Fonts
            font_title = pygame.font.SysFont('Arial', 16, bold=True)
            font_value = pygame.font.SysFont('Arial', 24, bold=True)
            font_label = pygame.font.SysFont('Arial', 14)
            
            x_pos = panel_x + 15
            y_pos = panel_y + 15
            
            # Distance
            dist_str = "N/A"
            color = TEXT_COLOR
            if self.visible_best_distance != float('inf'):
                dist_str = f"{self.visible_best_distance:.1f}"
                color = BEST_PATH_COLOR
            
            label = font_label.render("Best Distance:", True, TEXT_COLOR)
            self.screen.blit(label, (x_pos, y_pos))
            value = font_value.render(dist_str, True, color)
            self.screen.blit(value, (x_pos, y_pos + 18))
            
            # Current Iteration
            x_pos = panel_x + 180
            label = font_label.render("Current Iteration:", True, TEXT_COLOR)
            self.screen.blit(label, (x_pos, y_pos))
            value = font_value.render(f"{self.visible_iteration}", True, PHEROMONE_COLOR)
            self.screen.blit(value, (x_pos, y_pos + 18))
            
            # Best Found At
            x_pos = panel_x + 15
            y_pos += 60
            found_at = (self.visible_last_improvement_iter) if self.visible_best_distance != float('inf') else "-"
            label = font_label.render("Best Found at Iteration:", True, TEXT_COLOR)
            self.screen.blit(label, (x_pos, y_pos))
            value = font_value.render(str(found_at), True, ACCENT_COLOR)
            self.screen.blit(value, (x_pos, y_pos + 18))
            
            # Convergence Status
            if self.converged:
                x_pos = panel_x + 180
                label = font_label.render("Status:", True, TEXT_COLOR)
                self.screen.blit(label, (x_pos, y_pos))
                value = font_value.render("CONVERGED", True, RED)
                self.screen.blit(value, (x_pos, y_pos + 18))

    def show_chart(self):
        if not self.distance_history:
            return
        # Convert recorded distances to nearest integers for display
        integer_dist = [int(round(d)) for d in self.distance_history]

        plt.figure(figsize=(10, 6))
        plt.plot(integer_dist, linewidth=2, color='blue', label='Best Distance (int)')
        
        # Mark the start
        plt.scatter(0, integer_dist[0], color='green', zorder=5, label='Start')
        plt.annotate(f'{integer_dist[0]}', (0, integer_dist[0]), 
                     textcoords="offset points", xytext=(10,10), ha='center')
        
        # Mark the end
        end_idx = len(self.distance_history) - 1
        plt.scatter(end_idx, integer_dist[-1], color='red', zorder=5, label='Current/End')
        plt.annotate(f'{integer_dist[-1]}', (end_idx, integer_dist[-1]), 
                     textcoords="offset points", xytext=(10,-15), ha='center')

        plt.title('Convergence History')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        # Force integer ticks for both axes
        ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
        plt.show()

    def start_animation(self, tours):
        self.animating = True
        self.ant_sprites.empty()
        
        speed = self.slider_speed.get_current_value()
        
        # Create ant sprites for each tour
        for tour in tours:
            start_city_idx = tour[0]
            start_pos = self.aco.cities[start_city_idx]
            
            ant = AntSprite(self.ant_img, start_pos, speed)
            
            # Build path coordinates
            path_coords = []
            for city_idx in tour:
                path_coords.append(self.aco.cities[city_idx])
            # Add return to start
            path_coords.append(self.aco.cities[tour[0]])
            
            ant.set_path(path_coords)
            self.ant_sprites.add(ant)

    def run(self):
        running = True
        while running:
            time_delta = self.clock.tick(FPS) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.VIDEORESIZE:
                    self.handle_resize(event.w, event.h)

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.btn_start:
                        self.running_simulation = not self.running_simulation
                        if not self.running_simulation:
                            self.animating = False # Stop animation if paused
                    elif event.ui_element == self.btn_reset:
                        self.apply_settings() # Apply all settings including num cities
                        self.aco.reset()
                        self.visible_best_tour = None
                        self.visible_best_distance = float('inf')
                        self.visible_last_improvement_iter = 0
                        self.visible_iteration = 0
                        self.converged = False
                        self.running_simulation = False
                        self.animating = False
                        self.ant_sprites.empty()
                        self.distance_history = []
                    elif event.ui_element == self.btn_reset_pheromones:
                        self.aco.pheromones = np.ones((self.aco.num_cities, self.aco.num_cities)) * 0.1
                        self.aco.iteration = 0
                        self.aco.best_tour = None
                        self.aco.best_distance = float('inf')
                        self.aco.last_improvement_iter = 0
                        self.visible_best_tour = None
                        self.visible_best_distance = float('inf')
                        self.visible_last_improvement_iter = 0
                        self.visible_iteration = 0
                        self.converged = False
                        self.running_simulation = False
                        self.animating = False
                        self.ant_sprites.empty()
                        self.distance_history = []
                    elif event.ui_element == self.chk_labels:
                        self.show_labels = not self.show_labels
                    elif event.ui_element == self.btn_toggle_overlays:
                        self.show_overlays = not self.show_overlays
                    elif event.ui_element == self.btn_show_chart:
                        # Show convergence chart
                        self.show_chart()
                
                # Update settings in real-time if possible, or on release
                if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    self.update_ui_labels()
                    # We can update some params live
                    self.aco.num_ants = int(self.slider_ants.get_current_value())
                    self.aco.alpha = self.slider_alpha.get_current_value()
                    self.aco.beta = self.slider_beta.get_current_value()
                    self.aco.evaporation_rate = self.slider_evap.get_current_value()
                    # Update speed for all current ants immediately
                    new_speed = self.slider_speed.get_current_value()
                    for ant in self.ant_sprites:
                        ant.speed = new_speed

                self.manager.process_events(event)

            self.manager.update(time_delta)
            
            if self.running_simulation:
                if not self.animating:
                    # Check for convergence
                    if self.aco.iteration > 0 and (self.aco.iteration - self.aco.last_improvement_iter > ITERATION_CUTOFF):
                        self.converged = True
                        tours, distances = self.aco.run_best_path_demo()
                    else:
                        # Run one iteration logic
                        tours, distances = self.aco.run_iteration()
                    
                    if self.aco.best_distance != float('inf'):
                        self.distance_history.append(self.aco.best_distance)

                    self.update_ui_labels()
                    # Start animation
                    self.start_animation(tours)
                else:
                    # Update animation
                    self.ant_sprites.update()
                    # Check if all ants finished
                    all_finished = True
                    for ant in self.ant_sprites:
                        if not ant.finished:
                            all_finished = False
                            break
                    
                    if all_finished:
                        self.animating = False
                        # Update visible bests only after animation finishes
                        self.visible_best_tour = self.aco.best_tour
                        self.visible_best_distance = self.aco.best_distance
                        self.visible_last_improvement_iter = self.aco.last_improvement_iter
                        self.visible_iteration = self.aco.iteration
                        self.update_ui_labels()
                        
                        # Pheromones are already updated in run_iteration, 
                        # but visually we might want to update them here if we were deferring it.
                        # For now, they update instantly at start of iteration, which is fine.

            self.screen.fill(WHITE)
            
            # Draw simulation area with gradient background
            width, height = self.screen.get_size()
            sim_surface = pygame.Surface((width - UI_WIDTH, height))
            sim_surface.fill(BG_COLOR)
            self.screen.blit(sim_surface, (0, 0))
            
            self.draw_aco()
            self.manager.draw_ui(self.screen)
            
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    app = TSPVisualizer()
    app.run()
