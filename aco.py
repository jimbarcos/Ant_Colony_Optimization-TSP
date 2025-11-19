import numpy as np
import random

class ACO:
    def __init__(self, num_cities, num_ants, alpha, beta, evaporation_rate, q, width, height, grid_spacing=None):
        self.num_cities = num_cities
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.width = width
        self.height = height
        self.grid_spacing = grid_spacing
        
        self.cities = self.generate_cities()
        self.distances = self.calculate_distances()
        self.pheromones = np.ones((num_cities, num_cities)) * 0.1
        
        self.best_tour = None
        self.best_distance = float('inf')
        self.iteration = 0
        self.last_improvement_iter = 0

    def generate_cities(self):
        # Generate random cities within the bounds, leaving some padding
        padding = 50
        cities = []
        
        if self.grid_spacing:
            # Grid based generation
            cols = (self.width - 2 * padding) // self.grid_spacing
            rows = (self.height - 2 * padding) // self.grid_spacing
            possible_points = []
            for r in range(int(rows) + 1):
                for c in range(int(cols) + 1):
                    x = padding + c * self.grid_spacing
                    y = padding + r * self.grid_spacing
                    possible_points.append((x, y))
            
            if len(possible_points) >= self.num_cities:
                indices = random.sample(range(len(possible_points)), self.num_cities)
                cities = [possible_points[i] for i in indices]
            else:
                # Fallback if grid is too small
                cities = possible_points
                # Fill rest randomly
                for _ in range(self.num_cities - len(cities)):
                    x = random.randint(padding, self.width - padding)
                    y = random.randint(padding, self.height - padding)
                    cities.append((x, y))
        else:
            for _ in range(self.num_cities):
                x = random.randint(padding, self.width - padding)
                y = random.randint(padding, self.height - padding)
                cities.append((x, y))
                
        return np.array(cities)

    def calculate_distances(self):
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dist = np.linalg.norm(self.cities[i] - self.cities[j])
                    distances[i][j] = dist
        return distances

    def run_iteration(self):
        all_tours = []
        all_distances = []

        # Move ants
        for _ in range(self.num_ants):
            tour = self.construct_solution()
            dist = self.calculate_tour_distance(tour)
            all_tours.append(tour)
            all_distances.append(dist)
            
            if dist < self.best_distance:
                self.best_distance = dist
                self.best_tour = tour
                self.last_improvement_iter = self.iteration

        # Update pheromones
        self.update_pheromones(all_tours, all_distances)
        self.iteration += 1
        
        return all_tours, all_distances

    def construct_solution(self):
        start_city = random.randint(0, self.num_cities - 1)
        tour = [start_city]
        visited = {start_city}

        current_city = start_city
        while len(tour) < self.num_cities:
            probabilities = self.calculate_probabilities(current_city, visited)
            next_city = self.select_next_city(probabilities)
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        return tour

    def calculate_probabilities(self, current_city, visited):
        pheromones = self.pheromones[current_city]
        distances = self.distances[current_city]
        
        probabilities = []
        available_cities = []

        for city in range(self.num_cities):
            if city not in visited:
                tau = pheromones[city] ** self.alpha
                eta = (1.0 / distances[city]) ** self.beta if distances[city] > 0 else 0
                probabilities.append(tau * eta)
                available_cities.append(city)
        
        total = sum(probabilities)
        if total == 0:
            # Should not happen if initialized correctly, but fallback to uniform
            return [1.0 / len(available_cities)] * len(available_cities), available_cities
        
        return [p / total for p in probabilities], available_cities

    def select_next_city(self, prob_data):
        probabilities, available_cities = prob_data
        # Use random.choices for weighted selection
        return random.choices(available_cities, weights=probabilities, k=1)[0]

    def calculate_tour_distance(self, tour):
        distance = 0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)] # Return to start
            distance += self.distances[from_city][to_city]
        return distance

    def update_pheromones(self, tours, distances):
        # Evaporation
        self.pheromones *= (1 - self.evaporation_rate)
        
        # Deposit
        for tour, dist in zip(tours, distances):
            deposit = self.q / dist
            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i + 1) % len(tour)]
                self.pheromones[from_city][to_city] += deposit
                self.pheromones[to_city][from_city] += deposit # Symmetric TSP

    def reset(self):
        self.cities = self.generate_cities()
        self.distances = self.calculate_distances()
        self.pheromones = np.ones((self.num_cities, self.num_cities)) * 0.1
        self.best_tour = None
        self.best_distance = float('inf')
        self.iteration = 0
        self.last_improvement_iter = 0

    def run_best_path_demo(self):
        # Returns tours that are all the best tour
        if self.best_tour is None:
            return [], []
        
        all_tours = [self.best_tour[:] for _ in range(self.num_ants)]
        all_distances = [self.best_distance for _ in range(self.num_ants)]
        
        # Do not update pheromones or iteration count
        
        return all_tours, all_distances
