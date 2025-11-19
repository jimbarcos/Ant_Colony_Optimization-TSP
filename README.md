# Ant Colony Optimization for Traveling Salesperson Problem

## Overview
This application visualizes the Ant Colony Optimization (ACO) algorithm applied to the Traveling Salesperson Problem (TSP). It demonstrates how a population of agents (ants) can intelligently find optimal or near-optimal paths through a set of cities by communicating via simulated pheromone trails.

## Ant Colony Optimization (ACO)

### What is it?
Ant Colony Optimization is a probabilistic technique for solving computational problems that can be reduced to finding good paths through graphs. It is a member of the swarm intelligence family of algorithms, inspired by the foraging behavior of real ants. In nature, ants deposit pheromones on the ground to mark paths between their colony and food sources. Over time, shorter paths accumulate more pheromones, attracting more ants, which creates a positive feedback loop.

### How it works
1.  **Initialization**: A set of cities is generated, and pheromone levels on all paths connecting them are initialized to a small constant value.
2.  **Tour Construction**: In each iteration, a colony of ants is released. Each ant constructs a complete tour by visiting every city exactly once.
3.  **Probabilistic Decision Making**: When an ant is at a city, it chooses the next city to visit based on a probability rule. This probability depends on two factors:
    *   **Pheromone Trail**: The amount of pheromone on the edge connecting the cities (representing past experience).
    *   **Heuristic Information**: The inverse of the distance to the next city (representing immediate cost; closer cities are more attractive).
4.  **Pheromone Update**:
    *   **Evaporation**: Pheromones on all edges decay by a certain percentage. This simulates the natural evaporation of scent and prevents the algorithm from getting stuck in local optima by allowing bad decisions to be forgotten.
    *   **Deposit**: Ants deposit new pheromones on the edges they traversed. The amount of pheromone deposited is usually proportional to the quality of the tour (shorter total distance results in more pheromone).
5.  **Iteration**: This process repeats. Over time, the pheromone concentration grows on edges that are part of shorter tours, guiding the colony toward the optimal solution.

### Why it works
The algorithm relies on the interaction between **exploration** and **exploitation**.
*   **Exploitation**: Ants prefer paths with high pheromone levels (knowledge from the colony) and short distances (greedy heuristic).
*   **Exploration**: The probabilistic nature of the choice allows ants to occasionally choose less attractive paths, potentially discovering better unknown routes. Pheromone evaporation ensures that the system does not converge instantly to the first decent solution found, keeping the search dynamic.

## Key Parameters and Roles

The behavior of the ACO algorithm is heavily influenced by its configuration parameters. Understanding these roles is key to tuning the simulation.

### Number of Ants
*   **Role**: Represents the size of the search population in each iteration.
*   **Responsibility**: To explore the solution space.
*   **Impact**:
    *   **Higher Count**: Provides a more thorough search of the map in each iteration and results in more stable pheromone updates. However, it increases computational load.
    *   **Lower Count**: Faster simulation speed but may result in erratic behavior or slower convergence because the sample size for updating pheromones is small.

### Alpha (Pheromone Importance)
*   **Role**: A weighting factor that determines how much influence the pheromone trail has on the ant's decision.
*   **Responsibility**: To encourage **exploitation** of past successful paths.
*   **Impact**:
    *   **High Alpha**: Ants become very sensitive to pheromones. They will strongly follow the paths taken by previous successful ants. This can lead to very fast convergence, but often to a suboptimal solution (local optimum).
    *   **Low Alpha**: Ants largely ignore the pheromone trails, making the search more random or purely distance-based.

### Beta (Distance Importance)
*   **Role**: A weighting factor that determines how much influence the distance (visibility) has on the ant's decision.
*   **Responsibility**: To encourage **greedy** choices (choosing the closest city).
*   **Impact**:
    *   **High Beta**: Ants prioritize immediate gratification by choosing the closest available city. This makes the algorithm behave like a greedy heuristic. While often decent, it usually fails to find the global optimum in complex maps.
    *   **Low Beta**: Distance matters less, and ants rely more on pheromones or random exploration.

### Evaporation Rate
*   **Role**: The rate at which pheromones disappear from the edges in each iteration (value between 0 and 1).
*   **Responsibility**: To prevent stagnation and encourage **exploration**.
*   **Impact**:
    *   **High Rate**: Pheromones vanish quickly. The colony has a "short memory." This prevents the algorithm from getting stuck in bad loops early on but makes it harder to converge on a stable solution.
    *   **Low Rate**: Pheromones persist for a long time. The colony has a "long memory." This stabilizes the solution but increases the risk of converging to a suboptimal path that was found early by chance.

## Usage Instructions
1.  **Configuration**: Use the sliders on the right panel to adjust the parameters described above.
2.  **Start/Pause**: Click to begin the simulation or pause it to inspect the current state.
3.  **Reset / Generate New**: Generates a new random set of cities and resets the algorithm.
4.  **Reset Sim (Keep Map)**: Clears the pheromones and ants but keeps the current city layout, allowing you to test different parameters on the same problem instance.
5.  **Toggle Overlays**: Hides or shows the legend and statistics panel for a cleaner view.

The simulation will automatically detect convergence if the best distance does not improve for a set number of iterations, at which point the ants will demonstrate the best path found.

## Requirements / Installation

This project uses Python 3.11.9 and depends on the following Python packages (also listed in `requirements.txt`):

- pygame
- pygame_gui
- numpy
- matplotlib  (required only for the convergence chart)

Install the dependencies using pip (from the project root):

```powershell
python -m pip install -r requirements.txt
```

Running the visualizer:

```powershell
python main.py
```

Troubleshooting:
- If you get a ModuleNotFoundError for `matplotlib`, install it manually: `python -m pip install matplotlib`.
- On some platforms Pygame may require additional system libraries — consult the Pygame docs for your OS.

## Convergence Chart

There is a **Show Convergence Chart** button in the right-hand configuration panel. When you click it, a new Matplotlib window will open and plot the best distance found per iteration from the start of the run to the current iteration (or to convergence). The chart marks the initial value and the most recent value.

Notes:
- This chart reads the internal `distance_history` accumulated during the run; it updates when you click the button (it does not auto-update while a simulation is running).
- Install `matplotlib` if you want the chart functionality (it's optional for visualization and not needed to run the simulation itself).


## How nodes (cities) and edges (paths) are derived

Short explanation of how the program derives node (city) values and path (edge) values:

Cities (nodes)

- Generated coordinates:
  - If grid mode is on (grid_spacing provided), cities are snapped to grid intersections: city positions = (x * grid_spacing + offset, y * grid_spacing + offset).
  - Otherwise they are chosen randomly across the simulation area (inside bounds).
- Displayed value = the city index (0..N-1). If the current visible best tour exists, the first index is shown as "Start/End".
- The city "value" is simply its ID and its (x, y) screen coordinates — there is no extra numeric weight attached to a city.

Distances (edges)

- The program builds a full pairwise distance matrix:
  - distance[i][j] = Euclidean distance between city i and city j = sqrt((xi - xj)^2 + (yi - yj)^2)
- Distances are symmetric: distance[i][j] = distance[j][i].
- Edge distance displayed in the UI is read directly from that distance matrix, rounded to an integer for display.

How the algorithm constructs paths and derives a path value (total tour cost)

- Each ant constructs a tour (an ordering of unique city indices, ending back at the start).
- At each step an ant selects the next unvisited city using weighted probabilities:
  - P(i->j) ∝ (tau_ij)^alpha * (eta_ij)^beta,
    - tau_ij is the pheromone level on edge (i,j).
    - eta_ij is 1 / distance[i][j] (visibility or inverse distance).
  - alpha and beta weight the influence of pheromone vs distance.
- Once the ant completes a full tour, the tour distance is calculated by summing edge distances for each consecutive pair (including the return to the start).
- The best path is the tour with the smallest total distance found by the colony so far. That best distance is saved and shown.

Pheromone update and best path

- After all ants finish, pheromones evaporate by factor (1 - evaporation_rate).
- Each ant deposits pheromone on edges it used: commonly deposit = Q / tour_distance (Q is a constant).
- Best tour selects lowest distance from all tours; the algorithm stores the iteration where the new best was found (last_improvement).

Why these variables matter

- Number of ants: more ants explore more candidate tours — helps exploration but increases processing.
- Alpha: higher alpha increases reliance on pheromone history (exploitation).
- Beta: higher beta favors shorter local moves (exploitation of visibility).
- Evaporation rate: higher rate forgets old trails faster (encourages exploration), lower rate preserves trails (may cause premature convergence).
