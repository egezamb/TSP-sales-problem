import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Optional
import matplotlib
import os
matplotlib.use('TkAgg')



class Population:
    def __init__(self, solutions: np.ndarray, fitness_scores: np.ndarray):
        self.solutions = solutions
        self.fitness_scores = fitness_scores
        self.size = len(solutions)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        best_idx = np.argmin(self.fitness_scores)
        return self.solutions[best_idx], self.fitness_scores[best_idx]
    
    def get_worst(self) -> Tuple[np.ndarray, float]:
        worst_idx = np.argmax(self.fitness_scores)
        return self.solutions[worst_idx], self.fitness_scores[worst_idx]
    
    def get_median(self) -> float:
        return np.median(self.fitness_scores)

class RealTimePlot:
    def __init__(self, title: str = "Genetic Algorithm Progress", background_image: str = None):
        plt.style.use('seaborn-darkgrid')
        self.fig = plt.figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(111)
        self.background_image = background_image
        self.img_obj = None
        
        if background_image and os.path.exists(background_image):
            self.img = plt.imread(background_image)
            # Show background image initially with default extent
            self.img_obj = self.ax.imshow(self.img, aspect='auto', 
                                        extent=[0, 100, 0, 10000], 
                                        zorder=-1)
        
        self.ax.set_title(title)
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        
        # Add grey name
        self.ax.text(0.95, 0.95, "Ege Zambelli", 
                    transform=self.ax.transAxes,
                    color='white',
                    fontsize=14,
                    fontweight='bold',
                    alpha=0.8,
                    family='DejaVu Sans',
                    horizontalalignment='right')
        
        self.best_fitness_history = []
        self.generation_history = []
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def update(self, generation: int, best_fitness: float, avg_fitness: float,
              best_route: np.ndarray, cities_coords: np.ndarray):
        self.generation_history.append(generation)
        self.best_fitness_history.append(best_fitness)
        
        self.ax.clear()
        
        # Calculate proper extent based on data range
        x_min = 0
        x_max = max(self.generation_history) if self.generation_history else 100
        y_min = min(self.best_fitness_history) if self.best_fitness_history else 0
        y_max = max(self.best_fitness_history) if self.best_fitness_history else 10000
        
        # Add padding to the ranges
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        # First draw the background image with proper extent
        if hasattr(self, 'img'):
            self.img_obj = self.ax.imshow(self.img, aspect='auto', 
                                        extent=[x_min-x_padding, x_max+x_padding, 
                                               y_min-y_padding, y_max+y_padding], 
                                        zorder=-1)
        
        # Then plot the data
        self.ax.set_xlabel("Generation",color='magenta',fontsize=12)
        self.ax.set_ylabel("Fitness",color='magenta',fontsize=12)
        self.ax.plot(self.generation_history, self.best_fitness_history, 'm-', linewidth=2)
        
        # Add fitness score text
        self.ax.text(0.02, 0.98, f'Best: {best_fitness:.0f}', 
                    transform=self.ax.transAxes, 
                    verticalalignment='top',
                    fontsize=10)
        
        # Add grey name
        self.ax.text(0.95, 0.95, "Ege Zambelli", 
                    transform=self.ax.transAxes,
                    color='slategray',
                    fontsize=20,
                    fontweight='bold',
                    family='Arial Black',
                    alpha=0.8,
                    horizontalalignment='right')

        self.ax.text(0.88, 0.88, "00000", 
                    transform=self.ax.transAxes,
                    color='slategray',
                    fontsize=15,
                    fontweight='200',
                    family='Arial Black',
                    alpha=0.8,
                    horizontalalignment='right')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def save(self, filename: str):
        self.fig.savefig(filename, bbox_inches='tight', dpi=100)
    
    def close(self):
        plt.close(self.fig)

def calculate_fitness(solution: np.ndarray, distances: np.ndarray) -> float:
    total_distance = 0
    for i in range(len(solution)):
        total_distance += distances[solution[i-1], solution[i]]
    return total_distance

def create_initial_population(num_individuals: int, num_cities: int,
                            distances: np.ndarray, num_greedy: int = 5) -> Population:
    solutions = np.zeros((num_individuals, num_cities), dtype=int)
    fitness_scores = np.zeros(num_individuals)
    
    for i in range(min(num_greedy, num_individuals)):
        start_city = np.random.randint(num_cities)
        solution = greedy_solution(distances, start_city)
        solutions[i] = solution
        fitness_scores[i] = calculate_fitness(solution, distances)
    
    for i in range(num_greedy, num_individuals):
        solution = np.random.permutation(num_cities)
        solutions[i] = solution
        fitness_scores[i] = calculate_fitness(solution, distances)
    
    return Population(solutions, fitness_scores)

def greedy_solution(distances: np.ndarray, start_city: int) -> np.ndarray:
    num_cities = len(distances)
    unvisited = set(range(num_cities))
    solution = np.zeros(num_cities, dtype=int)
    
    current_city = start_city
    solution[0] = current_city
    unvisited.remove(current_city)
    
    for i in range(1, num_cities):
        next_city = min(unvisited, key=lambda x: distances[current_city, x])
        solution[i] = next_city
        unvisited.remove(next_city)
        current_city = next_city
    
    return solution

def tournament_selection(population: Population, tournament_size: int = 5) -> np.ndarray:
    indices = np.random.choice(population.size, tournament_size, replace=False)
    winner_idx = indices[np.argmin(population.fitness_scores[indices])]
    return population.solutions[winner_idx].copy()

def ordered_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    size = len(parent1)
    points = sorted(np.random.choice(size, 2, replace=False))
    start, end = points
    
    offspring = np.full(size, -1)
    offspring[start:end] = parent1[start:end]
    
    remaining_cities = [x for x in parent2 if x not in offspring[start:end]]
    j = 0
    for i in range(size):
        if offspring[i] == -1:
            offspring[i] = remaining_cities[j]
            j += 1
    
    return offspring

def PMX_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    size = len(parent1)
    points = sorted(np.random.choice(size, 2, replace=False))
    start, end = points
    
    offspring = np.full(size, -1)
    offspring[start:end] = parent1[start:end]
    
    mapping = dict(zip(parent1[start:end], parent2[start:end]))
    
    for i in range(size):
        if i < start or i >= end:
            current = parent2[i]
            while current in offspring[start:end]:
                current = mapping.get(current, current)
            offspring[i] = current
    
    return offspring

def cycle_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    size = len(parent1)
    offspring = np.full(size, -1)
    
    index = 0
    value = parent1[index]
    
    while offspring[index] == -1:
        offspring[index] = value
        value = parent2[index]
        index = np.where(parent1 == value)[0][0]
    
    mask = offspring == -1
    offspring[mask] = parent2[mask]
    
    return offspring

def swap_mutation(solution: np.ndarray, prob: float = 0.1) -> np.ndarray:
    if np.random.random() > prob:
        return solution.copy()
    
    mutated = solution.copy()
    idx1, idx2 = np.random.choice(len(solution), 2, replace=False)
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated

def inversion_mutation(solution: np.ndarray, prob: float = 0.1) -> np.ndarray:
    if np.random.random() > prob:
        return solution.copy()
    
    mutated = solution.copy()
    size = len(mutated)
    p1, p2 = sorted(np.random.choice(size, 2, replace=False))
    mutated[p1:p2] = mutated[p1:p2][::-1]
    return mutated

def scramble_mutation(solution: np.ndarray, prob: float = 0.1) -> np.ndarray:
    if np.random.random() > prob:
        return solution.copy()
    
    mutated = solution.copy()
    size = len(mutated)
    p1, p2 = sorted(np.random.choice(size, 2, replace=False))
    subsequence = mutated[p1:p2]
    np.random.shuffle(subsequence)
    mutated[p1:p2] = subsequence
    return mutated

def create_next_generation(current_pop: Population, distances: np.ndarray,
                         crossover_prob: float = 0.8,
                         mutation_prob: float = 0.1) -> Population:
    pop_size = current_pop.size
    num_cities = len(current_pop.solutions[0])
    new_solutions = np.zeros((pop_size, num_cities), dtype=int)
    new_fitness_scores = np.zeros(pop_size)
    
    elite_size = max(2, pop_size // 4)
    elite_indices = np.argpartition(current_pop.fitness_scores, elite_size)[:elite_size]
    new_solutions[:elite_size] = current_pop.solutions[elite_indices]
    new_fitness_scores[:elite_size] = current_pop.fitness_scores[elite_indices]
    
    for i in range(elite_size, pop_size):
        tournament_size = pop_size // 3
        parent1 = tournament_selection(current_pop, tournament_size=tournament_size)
        parent2 = tournament_selection(current_pop, tournament_size=tournament_size)
        
        if np.random.random() < crossover_prob:
            r = np.random.random()
            if r < 0.4:
                offspring = ordered_crossover(parent1, parent2)
            elif r < 0.7:
                offspring = PMX_crossover(parent1, parent2)
            else:
                offspring = cycle_crossover(parent1, parent2)
        else:
            offspring = parent1.copy()
        
        if np.random.random() < mutation_prob:
            r = np.random.random()
            if r < 0.4:
                offspring = swap_mutation(offspring, mutation_prob)
            elif r < 0.7:
                offspring = inversion_mutation(offspring, mutation_prob)
            else:
                offspring = scramble_mutation(offspring, mutation_prob)
        
        new_solutions[i] = offspring
        new_fitness_scores[i] = calculate_fitness(offspring, distances)
    
    return Population(new_solutions, new_fitness_scores)

def run_genetic_algorithm(initial_pop: Population, distances: np.ndarray,
                        cities_coords: Optional[np.ndarray] = None,
                        num_generations: int = 500,
                        crossover_prob: float = 0.8,
                        mutation_prob: float = 0.1,
                        print_interval: Optional[int] = None,
                        plotter: Optional[RealTimePlot] = None) -> Tuple[Population, List[float]]:
    current_pop = initial_pop
    best_fitness_history = []
    best_overall = float('inf')
    
    for gen in range(num_generations):
        current_pop = create_next_generation(
            current_pop,
            distances,
            crossover_prob,
            mutation_prob
        )
        
        best_solution, best_fitness = current_pop.get_best()
        avg_fitness = np.mean(current_pop.fitness_scores)
        best_fitness_history.append(best_fitness)
        best_overall = min(best_overall, best_fitness)
        
        if plotter is not None and cities_coords is not None:
            plotter.update(
                generation=gen,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                best_route=best_solution,
                cities_coords=cities_coords
            )
        
        if print_interval and (gen + 1) % print_interval == 0:
            print(f"Generation {gen + 1}/{num_generations}")
            print(f"Best Fitness: {best_fitness:.2f}")
            print(f"Average Fitness: {avg_fitness:.2f}")
            print(f"Best Overall: {best_overall:.2f}")
    
    return current_pop, best_fitness_history

def parse_tsp_file(filename: str) -> pd.DataFrame:
    cities = []
    reading_coords = False
    
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            elif line.strip() == "EOF":
                break
            
            if reading_coords:
                parts = line.strip().split()
                if len(parts) == 3:
                    cities.append({
                        'city': int(parts[0]) - 1,
                        'x': float(parts[1]),
                        'y': float(parts[2])
                    })
    
    return pd.DataFrame(cities)

def calculate_distances(cities_df: pd.DataFrame) -> np.ndarray:
    num_cities = len(cities_df)
    distances = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = np.sqrt(
                (cities_df.iloc[i]['x'] - cities_df.iloc[j]['x'])**2 +
                (cities_df.iloc[i]['y'] - cities_df.iloc[j]['y'])**2
            )
            distances[i, j] = distance
            distances[j, i] = distance
    
    return distances

def main():
    cities_df = parse_tsp_file("berlin52.tsp")
    distances = calculate_distances(cities_df)
    cities_coords = cities_df[['x', 'y']].values
    num_cities = len(cities_df)
    
    population_size = 1000
    num_generations = 2000
    crossover_prob = 0.95
    mutation_prob = 0.15
    num_greedy = 100
    
    plotter = RealTimePlot("Berlin52 TSP - Enhanced Genetic Algorithm", background_image="photo.jpg")
    
    initial_pop = create_initial_population(
        num_individuals=population_size,
        num_cities=num_cities,
        distances=distances,
        num_greedy=num_greedy
    )
    
    final_pop, fitness_history = run_genetic_algorithm(
        initial_pop=initial_pop,
        distances=distances,
        cities_coords=cities_coords,
        num_generations=num_generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        print_interval=20,
        plotter=plotter
    )
    
    best_solution, best_fitness = final_pop.get_best()
    print("\nFinal Results:")
    print(f"Best Fitness: {best_fitness:.2f}")
    print(f"Best Route: {best_solution.tolist()}")
    
    plotter.save("final_result.png")
    
    print("\nOptimization complete. Close the plot window to exit.")
    input("Press Enter to close...")
    plotter.close()

if __name__ == "__main__":
    main()
