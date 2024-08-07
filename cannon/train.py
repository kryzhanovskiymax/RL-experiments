from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class TrainSettings():
    dispersion: float = 1.0
    batch_size: int = 64
    learning_rate: float = 0.001
    start_speed: float = 10
    gamma: float = 0.001
    beta: float = 10.0


def sample_noise(batch_size, dispersion):
    return np.random.randn(batch_size) * dispersion


class EvolutionaryTrainer():
    def __init__(self,
                 env=None,
                 settings: TrainSettings = TrainSettings(),
                 writer=None,
                 smoothing=100):
        if env is None:
            raise ValueError("Environment not settled")
        self.settings = settings
        self.env = env
        self.writer = writer
        self.smoothing = smoothing

    def _base_init(self):
        pass
    
    def _calc_returns(self, returns):
        return self.settings.beta * np.exp(-(np.power(returns, 2)) * self.settings.gamma)

    def train(self, episodes=100, logging=False):
        self.env.reset()
        constant = self.settings.learning_rate / self.settings.dispersion / \
                   self.settings.batch_size
        updated_speed = self.settings.start_speed

        deltas = []
        log_speeds = []
        
        for episode in tqdm(range(episodes)):
            self.env.reset()
            returns = []
            noises = sample_noise(self.settings.batch_size, self.settings.dispersion)
            speeds = np.full(self.settings.batch_size, updated_speed) + \
                     noises
            for speed in speeds:
                _, reward, _, info = self.env.step(speed)
                returns.append(info['delta'])
                if logging:
                    print(f"Episode {episode}/{episodes}: Speed {speed}, Delta {info['delta']}, Reward {info['delta']}")

            returns = np.array(returns)
            avg_delta = np.mean(np.abs(returns))
            deltas.append(avg_delta)
            
            returns = self._calc_returns(returns)
            s = np.sum(returns * noises)
            if logging:
                print(f"Delta: {s}, Normalized: {constant * s}")
            updated_speed = self.settings.start_speed + constant * s
            log_speeds.append(updated_speed)
            if self.writer is not None:
                if episode >= self.smoothing: 
                    ls = np.mean(log_speeds[-self.smoothing:])
                    ld = np.mean(deltas[-self.smoothing:])
                    self.writer.add_scalar("Cannon/speed", ls, episode)
                    self.writer.add_scalar("Cannon/delta", ld, episode)
        self.settings.start_speed = updated_speed
        return updated_speed

@dataclass
class GeneticTrainSettings:
    population_size: int = 100
    mutation_rate: float = 0.1
    gamma: float = 0.01
    beta: float = 1.0
    
    

class GeneticTrainer():
    def __init__(self,
                 env,
                 settings: GeneticTrainSettings = GeneticTrainSettings(),
                 writer = None):
        self.settings = settings
        self.env = env
        self.writer = writer

    def _fitness_func(self, deltas: np.array):
        gamma = self.settings.gamma
        beta = self.settings.beta
        return beta * np.exp(-(np.power(deltas, 2) * gamma))

    def _init_generation(self, n):
        return np.random.uniform(low=0, high=100, size=n)  # Random initial speeds between 0 and 100

    def _evaluate_population(self, population):
        fitness_scores = []
        for speed in population:
            self.env.reset()
            _, reward, _, _ = self.env.step(speed)
            fitness_scores.append(-reward)  # Fitness is the negative reward (closer to target is better)
        return np.array(fitness_scores)

    def _select_parents(self, population, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)
        return population[sorted_indices][:self.settings.population_size // 2]

    def _crossover(self, parents):
        new_population = []
        for _ in range(self.settings.population_size):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = (parent1 + parent2) / 2  # Simple average crossover
            new_population.append(child)
        return np.array(new_population)

    def _mutate(self, population):
        for i in range(len(population)):
            if np.random.rand() < self.settings.mutation_rate:
                population[i] += np.random.uniform(-5, 5)  # Add random noise
        return population

    def train(self, num_generations: int = 1000, logging: bool = False):
        population = self._init_generation(self.settings.population_size)

        for generation in tqdm(range(num_generations)):
            fitness_scores = self._evaluate_population(population)
            parents = self._select_parents(population, fitness_scores)
            population = self._crossover(parents)
            population = self._mutate(population)

            best_fitness = np.max(fitness_scores)
            best_speed = population[np.argmax(fitness_scores)]
            if logging:
                print(f"Generation {generation}: Best Speed = {best_speed}, Best Fitness = {best_fitness}")
            if self.writer is not None:
                self.writer.add_scalar('Genetic/Speed',
                                  best_speed,
                                  generation)
                self.writer.add_scalar('Genetic/Fitness Function',
                                  best_fitness,
                                  generation)
                

        return best_speed


