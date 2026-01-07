import cma
import numpy as np
from ..models.model import CMAESAgent


def train_cmaes(num_generations=100, population_size=16):
    agent = CMAESAgent()

    # Initialize CMA-ES optimizer
    num_params = agent.observation_space * agent.action_space
    es = cma.CMAEvolutionStrategy(num_params * [0], 0.5, {"popsize": population_size})

    def fitness_function(weights):
        agent.weights = weights
        mean_reward, _ = agent.evaluate(num_episodes=5)
        return -mean_reward  # CMA-ES minimizes, so we negate the reward

    # Training loop
    for generation in range(num_generations):
        solutions = es.ask()  # Get new candidate solutions
        fitnesses = [fitness_function(x) for x in solutions]
        es.tell(solutions, fitnesses)  # Update CMA-ES parameters

        best_fitness = -min(fitnesses)  # Convert back to positive reward
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # Save best solution
        best_idx = np.argmin(fitnesses)
        agent.weights = solutions[best_idx]

    return agent


if __name__ == "__main__":
    # Train the agent
    agent = train_cmaes()

    # Save the model locally
    agent.save_pretrained("cartpole_cmaes")

    # Evaluate the final model
    mean_reward, std_reward = agent.evaluate(num_episodes=100)
    print(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
