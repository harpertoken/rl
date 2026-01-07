import gymnasium as gym
import numpy as np
from cma import CMAEvolutionStrategy
import matplotlib.pyplot as plt
import os
import json


class CMAESTrainer:
    def __init__(self, env_name="CartPole-v1"):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_type = "discrete"
            self.num_actions = self.env.action_space.n
            self.num_params = self.observation_space * self.num_actions
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.action_type = "continuous"
            self.action_dim = self.env.action_space.shape[0]
            self.action_bounds = (self.env.action_space.low, self.env.action_space.high)
            self.num_params = self.observation_space * self.action_dim
        else:
            raise ValueError("Unsupported action space type")

        # CMA-ES parameters
        self.sigma = 0.5  # Initial step size
        self.population_size = 4  # Population size (reduced for quick test)
        self.max_iterations = 5  # Maximum number of iterations (reduced for quick test)

        # Initialize CMA-ES optimizer
        self.es = CMAEvolutionStrategy(
            x0=np.zeros(self.num_params),  # Initial solution
            sigma0=self.sigma,
            inopts={"popsize": self.population_size},
        )

        self.training_history = {
            "best_fitness": [],
            "mean_fitness": [],
            "generation": [],
        }

    def evaluate_policy(self, weights, num_episodes=5):
        """Evaluate a policy (weights) over several episodes."""
        total_rewards = []
        if self.action_type == "discrete":
            weights_matrix = weights.reshape(self.observation_space, self.num_actions)
        elif self.action_type == "continuous":
            weights_matrix = weights.reshape(self.observation_space, self.action_dim)

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Get action using the policy
                if self.action_type == "discrete":
                    action_scores = np.dot(obs, weights_matrix)
                    action = int(np.argmax(action_scores))
                elif self.action_type == "continuous":
                    action = np.dot(obs, weights_matrix)
                    low, high = self.action_bounds
                    action = np.clip(action, low, high)

                # Take step in environment
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def train(self):
        """Train the model using CMA-ES."""
        best_fitness = float("-inf")
        best_weights = None
        iteration = 0

        while not self.es.stop() and iteration < self.max_iterations:
            solutions = self.es.ask()
            fitnesses = []

            for weights in solutions:
                reward = self.evaluate_policy(weights)  # Remove the [0] indexing
                fitnesses.append(reward)

            self.es.tell(solutions, [-f for f in fitnesses])  # CMA-ES minimizes

            generation_best = max(fitnesses)
            generation_mean = sum(fitnesses) / len(fitnesses)

            # Store training metrics
            self.training_history["best_fitness"].append(generation_best)
            self.training_history["mean_fitness"].append(generation_mean)
            self.training_history["generation"].append(iteration)

            if generation_best > best_fitness:
                best_fitness = generation_best
                best_weights = solutions[fitnesses.index(generation_best)]

            print(f"G{iteration}: B{generation_best:.2f}, M{generation_mean:.2f}")
            iteration += 1

        # Create and save the convergence plot
        self.plot_training_history()

        # Save training history
        np.save("training_history.npy", self.training_history)

        np.save("weights.npy", best_weights)
        metadata = {
            "fitness": best_fitness,
            "observation_space": self.observation_space,
            "action_type": self.action_type,
        }
        if self.action_type == "discrete":
            metadata["num_actions"] = self.num_actions
        elif self.action_type == "continuous":
            metadata["action_dim"] = self.action_dim
            metadata["action_bounds"] = self.action_bounds
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)
        print("Model saved to weights.npy and metadata.json")

        return best_weights, best_fitness

    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.training_history["generation"],
            self.training_history["best_fitness"],
            label="Best Fitness",
            color="blue",
        )
        plt.plot(
            self.training_history["generation"],
            self.training_history["mean_fitness"],
            label="Mean Fitness",
            color="orange",
            alpha=0.7,
        )

        plt.xlabel("Generation")
        plt.ylabel("Fitness (Episode Return)")
        plt.title("CMA-ES Training Convergence on CartPole-v1")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/training_convergence.png", dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    trainer = CMAESTrainer()
    best_weights, best_fitness = trainer.train()

    print("\nTesting final model...")
    from harpertoken.models.model import CMAESAgent

    agent = CMAESAgent("CartPole-v1")
    with open("metadata.json") as f:
        metadata = json.load(f)
    agent.weights = np.load("weights.npy")
    agent.action_type = metadata["action_type"]
    if agent.action_type == "discrete":
        agent.num_actions = metadata["num_actions"]
    elif agent.action_type == "continuous":
        agent.action_dim = metadata["action_dim"]
        agent.action_bounds = metadata["action_bounds"]
    mean_reward, std_reward = agent.evaluate(num_episodes=5)
    print(f"Mean reward: {mean_reward}, Std: {std_reward}")
