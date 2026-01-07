from huggingface_hub import HfApi, create_repo, metadata_update
import os
from ..models.model import CMAESAgent
import numpy as np
import json
from datetime import datetime


def push_to_hub():
    # Initialize API
    api = HfApi()

    # Repository details
    repo_id = "harpertoken/harpertoken-cartpole"

    try:
        # Create or get repository
        create_repo(repo_id, exist_ok=True)

        # Load and prepare model
        with open("metadata.json") as f:
            metadata = json.load(f)
        weights = np.load("weights.npy")

        # Initialize agent with the environment name
        agent = CMAESAgent(env_name="CartPole-v1")
        agent.weights = weights  # Directly assign the weights array

        # Evaluate model for metadata
        mean_reward, std_reward = agent.evaluate(num_episodes=10)

        # Convert numpy values to Python scalars
        mean_reward = float(mean_reward)
        std_reward = float(std_reward)

        # Save model locally
        os.makedirs("temp_model", exist_ok=True)
        np.save("temp_model/weights.npy", weights)
        with open("temp_model/metadata.json", "w") as f:
            json.dump(metadata, f)

        # Push model files
        api.upload_file(
            path_or_fileobj="temp_model/weights.npy",
            path_in_repo="weights.npy",
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj="temp_model/metadata.json",
            path_in_repo="metadata.json",
            repo_id=repo_id,
        )

        # Create model card with metadata
        model_card = f"""---
language: en
tags:
- reinforcement-learning
- gymnasium
- cartpole
- cma-es
license: mit
library_name: custom
datasets:
- gymnasium/CartPole-v1
metrics:
- mean_reward
model-index:
- name: CartPole-CMA-ES
  results:
  - task: 
      type: reinforcement-learning
      name: CartPole-v1
    dataset:
      name: gymnasium/CartPole-v1
      type: gymnasium
    metrics:
      - type: mean_reward
        value: {mean_reward:.2f}
        name: Mean Reward
---

# CartPole CMA-ES Agent

This model implements a CartPole agent trained using the CMA-ES
(Covariance Matrix Adaptation Evolution Strategy) algorithm.

## Model Description

- **Model Type:** Reinforcement Learning Policy
- **Training Algorithm:** CMA-ES
- **Environment:** CartPole-v1 (Gymnasium)
- **Input:** State vector (4 dimensions)
- **Output:** Discrete action (2 dimensions)
- **Last Updated:** {datetime.now().strftime("%Y-%m-%d")}

## Performance Metrics

- **Mean Reward:** {mean_reward:.2f} ± {std_reward:.2f}
- **Evaluation Episodes:** 10
- **Training Episodes:** 100 generations with population size 16

## Usage

```python
from harpertoken.models.model import CMAESAgent

# Load the model
agent = CMAESAgent.from_pretrained("harpertoken/harpertoken-cartpole")

# Evaluate
mean_reward, std_reward = agent.evaluate(num_episodes=5)
print(f"Mean reward: {{mean_reward:.2f}} ± {{std_reward:.2f}}")
```

## Training Details

The agent was trained using the CMA-ES algorithm with the following specifications:
- Linear policy mapping states directly to actions
- No neural networks involved - pure evolutionary optimization
- Population size: 16
- Generations: 100

## Limitations and Biases

- The model is specifically trained for the CartPole-v1 environment
- Performance may vary due to the stochastic nature of the environment
- The linear policy might not generalize well to more complex tasks

## Citation

```bibtex
@misc{{cartpole-cmaes,
  author = {{Niladri Das}},
  title = {{CartPole CMA-ES Agent}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  journal = {{Hugging Face Hub}},
   howpublished = {{\\url{{https://huggingface.co/harpertoken/harpertoken-cartpole}}}}
}}
```"""

        # Write and upload README/model card
        with open("README.md", "w") as f:
            f.write(model_card)

        api.upload_file(
            path_or_fileobj="README.md", path_in_repo="README.md", repo_id=repo_id
        )

        # Update repository metadata using the correct function
        metadata_update(
            repo_id=repo_id,
            metadata={
                "tags": ["reinforcement-learning", "gymnasium", "cartpole", "cma-es"],
                "library_name": "custom",
                "language": "en",
                "license": "mit",
                "model-index": [
                    {
                        "name": "CartPole-CMA-ES",
                        "results": [
                            {
                                "task": {
                                    "type": "reinforcement-learning",
                                    "name": "CartPole-v1",
                                },
                                "dataset": {
                                    "name": "gymnasium/CartPole-v1",
                                    "type": "gymnasium",
                                },
                                "metrics": [
                                    {
                                        "type": "mean_reward",
                                        "value": mean_reward,
                                        "name": "Mean Reward",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        )

        print(f"Successfully uploaded model and metadata to {repo_id}")

    except Exception as e:
        print(f"Error during model upload or metadata update: {e}")
        print("Current working directory:", os.getcwd())
        print(
            "Available files in cartpole_cmaes:",
            os.listdir("cartpole_cmaes")
            if os.path.exists("cartpole_cmaes")
            else "Directory not found",
        )


if __name__ == "__main__":
    push_to_hub()
