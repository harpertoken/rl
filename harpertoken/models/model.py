import numpy as np
import gymnasium as gym
from huggingface_hub import hf_hub_download
from huggingface_hub import ModelHubMixin
import os
import json
from pathlib import Path
from typing import Optional, Dict, Union


class CMAESAgent(ModelHubMixin):
    def __init__(self, env_name="CartPole-v1"):
        super().__init__()
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_type = "discrete"
            self.num_actions = self.env.action_space.n
            self.action_space = self.num_actions
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.action_type = "continuous"
            self.action_dim = self.env.action_space.shape[0]
            self.action_bounds = (self.env.action_space.low, self.env.action_space.high)
            self.action_space = self.action_dim
        else:
            raise ValueError("Unsupported action space type")
        self.weights = None

    def get_action(self, state):
        if self.action_type == "discrete":
            shape = (self.observation_space, self.num_actions)
            weights_matrix = self.weights.reshape(shape)
            action_scores = np.dot(state, weights_matrix)
            return int(np.argmax(action_scores))
        elif self.action_type == "continuous":
            shape = (self.observation_space, self.action_dim)
            weights_matrix = self.weights.reshape(shape)
            action = np.dot(state, weights_matrix)
            return np.clip(action, self.action_bounds[0], self.action_bounds[1])

    def evaluate(self, num_episodes=100, render=False):
        total_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.get_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        return np.mean(total_rewards), np.std(total_rewards)

    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model weights"""
        os.makedirs(save_directory, exist_ok=True)
        weights_path = os.path.join(save_directory, "weights.npy")
        metadata_path = os.path.join(save_directory, "metadata.json")
        np.save(weights_path, self.weights)
        metadata = {
            "observation_space": int(self.observation_space),
            "action_type": self.action_type,
            "env_name": self.env_name,
        }
        if self.action_type == "discrete":
            metadata["num_actions"] = int(self.num_actions)
        elif self.action_type == "continuous":
            metadata["action_dim"] = int(self.action_dim)
            metadata["action_bounds"] = self.action_bounds
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> "CMAESAgent":
        """Load model weights"""
        # Initialize the model
        model = cls(**model_kwargs)

        if os.path.isdir(model_id):
            metadata_path = os.path.join(model_id, "metadata.json")
            weights_path = os.path.join(model_id, "weights.npy")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                model.weights = np.load(weights_path)
                model.observation_space = metadata["observation_space"]
                model.action_type = metadata["action_type"]
                model.env_name = metadata["env_name"]
                model.env = gym.make(model.env_name)
                if model.action_type == "discrete":
                    model.num_actions = metadata["num_actions"]
                    model.action_space = model.num_actions
                elif model.action_type == "continuous":
                    model.action_dim = metadata["action_dim"]
                    model.action_bounds = metadata["action_bounds"]
                    model.action_space = model.action_dim
            else:
                raise ValueError(
                    "Old insecure model format detected. Pickle loading disabled. "
                    "Convert to new format: weights.npy and metadata.json."
                )
        else:
            # HF Hub
            metadata_path = hf_hub_download(
                repo_id=model_id,
                filename="metadata.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
            )
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename="weights.npy",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
            )
            with open(metadata_path) as f:
                metadata = json.load(f)
            model.weights = np.load(weights_path)
            model.observation_space = metadata["observation_space"]
            model.action_type = metadata["action_type"]
            model.env_name = metadata["env_name"]
            model.env = gym.make(model.env_name)
            if model.action_type == "discrete":
                model.num_actions = metadata["num_actions"]
                model.action_space = model.num_actions
            elif model.action_type == "continuous":
                model.action_dim = metadata["action_dim"]
                model.action_bounds = metadata["action_bounds"]
                model.action_space = model.action_dim

        return model
