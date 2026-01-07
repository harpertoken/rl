from harpertoken.models.model import CMAESAgent
import numpy as np


def test_evaluate(capsys):
    agent = CMAESAgent()
    agent.weights = np.random.rand(8)
    mean_reward, std_reward = agent.evaluate(num_episodes=1)
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)
