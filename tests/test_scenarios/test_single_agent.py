import unittest
import torch
from vmas import make_env

class TestSingleAgent(unittest.TestCase):
    def setup_env(self):
        self.env = make_env(
            scenario="single_agent",  # You might need to create this scenario
            num_envs=1,
            device="cpu"
        )
        self.env.seed(0)

    def test_move_to_goal(self):
        self.setup_env()
        obs = self.env.reset()
        goal_pos = torch.tensor([0.5, 0.5])  # Set a random goal position

        for _ in range(100):  # Number of steps
            action = self.compute_action(obs, goal_pos)
            obs, rews, dones, _ = self.env.step([action])

    def compute_action(self, obs, goal_pos):
        # Implement your heuristic policy here
        # This should return the next action for the agent based on its current state and the goal
        pass

if __name__ == "__main__":
    unittest.main()
