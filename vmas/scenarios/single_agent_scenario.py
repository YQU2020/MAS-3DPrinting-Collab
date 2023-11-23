import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils

# To run the scenario interactively, I need to import the make_env function
from vmas import make_env
import pygame

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Create a world with a single agent
        world = World(batch_dim, device)
        agent = Agent(name="agent_0", u_multiplier=1.0, shape=Sphere(0.03))
        world.add_agent(agent)

        # Set the goal position
        self.goal_pos = torch.tensor([0.5, 0.5], device=device)
        return world

    def reset_world_at(self, env_index: int = None):
        # Randomly initialize the agent's position
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros((1, self.world.dim_p), device=self.world.device).uniform_(-1.0, 1.0),
                batch_index=env_index
            )

    def reward(self, agent: Agent):
        # Reward based on the distance to the goal
        distance_to_goal = torch.norm(agent.state.pos - self.goal_pos)
        return -distance_to_goal

    def observation(self, agent: Agent):
        # Expand the goal position to match the batch dimension
        expanded_goal_pos = self.goal_pos.unsqueeze(0).expand(agent.state.pos.size(0), -1)

        # Observation includes agent's position and velocity, and goal position, since it is all the information the agent has
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)


class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        
        # Extract agent position and goal position from the observation
        pos_agent = observation[:, :2]
        goal_pos = observation[:, -2:]

        # Here I compute action as direction vector towards the goal, scaled by u_range
        action = torch.clamp(goal_pos - pos_agent, min=-u_range, max=u_range)
        return action

"""
# trying to render the scenario interactively
if __name__ == "__main__":
    # Initialize the environment 
    env = make_env(scenario=Scenario, num_envs=1, device="cpu")
    obs = env.reset()

    # Set up a basic loop to render the environment
    running = True
    while running:
        # Render the current state of the environment
        render_interactively(env)

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
"""

if __name__ == "__main__":
    render_interactively(
        __file__,
        desired_velocity=0.05,
        n_agents=1,
    )
