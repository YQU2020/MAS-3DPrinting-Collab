import torch
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color
from vmas import render_interactively

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Create a world with a single agent
        world = World(batch_dim, device)
        agent = Agent(name="agent_0", collide=False)
        world.add_agent(agent)

        # Add a single landmark as the goal
        goal = Landmark(name="goal", collide=False, shape=Sphere(radius=0.03), color=Color.GREEN)
        world.add_landmark(goal)

        return world

    def reset_world_at(self, env_index: int = None):
        # Randomly initialize the agent's position
        agent = self.world.agents[0]
        agent.set_pos(
            torch.zeros((1, self.world.dim_p), device=self.world.device).uniform_(-1.0, 1.0),
            batch_index=env_index
        )

        # Randomly initialize the goal's position
        goal = self.world.landmarks[0]
        goal.set_pos(
            torch.zeros((1, self.world.dim_p), device=self.world.device).uniform_(-1.0, 1.0),
            batch_index=env_index
        )

    def reward(self, agent: Agent):
        # Reward based on the distance to the goal
        distance_to_goal = torch.norm(agent.state.pos - self.world.landmarks[0].state.pos)
        return -distance_to_goal.unsqueeze(0)  # Ensure the reward is at least 1D

    def observation(self, agent: Agent):
        # Observation includes agent's position, velocity, and relative position to the goal
        return torch.cat([agent.state.pos, agent.state.vel, self.world.landmarks[0].state.pos - agent.state.pos], dim=-1)

    def done(self):
        # Check if the agent is close enough to the goal to consider the task done
        distance_to_goal = torch.norm(self.world.agents[0].state.pos - self.world.landmarks[0].state.pos)
        return distance_to_goal < 0.05  # Threshold for completion

if __name__ == "__main__":
    render_interactively(__file__, control_one_agents=True, n_agents=1)
