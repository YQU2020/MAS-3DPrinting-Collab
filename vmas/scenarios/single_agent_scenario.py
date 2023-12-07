import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere, Landmark
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim, device)
        agent = Agent(name="agent_0", u_multiplier=1.0, shape=Sphere(0.03))
        world.add_agent(agent)

        # Set a random goal position
        self.goal_pos = torch.rand(2) * 2 - 1  # Random position between -1 and 1 for both x and y
        print(f"Goal position: {self.goal_pos}")

        # Add a landmark to visually represent the goal position
        goal_landmark = Landmark(
            name="goal_landmark",
            collide=False, 
            shape=Sphere(radius=0.03),
            color=Color.GREEN,
        )
        world.add_landmark(goal_landmark)
        goal_landmark.set_pos(self.goal_pos.unsqueeze(0), batch_index=0)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros((1, self.world.dim_p), device=self.world.device).uniform_(-1.0, 1.0),
                batch_index=env_index
            )

    def reward(self, agent: Agent):
        distance_to_goal = torch.norm(agent.state.pos - self.goal_pos)
        return -distance_to_goal.unsqueeze(0)

    def observation(self, agent: Agent):
        expanded_goal_pos = self.goal_pos.unsqueeze(0).expand(agent.state.pos.size(0), -1)
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)

class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        pos_agent = observation[:, :2]
        goal_pos = observation[:, -2:]
        action = torch.clamp(goal_pos - pos_agent, min=-u_range, max=u_range)
        return action

if __name__ == "__main__":
    render_interactively(
        __file__,
        desired_velocity=0.05,
        n_agents=1,
    )
