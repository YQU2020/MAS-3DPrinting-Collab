import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere, Landmark
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils, Color

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim, device)
        agent = Agent(name="agent_0", u_multiplier=1.0, shape=Sphere(0.03))
        world.add_agent(agent)

        # Add a landmark to visually represent the target position
        self.goalpoint = Landmark(
            name="target_landmark",
            collide=False, 
            shape=Sphere(radius=0.03),
            color=Color.GREEN,
        )
        world.add_landmark(self.goalpoint)
        self.goal_pos = torch.rand(2) * 2 - 1  # Random position between -1 and 1 for both x and y
        self.goal_pos = torch.tensor([1.0, 1.0], device=device)  # Far corner
        
        # Issue: The landmark appears at the center of the screen despite being set to a random position.
        # Potential causes could be incorrect initialization, rendering logic, or coordinate system issues.
        # Removed the landmark position initialization from the reset_world_at function to see if it would fix the issue.

        # initialize the landmark's position
        self.goalpoint.set_pos(
            self.goal_pos,
            batch_index=None
        )
        # Now the landmark always appears at the center of the screen, regardless of the position set above.
        # But the agent will still move towards the goal position, so the issue is only with the rendering.
        return world

    def reset_world_at(self, env_index: int = None):
        # Randomly initialize the agent's position
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(-1.0, 1.0),
                batch_index=env_index
            )


        

    def reward(self, agent: Agent):
        # Reward based on the distance to the landmark
        distance_to_landmark = torch.norm(agent.state.pos - self.goalpoint.state.pos)
        return -distance_to_landmark.unsqueeze(0)  # Negative reward for distance

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.world.landmarks[0].state.pos - agent.state.pos,
            ],
            dim=-1,
        )

class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        # Extract agent position and landmark position from the observation
        pos_agent = observation[:, :2]
        landmark_pos = observation[:, -2:]
        action = torch.clamp(landmark_pos - pos_agent, min=-u_range, max=u_range)
        return action

if __name__ == "__main__":
    render_interactively(
        __file__,
        desired_velocity=0.05,
        n_agents=1,
    )
