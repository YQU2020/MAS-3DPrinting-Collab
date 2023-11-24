import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim, device)
        agent = Agent(name="agent_0", u_multiplier=1.0, shape=Sphere(0.03))
        world.add_agent(agent)

        self.goal_pos = torch.tensor([0.5, 0.5], device=device)
        self.trail_active = False
        self.trail_points = []
        self.last_point = None
        self.trail_distance = 0.1  # Default distance, can be changed

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros((1, self.world.dim_p), device=self.world.device).uniform_(-1.0, 1.0),
                batch_index=env_index
            )
            self.trail_points.clear()
            self.last_point = None
            self.trail_active = False

    def reward(self, agent: Agent):
        distance_to_goal = torch.norm(agent.state.pos - self.goal_pos)
        return -distance_to_goal.unsqueeze(0)

    def observation(self, agent: Agent):
        expanded_goal_pos = self.goal_pos.unsqueeze(0).expand(agent.state.pos.size(0), -1)
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)

    def update_trail(self, agent: Agent):
        if self.trail_active and (self.last_point is None or torch.norm(agent.state.pos - self.last_point) >= self.trail_distance):
            if self.last_point is not None:
                line = (self.last_point.clone(), agent.state.pos.clone())
                self.trail_points.append(line)
            self.last_point = agent.state.pos.clone()

    def start_trail(self):
        self.trail_active = True
        self.last_point = self.world.agents[0].state.pos.clone()
        self.trail_points.append(self.last_point)

    def stop_trail(self):
        self.trail_active = False
        self.update_trail(self.world.agents[0])

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
