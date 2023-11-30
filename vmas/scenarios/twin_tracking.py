import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils

# Extends the Agent class to support trail laying functionality
class TrailLayingAgent(Agent):
    def __init__(self, name, u_multiplier, shape):
        super().__init__(name, u_multiplier, shape)
        # Initialize trail-related attributes
        self.trail_active = False
        self.trail_points = []
        self.last_point = None
        self.trail_distance = 0.1  # Default distance between trail points

    # Starts laying the trail
    def start_trail(self):
        self.trail_active = True
        self.last_point = self.state.pos.clone()
        self.trail_points.append(self.last_point)

    # Stops laying the trail
    def stop_trail(self):
        self.trail_active = False
        self._add_trail_point(self.state.pos)

    # Updates the trail as the agent moves
    def update_trail(self):
        if self.trail_active and (self.last_point is None or torch.norm(self.state.pos - self.last_point) >= self.trail_distance):
            self._add_trail_point(self.state.pos)

    # Adds a new point to the trail
    def _add_trail_point(self, new_point):
        if self.last_point is not None:
            line = (self.last_point.clone(), new_point.clone())
            self.trail_points.append(line)
        self.last_point = new_point.clone()

# Scenario class for the simulation environment
class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim, device)
        # Create a TrailLayingAgent
        agent = TrailLayingAgent(name="agent_0", u_multiplier=1.0, shape=Sphere(0.03))
        world.add_agent(agent)

        # Set a goal position for the agent
        self.goal_pos = torch.tensor([0.5, 0.5], device=device)
        return world

    # Resets the world and agents for a new simulation
    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            # Reset agent position and trail
            agent.set_pos(
                torch.zeros((1, self.world.dim_p), device=self.world.device).uniform_(-1.0, 1.0),
                batch_index=env_index
            )
            agent.trail_points.clear()
            agent.last_point = None
            agent.trail_active = False

    # Calculates the reward for the agent based on its distance to the goal
    def reward(self, agent: Agent):
        distance_to_goal = torch.norm(agent.state.pos - self.goal_pos)
        return -distance_to_goal.unsqueeze(0)

    # Generates the observation for the agent
    def observation(self, agent: Agent):
        expanded_goal_pos = self.goal_pos.unsqueeze(0).expand(agent.state.pos.size(0), -1)
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)

# Simple policy for agent's actions
class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        pos_agent = observation[:, :2]
        goal_pos = observation[:, -2:]
        action = torch.clamp(goal_pos - pos_agent, min=-u_range, max=u_range)
        return action

# Main execution block
if __name__ == "__main__":
    # Render the simulation interactively
    render_interactively(
        __file__,
        desired_velocity=0.05,
        n_agents=1,
    )


'''
Current problem:
Traceback (most recent call last):
  File "twin_tracking.py", line 85, in <module>
    render_interactively(
  File "/home/willqu/VectorizedMultiAgentSimulator/vmas/interactive_rendering.py", line 316, in render_interactively
    InteractiveEnv(
  File "/home/willqu/VectorizedMultiAgentSimulator/vmas/interactive_rendering.py", line 76, in __init__
    self.env.render()
  File "/home/willqu/VectorizedMultiAgentSimulator/vmas/simulator/environment/gym.py", line 79, in render
    return self._env.render(
  File "/home/willqu/VectorizedMultiAgentSimulator/vmas/simulator/environment/environment.py", line 567, in render
    [agent.shape.circumscribed_radius() for agent in self.world.agents]
  File "/home/willqu/VectorizedMultiAgentSimulator/vmas/simulator/environment/environment.py", line 567, in <listcomp>
    [agent.shape.circumscribed_radius() for agent in self.world.agents]
AttributeError: 'float' object has no attribute 'circumscribed_radius'
'''