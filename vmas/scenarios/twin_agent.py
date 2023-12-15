import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere, Line, Landmark
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils, Color

class Scenario(BaseScenario):
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim, device)

        # Add two agents
        for i in range(2):
            agent = Agent(name=f"agent_{i}", u_multiplier=1.0, shape=Sphere(0.03), color=Color.GRAY)
            world.add_agent(agent)
            
        # Set a random goal position
        self.nth_goal_pos = [torch.rand(2) * 2 - 1 for _ in range(2)]  # Two random goal positions

        # Create two landmarks as goals
        for i in range(2):
            goal = Landmark(name=f"goal_{i}", collide=False, shape=Sphere(0.05), color=Color.RED)
            world.add_landmark(goal)
            print(f"nth_goal_pos: {self.nth_goal_pos[i]}")
            goal.set_pos(self.nth_goal_pos[i], batch_index=0)

        print(f"nth_goal_pos: {self.nth_goal_pos}")
        self.trail_active = False
        self.trail_points = []
        self.last_points = [None, None]
        self.trail_distance = 0.1

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            random_start_pos = torch.rand((1, self.world.dim_p), device=self.world.device) * 2 - 1
            agent.set_pos(random_start_pos, batch_index=env_index)

        self.trail_points.clear()
        self.last_points = [None, None]
        self.trail_active = False
        for i, goal in enumerate(self.world.landmarks):
            goal.set_pos(self.nth_goal_pos[i], batch_index=env_index)
        
    def observation(self, agent: Agent):
        agent_index = int(agent.name.split('_')[-1])
        self.goal_pos = self.nth_goal_pos[agent_index]
        expanded_goal_pos = self.goal_pos.expand(agent.state.pos.size(0), -1)
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)

    def reward(self, agent: Agent):
        goal_pos = torch.rand((1, self.world.dim_p), device=self.world.device) * 2 - 1
        distance_to_goal = torch.norm(agent.state.pos - goal_pos)
        return -distance_to_goal.unsqueeze(0)

    def update_trail(self):
        if self.trail_active:
            for i, agent in enumerate(self.world.agents):
                if self.last_points[i] is None or torch.norm(agent.state.pos - self.last_points[i]) >= self.trail_distance:
                    if self.last_points[i] is not None:
                        line = Line(start=self.last_points[i].clone(), end=agent.state.pos.clone(), color=Color.GREEN)
                        self.trail_points.append(line)
                    self.last_points[i] = agent.state.pos.clone()

    def start_trail(self):
        self.trail_active = True
        for i, agent in enumerate(self.world.agents):
            self.last_points[i] = agent.state.pos.clone()

    def stop_trail(self):
        self.trail_active = False
        for agent in self.world.agents:
            self.update_trail()

    def add_landmark(self, position):
            landmark = Landmark(
                name=f"landmark_{len(self.world.landmarks)}",
                collide=False,
                shape=Sphere(radius=0.018),
                color=Color.GREEN,
            )
            self.world.add_landmark(landmark)
            landmark.set_pos(position.unsqueeze(0), batch_index=0)

class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        pos_agent = observation[:, :2]  # Agent's current position
        goal_pos = observation[:, -2:]  # Goal position
        
        # Calculate the vector from the agent to the goal
        vector_to_goal = goal_pos - pos_agent
        print(f"pos_agent: {pos_agent[0]}, goal_pos: {goal_pos[0]}, vector_to_goal: {vector_to_goal[0]}")
        # Normalize the direction vector to have a magnitude of 1
        norm_direction_to_goal = vector_to_goal / torch.clamp(torch.norm(vector_to_goal, dim=1, keepdim=True), min=1e-6)

        # Scale the normalized direction vector by the maximum action range (u_range)
        action = norm_direction_to_goal * u_range

        # Ensure the action magnitude is at least a minimum value
        min_action_magnitude = 0.05
        action_magnitude = torch.norm(action, dim=1, keepdim=True)
        action = action / torch.clamp(action_magnitude, min=min_action_magnitude) * u_range
        
        return action

if __name__ == "__main__":
    render_interactively(
        __file__,
        desired_velocity=0.05,
        n_agents=2,
    )
