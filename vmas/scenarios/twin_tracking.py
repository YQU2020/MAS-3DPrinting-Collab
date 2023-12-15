import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere,Landmark
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils, Color

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim, device)
        agent = Agent(name="agent_0", u_multiplier=1.0, shape=Sphere(0.03))
        world.add_agent(agent)
        '''
        # Set a specific goal position far from the start position
        self.goal_pos = torch.tensor([1.0, 1.0], device=device)  # Far corner
        '''
        # Set a random goal position
        self.goal_pos = torch.rand(2) * 2 - 1  # Random position between -1 and 1 for both x and y
        
        # trail
        self.trail_active = False
        self.trail_points = []
        self.last_point = None
        self.trail_distance = 0.5
        
        #**********************************************************************************************************************
        self.goal_landmark = Landmark(
            name="goal_landmark",
            collide=False,
            shape=Sphere(radius=0.03),
            color=Color.GREEN,
        )
        world.add_landmark(self.goal_landmark)
        self.goal_landmark.set_pos(self.goal_pos.unsqueeze(0), batch_index=0)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            '''
            # Set a specific start position far from the goal position
            start_pos = torch.tensor([[-1.0, -1.0]], device=self.world.device)  # Example: Opposite corner
            agent.set_pos(start_pos, batch_index=env_index)
            '''
            # Set a random start position for the agent
            random_start_pos = torch.rand((1, self.world.dim_p), device=self.world.device) * 2 - 1
            agent.set_pos(random_start_pos, batch_index=env_index)
            
            self.trail_points.clear()
            self.last_point = None
            self.trail_active = False
        # Reset landmark's position
        self.goal_landmark.set_pos(self.goal_pos.unsqueeze(0), batch_index=env_index)
        
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
    '''
    # Abandoned code, add static agent (not landmark) to represent the trail
    #**********************************************************************************************************************
    # Add a static agent to the world
    def add_static_agent(self, position):
            static_agent = Agent(name=f"static_{len(self.world.agents)}", u_multiplier=0.0, shape=Sphere(0.03), collide=False) #collide=False
            self.world.add_agent(static_agent)  # Add the agent to the world first

            # Extract the relevant slice from the position tensor for a single agent
            single_agent_position = position[0]
            static_agent.set_pos(single_agent_position, batch_index=0)
    '''        
            
    def add_landmark(self, position):
            landmark = Landmark(
                name=f"landmark_{len(self.world.landmarks)}",
                collide=False,
                shape=Sphere(radius=0.03),
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
        print(f"vector_to_goal: {vector_to_goal[0]}")
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
        n_agents=1,
        
    )
