import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere, Landmark, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils, Color

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self._world = World(batch_dim, device)
        self.num_agents = 2  # M=2
        self.agents = [Agent(name=f"agent_{i}", u_multiplier=1.0, shape=Sphere(0.03)) for i in range(self.num_agents)]
        for agent in self.agents:
            self._world.add_agent(agent)
            agent.is_printing = False  # If the agent is printing
            agent.current_line_segment = None  # Current line segment the agent is assigned to
        
        '''
        # Set a specific goal position far from the start position
        self.goal_pos = torch.tensor([1.0, 1.0], device=device)  # Far corner
        '''
        # Set a random goal position
        self.goal_pos = torch.rand(2) * 2 - 1  # Random position between -1 and 1 for both x and y
        print(f"Initial Goal position: {self.goal_pos.tolist()}")
        # trail
        self.trail_active = False
        self.trail_points = []
        self.last_point = None
        self.trail_distance = 0.5
        
        # -----------------------------------------------------------------------------------------------
        # Initialize the print path
        self.print_path_points = self.create_print_path(num_points=10)
        self.current_segment_index = 0  # Index to track the current segment

        # ***********************************************************************************************
        # Goal landmark
        self.goal_landmark = Landmark(
            name="goal_landmark",
            collide=False,
            shape=Sphere(radius=0.03),
            color=Color.GREEN,
        )
        self._world.add_landmark(self.goal_landmark)
        # Set the goal landmark's position IF there are any points in the print path
        if len(self.print_path_points) > 0:
            self.goal_pos = self.print_path_points[self.current_segment_index]
            self.goal_landmark.set_pos(self.goal_pos.unsqueeze(0), batch_index=0)  # Update the goal landmark's position
        
        #  Assign the initial line segments to the agents
        self.assign_print_paths()
        
        return self._world
    
    def assign_print_paths(self):
        # distribute the initial line segments to the agents
        for i, agent in enumerate(self.agents):
            if i < len(self.print_path_points) - 1:
                # 分配线段(start_point, end_point)
                agent.current_line_segment = (self.print_path_points[i], self.print_path_points[i + 1])
            else:
                agent.current_line_segment = None  # No more line segments to assign
                
    def create_print_path(self, num_points):
        # This function creates the print path, the last point is self.goal_pos
        # This function returns a list of points
        path_points = []
        print("Print Path Points Coordinates:")  # Print the header for the coordinates
        for i in range(num_points - 1):  # Generate one less point than num_points because the last point will be self.goal_pos
            # Random point between -1 and 1 for both x and y
            point = torch.rand(2) * 2 - 1 
            path_points.append(point)
            print(f"Point {i}: {point.tolist()}")  # Print the coordinates of the point

        # Add self.goal_pos as the last point of the path
        path_points.append(self.goal_pos)
        print(f"Point {num_points - 1} (Goal): {self.goal_pos.tolist()}")  # Print the coordinates of the goal pointW

        return path_points

    def visualize_print_path(self):
        for point in self.print_path_points:
            self.pathpoints_landmark = Landmark(
            name="pathpoints_landmark",
            collide=False,
            shape=Sphere(radius=0.03),
            color=Color.RED,
            )
            self._world.add_landmark(self.pathpoints_landmark)
            self.pathpoints_landmark.set_pos(point, batch_index=0)
            
            
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
            
            self.current_segment_index = 0
            if len(self.print_path_points) > 0:
                self.goal_pos = self.print_path_points[self.current_segment_index]
                self.goal_landmark.set_pos(self.goal_pos.unsqueeze(0), batch_index=0)  # Update the goal landmark's position
            else:
                self.goal_pos = torch.tensor([0.0, 0.0], device=self._world.device)  # Default goal position
    
            
        # Reset landmark's position
        self.goal_landmark.set_pos(self.goal_pos.unsqueeze(0), batch_index=env_index)
        
        self.visualize_print_path()
        
    def reward(self, agent: Agent):
        distance_to_goal = torch.norm(agent.state.pos - self.goal_pos)
        return -distance_to_goal.unsqueeze(0)

    def observation(self, agent: Agent):
        expanded_goal_pos = self.goal_pos.unsqueeze(0).expand(agent.state.pos.size(0), -1)
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)

    def update_trail(self, agent: Agent):
        if self.trail_active:
            # Check if the agent has moved far enough from the last point to add a new point
            if torch.norm(agent.state.pos - self.goal_pos) < self.trail_distance:
                # Update the goal position to the next point in the print path
                self.current_segment_index += 1
                if self.current_segment_index < len(self.print_path_points):
                    self.goal_pos = self.print_path_points[self.current_segment_index]
                    self.goal_landmark.set_pos(self.goal_pos.unsqueeze(0), batch_index=0)  # Update the goal landmark's position
                    print(f"Updated goal position to: {self.goal_pos}")  # Print the new goal position
                else:
                    self.stop_trail()  # If the agent has reached the last point, stop the trail


    def start_trail(self):
        self.trail_active = True
        self.last_point = self.world.agents[0].state.pos.clone()
        self.trail_points.append(self.last_point)

    def stop_trail(self):
        self.trail_active = False
        self.update_trail(self.world.agents[0])
            
    def add_landmark(self, position, color=Color.GREEN):
            landmark = Landmark(
                name=f"landmark_{len(self.world.landmarks)}",
                collide=False,
                shape=Sphere(radius=0.03),
                color=color,
            )
            self.world.add_landmark(landmark)
            landmark.set_pos(position.unsqueeze(0), batch_index=0)
            
class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, u_range: float, current_goal_pos) -> torch.Tensor:
        print(f"observation: {observation}")
        pos_agent = observation[:, :2]  # Agent's current position
        goal_pos = current_goal_pos.unsqueeze(0).expand(pos_agent.size(0), -1)  # set the goal position to the current goal position
        
        print(f"goal_pos: {goal_pos[0]}")

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