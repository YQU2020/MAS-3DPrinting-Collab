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
            agent.is_printing = False  # Whether the agent is printing
            agent.current_line_segment = None  # The current line segment the agent is assigned to
        
        # Set a random goal position
        self.goal_pos = torch.rand(2) * 2 - 1  # Random position between -1 and 1 for both x and y
        print(f"Initial Goal position: {self.goal_pos.tolist()}")
        # trail
        self.trail_active = False
        self.trail_points = []
        self.last_point = None
        self.trail_distance = 0.1
        
        # -----------------------------------------------------------------------------------------------
        # Initialize the print path
        self.print_path_points = self.create_print_path(num_points=10)
        self.current_segment_index = 0  # Index to track the current segment
        
        # Initialize the list of unprinted line segments
        self.unprinted_segments = [(self.print_path_points[i], self.print_path_points[i + 1]) for i in range(len(self.print_path_points) - 1)]
        
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

        # Assign the initial line segments to the agents
        #self.assign_print_paths()
        
        return self._world
    
    def assign_print_paths(self):
        # Reset the current line segment of all agents
        for agent in self.agents:
            agent.current_line_segment = None

        segment_idx = 0  # Current line segment index
        for i, agent in enumerate(self.agents):
            if segment_idx < len(self.unprinted_segments):
                # Assign line segment to agent
                agent.current_line_segment = self.unprinted_segments[segment_idx]
                segment_idx += 1  # Move to the next segment
                # For even-numbered agents, skip a segment (to maintain spacing)
                if i % 2 == 0 and segment_idx < len(self.unprinted_segments):
                    segment_idx += 1
                agent.is_printing = False
                print(f"Agent {i} initial line segment: {agent.current_line_segment}")
            else:
                # No more segments to assign
                break



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
        
        # Reassign unprinted line segments to agents
        self.unprinted_segments = [(self.print_path_points[i], self.print_path_points[i + 1]) for i in range(len(self.print_path_points) - 1)]
        for agent in self.agents:
            if self.unprinted_segments:
                agent.current_line_segment = self.unprinted_segments.pop(0)
                agent.is_printing = False
            else:
                agent.current_line_segment = None
                agent.is_printing = False
        
        #self.assign_print_paths()
        self.visualize_print_path()
        
    def reward(self, agent: Agent):
        distance_to_goal = torch.norm(agent.state.pos - self.goal_pos)
        return -distance_to_goal.unsqueeze(0)

    def observation(self, agent: Agent):
        expanded_goal_pos = self.goal_pos.unsqueeze(0).expand(agent.state.pos.size(0), -1)
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)

    def update_trail(self):
        for agent in self.agents:
            if agent.current_line_segment:
                start_point, end_point = agent.current_line_segment

                # If the agent is printing and has reached the endpoint
                if agent.is_printing and torch.norm(agent.state.pos - end_point) <= self.trail_distance:
                    print(f"Agent {agent.name} has finished printing line segment {agent.current_line_segment}")
                    # Stop printing
                    agent.is_printing = False
                    # Set agent's new target position to (1, 1)
                    agent.target_pos = torch.tensor([[1, 1]], device=self._world.device)
                    # Reset the current line segment, waiting for the next one to be assigned
                    agent.current_line_segment = None

                    if self.unprinted_segments:
                        # If there are still unprinted line segments, assign one to the agent
                        agent.current_line_segment = self.unprinted_segments.pop(0)
                    else:
                        # If there are no more line segments, print message indicating all line segments have been printed
                        print(f"All line segments have been printed. Agent {agent.name} is moving to the side.")
                
                # If the agent has a target position, move towards the target position
                if hasattr(agent, 'target_pos') and agent.target_pos is not None:
                    self.move_agent_towards(agent, agent.target_pos)
                    # Check if the agent has reached the target position
                    if torch.norm(agent.state.pos - agent.target_pos) <= self.trail_distance:
                        # Reached the target position, remove the target position
                        agent.target_pos = None
                        print(f"Agent {agent.name} has moved to the side.")
                
                elif not agent.is_printing and torch.norm(agent.state.pos - start_point) <= self.trail_distance:
                    # If the agent has not started printing and is near the starting point, start printing
                    agent.is_printing = True
                    print(f"Agent {agent.name} is starting to print line segment {agent.current_line_segment}")
                elif agent.is_printing:
                    # If the agent is printing, continue moving towards the endpoint
                    self.move_agent_towards(agent, end_point)
                else:
                    # If the agent has not started printing, move towards the starting point
                    self.move_agent_towards(agent, start_point)


    def move_agent_towards(self, agent, target_pos):
        # Calculate the direction vector from the agent's position to the target position and update the agent's position
        direction = target_pos - agent.state.pos
        direction_norm = torch.norm(direction)
        if direction_norm > 0.05:
            direction = direction / direction_norm
        agent.state.pos += direction * agent.u_multiplier  # Update position
            
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
        #print(f"observation: {observation}")
        pos_agent = observation[:, :2]  # Agent's current position
        goal_pos = current_goal_pos.unsqueeze(0).expand(pos_agent.size(0), -1)  # Set the goal position to the current goal position
        
        #print(f"goal_pos: {goal_pos[0]}")

        # Calculate the vector from the agent to the goal
        vector_to_goal = goal_pos - pos_agent
        #print(f"vector_to_goal: {vector_to_goal[0]}")
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
