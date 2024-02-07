import torch, math, csv
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere, Landmark, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils, Color
from vmas.simulator import rendering

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self._world = World(batch_dim, device)
        self.num_agents = 2  # M=2
        agent_colors = [Color.RED, Color.BLUE, Color.LIGHT_GREEN]  # Define the colors for the agents

        # Create agents with different colors
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(name=f"agent_{i}", u_multiplier=0.4, shape=Sphere(0.03), collide=False, color=agent_colors[i])
            self.agents.append(agent)
            self._world.add_agent(agent)
            agent.is_printing = False
            agent.current_line_segment = None
            agent.at_start = False
            agent.goal_pos = torch.tensor([0.0, 0.0])
    
        # trail
        self.trail_active = False
        self.trail_points = []
        self.last_point = None
        self.trail_distance = 0.1

        self.unprinted_segments = []
        # visualize in run_assign_print_path.py! 
        
        # Create a new print path, complex shape, hard-coded
        #self.print_path_points = self.create_complex_print_path(20)
        
        # Read the print path from a CSV file (200_coordinates.csv)
        self.print_path_points = self.read_print_path_from_csv(f"/home/abcd/Documents/GitHub/VMAS_yqu_implementation/vmas/scenarios/coordinates.csv")
        
        return self._world
            
    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:

            # Set a random start position for the agent
            random_start_pos = torch.rand((1, self.world.dim_p), device=self.world.device) * 2 - 1
            agent.set_pos(random_start_pos, batch_index=env_index)
            
            self.trail_points.clear()
            self.last_point = None
            self.trail_active = False
            
            self.current_segment_index = 0
        
        # Only initialize unprinted_segments if it hasn't been done yet
        
        if not hasattr(self, 'unprinted_segments') or not self.unprinted_segments:
            for i in range(0, len(self.print_path_points), 2):  # Iterate through pairs of points
                if i+1 < len(self.print_path_points):
                    self.unprinted_segments.append((self.print_path_points[i], self.print_path_points[i+1]))
                    
        for agent in self.agents:
            if self.unprinted_segments:
                agent.current_line_segment = self.unprinted_segments.pop(0)
                agent.is_printing = False
            else:
                agent.current_line_segment = None
                agent.is_printing = False
                
        #print(f"Unprinted Segments: {self.unprinted_segments}")
        

    def reward(self, agent: Agent):
        distance_to_goal = torch.norm(agent.state.pos - agent.goal_pos)
        return -distance_to_goal.unsqueeze(0)

    def observation(self, agent: Agent):
        num_envs = agent.state.pos.size(0)  # size of batch
        expanded_goal_pos = agent.goal_pos.expand(num_envs, -1)  # expand goal position to match batch size
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)

    def update_trail(self):
        for agent in self.agents:
            if agent.current_line_segment:
                start_point, end_point = agent.current_line_segment
                if agent.is_printing and torch.norm(agent.state.pos - end_point) <= self.trail_distance:
                    agent.is_printing = False  # Finished printing
                    self.set_line_collidable(start_point, end_point, collidable=True)  # set line collidable
                    agent.current_line_segment = None  # erase current line segment
                    if self.unprinted_segments:
                        agent.current_line_segment = self.unprinted_segments.pop(0)
                        agent.at_start = False
                        agent.goal_pos = agent.current_line_segment[0]  # set goal position to start of next line segment
                    else:
                        agent.goal_pos = agent.state.pos  # if no more line segments, stay at current position
                    print(f"Agent {agent.name} finished segment. Moving to next segment.")
                elif not agent.is_printing:
                    self.move_agent(agent)

    def move_agent(self, agent):
        # Decide whether agent has reached the start of the line segment
        target_point = agent.current_line_segment[1] if agent.is_printing else agent.current_line_segment[0]
        print(f"Agent {agent.name} is printing: {agent.is_printing}, target point: {target_point.tolist()}")
        # logic to decide whether agent has reached the start of the line segment
        direction = target_point - agent.state.pos
        direction_norm = torch.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        agent.state.pos += direction * agent.u_multiplier
            
    def add_landmark(self, position, color=Color.GREEN):
            landmark = Landmark(
                name=f"landmark_{len(self.world.landmarks)}",
                collide=False,
                shape=Sphere(radius=0.03),
                color=color,
            )
            self.world.add_landmark(landmark)
            landmark.set_pos(position.unsqueeze(0), batch_index=0)
            
    def avoid_collision(self, agent, other_agent):
        # Calculate the direction vector from agent to other_agent
        direction_to_other = other_agent.state.pos - agent.state.pos
        direction_to_other_norm = torch.norm(direction_to_other)

        if direction_to_other_norm > 0:
            # Normalize the direction vector
            direction_to_other = direction_to_other / direction_to_other_norm

            # Create a new direction that is perpendicular to the direction to the other agent
            # This is a simple way to create an avoidance maneuver
            avoidance_direction = torch.tensor([-direction_to_other[1], direction_to_other[0]])

            # Update the agent's position to move away from the other agent
            agent.state.pos += avoidance_direction * agent.u_multiplier

    def visulalize_endpoints(self, start_points, end_points):
        for point in start_points, end_points:
            self.pathpoints_landmark = Landmark(
            name="pathpoints_landmark",
            collide=False,
            shape=Sphere(radius=0.01),
            color=Color.GRAY,
            )
            self._world.add_landmark(self.pathpoints_landmark)
            self.pathpoints_landmark.set_pos(point, batch_index=0)

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
                print(f"No more segments to assign to agent {i}")
                break

    def add_line_to_world(self, start_point, end_point, color):
        print(f"Adding line from {start_point.tolist()} to {end_point.tolist()}")
        # Create a line landmark
        line = Landmark(
            name="line",
            collide=False,  # Set collision detection as needed
            # The Line class may require a length parameter, you may need to calculate the distance between start_point and end_point
            shape=Line(length=torch.norm(end_point - start_point).item()),
            color=color,
        )
        # Add the line to the world
        self._world.add_landmark(line)
        # Set the position and orientation of the line
        line.set_pos((start_point + end_point) / 2.0, batch_index=0)  # Set position to the midpoint of the two points
        angle = torch.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        line.set_rot(angle.unsqueeze(0), batch_index=0)  # Set rotation angle
        
    def set_line_collidable(self, start_point, end_point, collidable):
        # Iterate through all landmarks to find the corresponding line segment
        for landmark in self._world.landmarks:
            if isinstance(landmark, Line) and torch.all(torch.eq(landmark.start, start_point)) and torch.all(torch.eq(landmark.end, end_point)):
                # Found the corresponding line segment, update its collision property
                landmark.collide = collidable
                break
    # ============== Set of functions to create different print paths ================
                
    def create_complex_print_path(self, num_segments):
        # Create a complex print path with multiple line segments
        path_segments = []
        num_points = num_segments * 2  

        # Create points to form a complex shape (e.g., a square or curve)
        for i in range(0, num_points, 2):  # Each line segment is defined by two points
            if i < num_points // 3:
                # Horizontal line segment
                start_point = torch.tensor([i * 3.0 / num_points, 0.0]) * 2 - 1
                end_point = torch.tensor([(i+1) * 3.0 / num_points, 0.0]) * 2 - 1
            elif i < 2 * num_points // 3:
                # Vertical line segment
                start_point = torch.tensor([1.0, (i - num_points // 3) * 3.0 / num_points]) * 2 - 1
                end_point = torch.tensor([1.0, ((i+1) - num_points // 3) * 3.0 / num_points]) * 2 - 1
            else:
                # Circular line segment
                start_angle = (i - 2 * num_points // 3) * 2 * math.pi / (num_points // 3)
                end_angle = ((i+1) - 2 * num_points // 3) * 2 * math.pi / (num_points // 3)
                start_point = torch.tensor([math.cos(start_angle), math.sin(start_angle)]) * 0.5 + torch.tensor([0.5, 0.5])
                end_point = torch.tensor([math.cos(end_angle), math.sin(end_angle)]) * 0.5 + torch.tensor([0.5, 0.5])

            path_segments.append(start_point)
            path_segments.append(end_point)

        return path_segments

    def read_print_path_from_csv(self, file_path):
        path_points = []
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Ensure that each row has four elements (coordinates of two points)
                if len(row) == 4:
                    start_point = torch.tensor([float(row[0]), float(row[1])])
                    end_point = torch.tensor([float(row[2]), float(row[3])])
                    # Add line segment (pair of start and end points)
                    path_points.append(start_point)
                    path_points.append(end_point)
            
        #print("Print Path Points Coordinates:", path_points)
        return path_points
        
class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, agent: Agent, u_range: float) -> torch.Tensor:
        pos_agent = observation[:, :2]  # Agent's current position
        # Position of goal_pos in observation data
        goal_pos = observation[:, 4:6]  # Goal position
        print(f"Agent {agent.name} at_start: {agent.at_start}, goal_pos: {agent.goal_pos[0]}")
        if agent.current_line_segment is not None:
            if not agent.at_start:
                # Agent has not reached the starting point, target position is the start of the line segment
                goal_pos = agent.current_line_segment[0].unsqueeze(0)
            else:
                # Agent has reached the starting point, target position is the end of the line segment
                goal_pos = agent.current_line_segment[1].unsqueeze(0)
                agent.is_printing = True

        # Compute the vector from the agent to the goal position
        vector_to_goal = goal_pos - pos_agent
        
        # Normalize the direction vector
        norm_direction_to_goal = vector_to_goal / torch.clamp(torch.norm(vector_to_goal, dim=1, keepdim=True), min=1e-6)
        # Compute the action
        action = norm_direction_to_goal * u_range
        return action

if __name__ == "__main__":
    
    render_interactively(
        __file__,
        desired_velocity=0.05,
        n_agents=2,
        
    )