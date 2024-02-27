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
            agent.completed_tasks = []  # List to store completed tasks
    
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
                    

    def reward(self, agent: Agent):
        distance_to_goal = torch.norm(agent.state.pos - agent.goal_pos)
        return -distance_to_goal.unsqueeze(0)

    def observation(self, agent: Agent):
        num_envs = agent.state.pos.size(0)  # size of batch
        expanded_goal_pos = agent.goal_pos.expand(num_envs, -1)  # expand goal position to match batch size
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)

    def update_trail(self):
        need_reassignment = False  # Flag to indicate whether task reassignment is needed
        
        for agent in self.agents:
            if agent.current_line_segment:
                start_point, end_point = agent.current_line_segment
                if agent.is_printing and torch.norm(agent.state.pos - end_point) <= self.trail_distance:
                    agent.is_printing = False  # Finished printing
                    agent.at_start = False  # Reset the flag
                    agent.completed_tasks.append(agent.current_line_segment)  # Add the completed task to the list
                    self.set_line_collidable(start_point, end_point, collidable=True)  # set line collidable
                    agent.current_line_segment = None  # erase current line segment
                    agent.goal_pos = agent.state.pos  # if no more line segments, stay at current position
                    print(f"Agent {agent.name} finished segment. Moving to next segment.printed {len(agent.completed_tasks)} segments")
                    
                    need_reassignment = True  # Flag that task reassignment is needed
                    
        # If there are unprinted segments and some agents are not printing, reassign tasks
        if need_reassignment and any(agent.current_line_segment is None for agent in self.agents):
            self.execute_tasks_allocation()

    def move_agent(self, agent):
        # Decide whether agent has reached the start of the line segment
        target_point = agent.current_line_segment[1] if agent.is_printing else agent.current_line_segment[0]
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

    def add_line_to_world(self, start_point, end_point, color):
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
            
        return path_points
    
    # ============== Auction-based assignment ================
    
    def collect_bids(self):
        WORKLOAD_WEIGHT = 0.1  # Weight for the workload factor
        bids = []  # Save in the form of (task_idx, agent_idx, bid)
        for task_idx, task in enumerate(self.unprinted_segments):
            for agent_idx, agent in enumerate(self.agents):
                start_point, _ = task
                distance = torch.norm(agent.state.pos - start_point)
                workload_factor = len(agent.completed_tasks)  # Assume each agent has a 'completed_tasks' list
                bid = distance + workload_factor * WORKLOAD_WEIGHT  # WORKLOAD_WEIGHT is a predefined constant
                bids.append((task_idx, agent_idx, bid.item()))
        return bids


    def assign_tasks_based_on_bids(self):
        bids = self.collect_bids()
        sorted_bids = sorted(bids, key=lambda x: x[2])

        assigned_tasks = set()
        assigned_agents = set()
        tasks_to_remove = []  # Save the task indices to be removed

        for task_idx, agent_idx, bid in sorted_bids:
            if task_idx not in assigned_tasks and agent_idx not in assigned_agents and self.agents[agent_idx].is_printing == False:
                self.agents[agent_idx].current_line_segment = self.unprinted_segments[task_idx]
                self.agents[agent_idx].is_printing = True
                assigned_tasks.add(task_idx)
                assigned_agents.add(agent_idx)
                tasks_to_remove.append(task_idx)  # Add to the list to be removed
                
                print(f"Task {task_idx} assigned to Agent {self.agents[agent_idx].name} with bid {bid}")

                if len(assigned_tasks) == len(self.unprinted_segments):
                    break

        # Use reverse order to remove assigned tasks to avoid index issues
        for task_idx in sorted(tasks_to_remove, reverse=True):
            self.unprinted_segments.pop(task_idx)

                    
    def execute_tasks_allocation(self):
        if not self.unprinted_segments:
            return  # If there are no unprinted line segments, return directly
        
        # Implement the auction
        self.assign_tasks_based_on_bids()
        
        # Labeling each task-assigned intelligence ready to perform the task 
        for agent in self.agents:
            if agent.current_line_segment:
                agent.is_moving_to_task = True  # Labeling each task-assigned intelligence ready to perform the task
                

class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, agent: Agent, u_range: float) -> torch.Tensor:
        if agent.current_line_segment is not None:
            if not agent.at_start:
                # If the agent is not at the start of the line segment, the target position is the start point
                target_pos = agent.current_line_segment[0]  
                if torch.norm(agent.state.pos - target_pos) < 0.05:
                    # If the agent is close to the start point, mark it as at the start
                    agent.at_start = True  
                    agent.goal_pos = agent.current_line_segment[1]  # Update the goal position
            else:
                # If the agent is at the start of the line segment, the target position is the end point
                target_pos = agent.current_line_segment[1]
                
            # Update the goal position in the observation
            goal_pos = target_pos.unsqueeze(0) if target_pos.dim() == 1 else target_pos
        else:
            # If there is no current task, the goal position remains unchanged (may be the initialized position)
            goal_pos = agent.goal_pos
        
        pos_agent = observation[:, :2]  # Agent's current position
        vector_to_goal = goal_pos - pos_agent  # Vector to the target position
        
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