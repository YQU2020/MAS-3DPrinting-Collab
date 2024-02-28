import torch, math, csv
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere, Landmark, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import TorchUtils, Color
from vmas.simulator import rendering
import numpy as np
import operator

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
            agent.task_queue = []  # List to store the tasks to be executed
            
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

    def visulalize_endpoints(self, start_points, end_points, color=Color.GRAY):
        for point in start_points, end_points:
            self.pathpoints_landmark = Landmark(
            name="pathpoints_landmark",
            collide=False,
            shape=Sphere(radius=0.01),
            color=color
            )
            self._world.add_landmark(self.pathpoints_landmark)
            self.pathpoints_landmark.set_pos(point, batch_index=0)

    def add_line_to_world(self, start_point, end_point, color=Color.GRAY):
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
        num_tasks = len(self.unprinted_segments)
        num_agents = len(self.agents)
        # Initialize a 2D array with zeros. Shape: [num_tasks, num_agents]
        bids = np.zeros((num_tasks, num_agents, 3))
        
        for task_idx, task in enumerate(self.unprinted_segments):
            for agent_idx, agent in enumerate(self.agents):
                start_point, _ = task
                distance = torch.norm(agent.state.pos - start_point)
                workload_factor = len(agent.completed_tasks)
                bid = distance + workload_factor * WORKLOAD_WEIGHT
                bids[task_idx, agent_idx] = [task_idx, agent_idx, bid.item()]
                
        return bids

    def assign_tasks_based_on_bids(self):
        bids = self.collect_bids()
        # Flatten the array to 2D for sorting by bid
        bids_flat = bids.reshape(-1, 3)
        sorted_indices = np.argsort(bids_flat[:, 2])

        # Initialize an empty task queue for each agent
        for agent in self.agents:
            agent.task_queue = []

        # Temporary structure to keep track of tasks assigned to each agent
        tasks_assigned_to_agents = {agent_idx: [] for agent_idx in range(len(self.agents))}

        # Assign tasks based on sorted bids
        for flat_index in sorted_indices:
            task_idx, agent_idx, _ = [int(bids_flat[flat_index][i]) for i in range(3)]
            if len(tasks_assigned_to_agents[agent_idx]) < 2 and task_idx not in tasks_assigned_to_agents[agent_idx]:
                # Add task to agent's queue if it doesn't already contain it and has less than 2 tasks
                tasks_assigned_to_agents[agent_idx].append(task_idx)
                # Convert task index to actual task (assuming unprinted_segments are tensors that need to be tolist)
                agent_task = self.unprinted_segments[task_idx]
                self.agents[agent_idx].task_queue.append(agent_task)

                    
    def execute_tasks_allocation(self):
        
        self.assign_tasks_based_on_bids()  # Ensure task queues are up-to-date
        
        for agent in self.agents: 
            if not agent.is_printing and agent.task_queue:
                for task in agent.task_queue:
                    if not any(
                        other_agent.is_printing and 
                        torch.equal(other_agent.current_line_segment[0], task[0]) and
                        torch.equal(other_agent.current_line_segment[1], task[1])
                        for other_agent in self.agents if other_agent != agent
                    ):
                        agent.current_line_segment = task
                        agent.is_printing = True
                        index_to_remove = None
                        for i, segment in enumerate(self.unprinted_segments):
                            # Check if both tensors in the tuple match the target tensors
                            if torch.all(torch.eq(segment[0], task[0])) and torch.all(torch.eq(segment[1], task[1])):
                                index_to_remove = i
                                break
                        
                        # If an item was found, remove it by index
                        if index_to_remove is not None:
                            self.add_line_to_world(task[0], task[1], color=agent.color)
                            self.unprinted_segments.pop(index_to_remove)
                        break  # Task assigned, exit loop

                # Update task queue after assignment
                agent.task_queue = [
                    task for task in agent.task_queue
                    if not operator.eq(task[0], agent.current_line_segment)  # Assuming both are tuples
                ]


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