import torch, math, csv, os
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
        self.shared_reward = kwargs.get("shared_reward", False)
        self.shaping_factor = 100  # Shaping factor for the reward functionW
        self.num_agents = 2  # M=2
        self.agents_radius = 0.01
        self.landmarks_radius = 0.01
        agent_colors = [Color.RED, Color.BLUE, Color.LIGHT_GREEN]  # Define the colors for the agents
        self.A_star_step_size = 0.10  # Step size for A* pathfinding

        # Create agents with different colors
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(name=f"agent_{i}", u_multiplier=0.05, shape=Sphere(self.agents_radius), collide=False, color=agent_colors[i])
            self.agents.append(agent)
            self._world.add_agent(agent)
            agent.is_printing = False
            agent.current_line_segment = None
            agent.at_start = False
            agent.goal_pos = torch.tensor([0.0, 0.0])
            agent.farther_goal_pos = torch.tensor([0.0, 0.0])
            agent.completed_tasks = []  # List to store completed tasks
            agent.completed_total_dis = 0.0
            agent.task_queue = []  # List to store the tasks to be executed
            agent.previus_pos = None
            agent.path = []
            
        # trail
        self.trail_active = False
        self.trail_points = []
        self.last_point = None

        self.unprinted_segments = []
        self.printed_segments = []
        self.all_segments = []
        # visualize in run_assign_print_path.py! 
        
        # Read the print path from a CSV file (200_coordinates.csv)
        current_directory = os.path.dirname(__file__)
        csv_file_path = os.path.join(current_directory, "../scenarios/house_floor_plan.csv")
        self.print_path_points = self.read_print_path_from_csv(csv_file_path)
        
        return self._world
            
    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:

            # Set a random start position for the agent
            random_start_pos = torch.rand((1, self.world.dim_p), device=self.world.device) * 1 - 0.8

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
                    self.all_segments.append((self.print_path_points[i], self.print_path_points[i+1]))
                    
        
    def reward(self, agent: Agent):
        self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

        # Calculate distance to the goal and give reward
        if agent.goal is not None:  # Make sure the agent has a goal
            dist_to_goal = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1)
            agent_shaping = dist_to_goal * self.shaping_factor
            self.rew += agent.global_shaping - agent_shaping
            agent.global_shaping = agent_shaping

        else:
            distance_to_goal = torch.norm(agent.state.pos - agent.goal_pos)
            self.rew -= distance_to_goal.unsqueeze(0)

        # Collision penalty
        collision_penalty = -5.0
        for landmark in self.world.landmarks:
            if landmark.collide:  # Assume only collideable landmarks will affect the reward
                if self.is_overlapping(agent, landmark):
                    self.rew += collision_penalty
                    
            return self.rew


    def is_overlapping(self, agent, landmark):
        # Check if agent is overlapping (colliding) with the landmark
        # Here we use a simple circular collision detection logic
        agent_pos = agent.state.pos
        landmark_pos = landmark.state.pos
        distance = torch.linalg.vector_norm(agent_pos - landmark_pos, dim=1)
        average_distance = torch.mean(distance)
        # Assume the radius of both agent and landmark are known
        return average_distance < (self.agents_radius)

    def observation(self, agent: Agent):
        num_envs = agent.state.pos.size(0)  # size of batch
        expanded_goal_pos = agent.goal_pos.expand(num_envs, -1)  # expand goal position to match batch size
        return torch.cat([agent.state.pos, agent.state.vel, expanded_goal_pos], dim=-1)
                        
    def add_landmark(self, position, color=Color.GREEN):
            landmark = Landmark(
                name=f"landmark_{len(self.world.landmarks)}",
                collide=False,
                shape=Sphere(radius=self.landmarks_radius),
                color=color,
            )
            self.world.add_landmark(landmark)
            landmark.set_pos(position.unsqueeze(0), batch_index=0)
            
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

    def add_line_to_world(self, start_point, end_point, color=Color.GRAY, collide=False):
        # Create a line landmark
        line = Landmark(
            name="line",
            collide=collide,  # Set collision detection as needed
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
        
    # Which endpoint to start printing from
    def choose_task_endpoint(self, agent, task):
        start_point, end_point = task
        dist_to_start = torch.norm(agent.state.pos - start_point)
        dist_to_end = torch.norm(agent.state.pos - end_point)
        if dist_to_start <= dist_to_end:
            return start_point, end_point  # Start printing from the start point
        else:
            return end_point, start_point  # Start printing from the end point
            
    # ============== Set of functions to read different print paths ================

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
                workload_factor = len(agent.completed_tasks) * 0.1 + agent.completed_total_dis * 0.1 
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
        self.assign_tasks_based_on_bids()

        for agent in self.agents:
            if not agent.is_printing and agent.task_queue:
                task = agent.task_queue[0]  # Consider only the next task in the queue
                if not self.is_task_being_executed_by_another_agent(task, agent):
                    agent.current_line_segment = task
                    agent.is_printing = True

                    self.remove_task_from_unprinted(task)
                    self.printed_segments.append(task)

                    print(f"Agent {agent.name} assigned to segment {task}.")

                    # Calculate path to task start; if found, update agent.path; else, set direct goal_pos
                    self.calculate_and_set_path_to_task_start(agent, task)

                # After assigning a task and potentially a path, break to avoid over-assignment
                break

        # Update the task queues considering the current task assignments
        self.update_agents_task_queues()

    def calculate_and_set_path_to_task_start(self, agent, task):
        start_point_of_task = task[0]
        path_to_task_start = self.a_star_pathfinding(agent.state.pos[0], start_point_of_task, self.printed_segments)
        
        if path_to_task_start:
            agent.path = path_to_task_start
            agent.goal_pos = path_to_task_start[1] if len(path_to_task_start) > 1 else path_to_task_start[0]  # Next step in path
            print(f"Path calculated for Agent {agent.name} to start of segment {task}.")
        else:
            print(f"No path found for Agent {agent.name}. Direct movement to {start_point_of_task}.")
            agent.goal_pos = start_point_of_task  # Direct goal set as fallback

    def is_task_being_executed_by_another_agent(self, task, current_agent):
        for other_agent in self.agents:
            if other_agent != current_agent and other_agent.is_printing:
                if torch.equal(other_agent.current_line_segment[0], task[0]) and torch.equal(other_agent.current_line_segment[1], task[1]):
                    return True
        return False

    def remove_task_from_unprinted(self, task):
        self.unprinted_segments = [
            seg for seg in self.unprinted_segments if not (
                torch.all(torch.eq(seg[0], task[0])) and torch.all(torch.eq(seg[1], task[1]))
            )
        ]

    def update_agents_task_queues(self):
        for agent in self.agents:
            if agent.current_line_segment is not None:
                agent.task_queue = [
                    task for task in agent.task_queue
                    if not all(torch.equal(a, b) for a, b in zip(task, agent.current_line_segment))
                ]


    # ============== Obstacle Avoidence ================

    # Calculate the Euclidean distance between two points
    def euclidean_distance(self, point1, point2):
        # Ensure point1 and point2 are torch.Tensor types
        point1 = torch.tensor(point1, dtype=torch.float32)
        point2 = torch.tensor(point2, dtype=torch.float32)
        
        return torch.norm(point1 - point2).item()
    
    # Calculate the Manhattan distance between two points
    def manhattan_distance(self, point1, point2):
        point1 = torch.tensor(point1, dtype=torch.float32)
        point2 = torch.tensor(point2, dtype=torch.float32)
        
        return torch.sum(torch.abs(point1 - point2)).item()

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        # Tensorize the path
        return [torch.tensor(point, dtype=torch.float32) for point in total_path]
    
    # A* pathfinding algorithm
    def a_star_pathfinding(self, start, goal, obstacles):
        # Convert start and goal to rounded tuples to ensure consistency
        start_key = tuple(round(x.item(), 2) for x in torch.tensor(start, dtype=torch.float32))
        goal_key = tuple(round(x.item(), 2) for x in torch.tensor(goal, dtype=torch.float32))
        goal_tolerance = self.A_star_step_size

        # Initialize sets and dictionaries using rounded keys
        open_set = set([start_key])
        came_from = {}
        g_score = {start_key: 0}
        f_score = {start_key: self.manhattan_distance(torch.tensor(start_key), torch.tensor(goal_key))}

        while open_set:
            current_key = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            current_pos = torch.tensor(current_key, dtype=torch.float32)

            # Check if current position is within goal_tolerance of the goal
            if torch.norm(current_pos - torch.tensor(goal_key), p=1) <= goal_tolerance:
                return self.reconstruct_path(came_from, current_key)

            open_set.remove(current_key)
            for neighbor in self.get_neighbors(current_pos, obstacles):
                neighbor_key = tuple(round(n.item(), 2) for n in neighbor)

                # Use neighbor_key consistently for accessing g_score and f_score
                tentative_g_score = g_score[current_key] + self.manhattan_distance(current_pos, torch.tensor(neighbor_key))

                if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                    came_from[neighbor_key] = current_key
                    g_score[neighbor_key] = tentative_g_score
                    f_score[neighbor_key] = tentative_g_score + self.manhattan_distance(torch.tensor(neighbor_key), torch.tensor(goal_key))
                    open_set.add(neighbor_key)
            #print(f"Open set: {open_set}, came from: {came_from}, g_score: {g_score}, f_score: {f_score}")
        return []


    def get_neighbors(self, position, obstacles, step_size=0.10):  # Enlarge step_size
        step_size = self.A_star_step_size
        directions = [torch.tensor([step_size, 0]), torch.tensor([-step_size, 0]),
                    torch.tensor([0, step_size]), torch.tensor([0, -step_size])]
        neighbors = []

        # Define max and min borders
        max_border = 2.00
        min_border = -2.00

        for d in directions:
            neighbor_pos = position + d
            # Check if neighbor is within the specified borders before checking for collision
            if min_border <= neighbor_pos[0] <= max_border and min_border <= neighbor_pos[1] <= max_border:
                # Add a check for obstacles to ensure the neighbor position is not inside an obstacle
                if not self.is_collision(neighbor_pos, obstacles):
                    neighbors.append(neighbor_pos)

        return neighbors

    def point_to_line_segment_distance(self, point, segment_start, segment_end):
        # Convert to numpy arrays for easier manipulation
        p = point.numpy()
        a = segment_start.numpy()
        b = segment_end.numpy()

        # Vector from a to b
        ab = b - a
        # Vector from a to p
        ap = p - a

        # Project vector ap onto ab, then find the magnitude of the projection
        projection = np.dot(ap, ab) / np.linalg.norm(ab)**2 * ab
        closest_point = a + projection

        # Ensure the closest point is within the line segment
        if np.dot(ab, projection) < 0:
            closest_point = a
        elif np.linalg.norm(projection) > np.linalg.norm(ab):
            closest_point = b

        # Calculate the distance between the point and the closest point on the segment
        distance = np.linalg.norm(p - closest_point)

        return distance
    
    def preprocess_obstacles(self, obstacles, step_size=0.10):
        impassable_grids = {}
        step_size = self.A_star_step_size
        for start, end in obstacles:
            x_min, y_min = np.floor(np.min([start[0], end[0]]) / step_size) * step_size, np.floor(np.min([start[1], end[1]]) / step_size) * step_size
            x_max, y_max = np.ceil(np.max([start[0], end[0]]) / step_size) * step_size, np.ceil(np.max([start[1], end[1]]) / step_size) * step_size

            x = x_min
            while x <= x_max:
                y = y_min
                while y <= y_max:
                    grid_key = (round(x, 2), round(y, 2))
                    impassable_grids[grid_key] = True
                    y += step_size
                x += step_size
                
        return impassable_grids

    
    def is_collision(self, position, obstacles):
        obstacle_hash = self.preprocess_obstacles(obstacles, self.A_star_step_size)
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position)
        if position.dim() > 1:
            position = position.view(-1)

        # Convert position to grid coordinates
        grid_x, grid_y = round(position[0].item() / self.A_star_step_size, 2), round(position[1].item() / self.A_star_step_size, 2)
        grid_key = (grid_x, grid_y)

        # Check the spatial hash map for collision
        return obstacle_hash.get(grid_key, False)

    
    def on_collision_detected(self, agent):
        # Check which endpoint of the current line segment is closer to the agent, then plan a new path
        current_task = agent.current_line_segment
        if current_task is None:
            print(f"Agent {agent.name} does not have a current task. No collision action taken.")
            return

        start_point, end_point = current_task

        # calculate a point that is closer to the agent and after the obstacle intersection
        # instead of go straight to the end_point
        closer_point_to_agent = agent.state.pos[0] + agent.state.vel[0] + self.landmarks_radius

        # Use A* pathfinding to find a new path
        path = self.a_star_pathfinding(agent.state.pos[0], closer_point_to_agent, self.printed_segments)
        #print(f"Path is {path}.")
        if path:
            # if a new path is found, set the goal position to the next point on the path
            #print(f"New path found for Agent {agent.name} to avoid the obstacle.")
            agent.goal_pos = path[1] if len(path) > 1 else path[0]  # set the next point on the path as the goal position
            #self.visualize_path(path)  # Visualize the new path (optional)
        else:
            print(f"No path found for Agent {agent.name} to avoid the obstacle.")
            raise Exception("No path found to avoid the obstacle.")
            

    def is_path_obstructed(self, agent_position, goal_position, obstacles):
        """
        Checks if the direct path from agent_position to goal_position is obstructed by any obstacle.
        
        :param agent_position: Current position of the agent (tensor).
        :param goal_position: Goal position the agent is trying to reach (tensor).
        :param obstacles: List of obstacles, each defined by its start and end points (list of tuples of tensors).
        :return: True if the path is obstructed by an obstacle, False otherwise.
        """
        # Convert tensors to numpy for easier calculations
        agent_pos_np = agent_position.numpy()
        goal_pos_np = goal_position.numpy()
        
        for obstacle in obstacles:
            obstacle_start_np, obstacle_end_np = obstacle[0].numpy(), obstacle[1].numpy()

            # Check for intersection using the line intersection formula
            # Formula: A1x + B1y = C1 for line 1 (agent to goal), A2x + B2y = C2 for line 2 (obstacle)
            A1 = goal_pos_np[1] - agent_pos_np[1]
            B1 = agent_pos_np[0] - goal_pos_np[0]
            C1 = A1 * agent_pos_np[0] + B1 * agent_pos_np[1]
            
            A2 = obstacle_end_np[1] - obstacle_start_np[1]
            B2 = obstacle_start_np[0] - obstacle_end_np[0]
            C2 = A2 * obstacle_start_np[0] + B2 * obstacle_start_np[1]
            
            determinant = A1 * B2 - A2 * B1
            
            if determinant != 0:  # Lines are not parallel
                x_inter = (C1 * B2 - C2 * B1) / determinant
                y_inter = (A1 * C2 - A2 * C1) / determinant
                # Check if the intersection point is within the line segments
                if (min(agent_pos_np[0], goal_pos_np[0]) <= x_inter <= max(agent_pos_np[0], goal_pos_np[0]) and
                    min(agent_pos_np[1], goal_pos_np[1]) <= y_inter <= max(agent_pos_np[1], goal_pos_np[1]) and
                    min(obstacle_start_np[0], obstacle_end_np[0]) <= x_inter <= max(obstacle_start_np[0], obstacle_end_np[0]) and
                    min(obstacle_start_np[1], obstacle_end_np[1]) <= y_inter <= max(obstacle_start_np[1], obstacle_end_np[1])):
                    return True  # Path is obstructed
        
        return False  # Path is not obstructed


    def attempt_new_path(self, agent):
        # assumed the agent has a goal position attribute agent.goal_pos
        new_path = self.a_star_pathfinding(agent.state.pos[0], agent.goal_pos, self.printed_segments)
        print(f"New path found for Agent {agent.name}, new path {new_path}.")
        if new_path:
            print(f"Path is {new_path}.")
            agent.path = new_path[1:]  # Save the path, not include the current pos
            if agent.path:
                agent.goal_pos = agent.path.pop(0)  # update the next point on the path as goal pos
                self.visualize_path(agent.path)  # visualize
            print(f"Agent {agent.name} has a new path. now heading to {agent.goal_pos}")
        else:
            # if cannot find a new path, set the closest endpoint of the current line segment as the goal position
            if not agent.is_printing:
                agent.goal_pos = agent.current_line_segment[0]
                agent.current_line_segment = (agent.current_line_segment[1], agent.current_line_segment[0])
                # if then still cannot find a new path, attempt to find a new path again!
                
            print(f"No path found for Agent {agent.name} at {agent.state.pos[0]}. target is {agent.goal_pos}.")
            
            
    def visualize_path(self, path):
        # visualize the A* path by small black points
        for point in path:
            self.add_landmark(point, color=Color.BLUE)
            
    # Find a safe position after finishing printing, 
    # to avoid collision with segments printed by itself
    def find_safe_position(self, current_pos, obstacles, step_size=0.10, max_attempts=20):
        step_size = self.A_star_step_size
        directions = [torch.tensor([step_size, 0]), torch.tensor([-step_size, 0]),
                    torch.tensor([0, step_size]), torch.tensor([0, -step_size])]
        for _ in range(max_attempts):
            # randomly choose a direction
            direction = directions[np.random.randint(len(directions))]
            new_pos = current_pos + direction
            # Check if the new position is within the bounds and not colliding with obstacles
            if not self.is_collision(new_pos, obstacles):
                return new_pos
        # If no safe position is found, return None
        return None

class SimplePolicy:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
    def compute_action(self, observation: torch.Tensor, agent: Agent, u_range: float) -> torch.Tensor:
        """    
        # If the agent is close to the start point of the line segment
        if agent.current_line_segment and torch.norm(agent.state.pos[0] - agent.goal_pos) < 0.01:        # 0.01 is agent's radius
            agent.is_printing = True
            agent.goal_pos = agent.current_line_segment[1]  # Set the goal position to the end point of the line segment
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Agent {agent.name} is now printing the segment {agent.current_line_segment}.")

        # If the agent is close to the end point of the line segment
        if agent.is_printing and torch.norm(agent.state.pos[0] - agent.current_line_segment[1]) < 0.01:                  # 0.01 is agent's radius
            agent.is_printing = False
            agent.completed_tasks.append(agent.current_line_segment)  # Add the completed task to the list
            agent.current_line_segment = None  # Remove the current line segment
            agent.goal_pos = agent.state.pos[0]  # IF there is no more task, stop at the current position
            print("**********************************************************************************************")
            print(f"Agent {agent.name} is now finish printing the segment {agent.current_line_segment}.")

        # If the agent has no task, stop at the current position
        if agent.current_line_segment is None:
            agent.goal_pos = agent.state.pos[0]

        # Based on the current position and the goal position, calculate the action
        vector_to_goal = agent.goal_pos - observation[:, :2]
        normalized_direction = vector_to_goal / torch.clamp(torch.norm(vector_to_goal, dim=1, keepdim=True), min=1e-6)
        action = normalized_direction * u_range
        return action

        """
        if agent.current_line_segment is not None:
            if not agent.at_start:
                # If the agent is not at the start of the line segment, the target position is the start point
                target_pos = agent.current_line_segment[0]  
                if torch.norm(agent.state.pos - target_pos) < 0.01:
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
            goal_pos = agent.state.pos
            
        if len(agent.path) > 0:
            goal_pos = agent.path.pop(0)  
            
        vector_to_goal = goal_pos - agent.state.pos  # Vector to the target position
        
        # Normalize the direction vector
        norm_direction_to_goal = vector_to_goal / torch.clamp(torch.norm(vector_to_goal, dim=1, keepdim=True), min=1e-6)
        action = norm_direction_to_goal * u_range
        return action

if __name__ == "__main__":
    
    render_interactively(
        __file__,
        desired_velocity=0.05,
        n_agents=2,
        
    )