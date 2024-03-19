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
            agent = Agent(name=f"agent_{i}", u_multiplier=0.1, shape=Sphere(self.agents_radius), collide=True, color=agent_colors[i])
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
        
        # Create a new print path, complex shape, hard-coded
        #self.print_path_points = self.create_complex_print_path(20)
        
        # Read the print path from a CSV file (200_coordinates.csv)
        current_directory = os.path.dirname(__file__)
        csv_file_path = os.path.join(current_directory, "../scenarios/house_floor_plan.csv")
        self.print_path_points = self.read_print_path_from_csv(csv_file_path)
        
        return self._world
            
    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:

            # Set a random start position for the agent
            random_start_pos = torch.rand((1, self.world.dim_p), device=self.world.device) * 1 - 0.7

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
                    
        
    def reward(self, agent):
        # Initialize reward as 0
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
        collision_penalty = -10
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
                            self.add_line_to_world(task[0], task[1], color=Color.GREEN, collide=False)
                            self.unprinted_segments.pop(index_to_remove)
                            print(f"Agent {agent.name} assigned to segment {task}.")
                        break  # Task assigned, exit loop

                # Update task queue after assignment
                agent.task_queue = [
                    task for task in agent.task_queue
                    if not operator.eq(task[0], agent.current_line_segment)  # Assuming both are tuples
                ]

    # ============== Obstacle Avoidence ================

    # Calculate the Euclidean distance between two points
    def euclidean_distance(self, point1, point2):
        # Ensure point1 and point2 are torch.Tensor types
        point1 = torch.tensor(point1, dtype=torch.float32)
        point2 = torch.tensor(point2, dtype=torch.float32)
        
        # Calculate Euclidean distance
        return torch.norm(point1 - point2).item()
    
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
        f_score = {start_key: self.euclidean_distance(torch.tensor(start_key), torch.tensor(goal_key))}

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
                tentative_g_score = g_score[current_key] + self.euclidean_distance(current_pos, torch.tensor(neighbor_key))

                if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                    came_from[neighbor_key] = current_key
                    g_score[neighbor_key] = tentative_g_score
                    f_score[neighbor_key] = tentative_g_score + self.euclidean_distance(torch.tensor(neighbor_key), torch.tensor(goal_key))
                    open_set.add(neighbor_key)

        return []


    def get_neighbors(self, position, obstacles, step_size=0.10):  # 增大 step_size
        step_size = self.A_star_step_size
        directions = [torch.tensor([step_size, 0]), torch.tensor([-step_size, 0]),
                      torch.tensor([0, step_size]), torch.tensor([0, -step_size])]
        neighbors = []
        for d in directions:
            neighbor_pos = position + d
            # 添加对障碍物的检查，确保邻居位置不在障碍物内
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
        start_point, end_point = current_task

        # calculate a point that is closer to the agent and after the obstacle intersection
        # instead of go straight to the end_point
        closer_point_to_agent = agent.state.pos[0] + agent.state.vel[0] + self.landmarks_radius

        # Use A* pathfinding to find a new path
        path = self.a_star_pathfinding(agent.state.pos[0], closer_point_to_agent, self.printed_segments)

        if path:
            # if a new path is found, set the goal position to the next point on the path
            print(f"New path found for Agent {agent.name} to avoid the obstacle.")
            agent.goal_pos = path[1] if len(path) > 1 else path[0]  # set the next point on the path as the goal position
            #self.visualize_path(path)  # Visualize the new path (optional)
        else:
            print(f"No path found for Agent {agent.name} to avoid the obstacle.")
            
    def is_agent_will_be_block(self, agent, obstacles):
        # Calculate the expected position of the agent after one step
        expected_pos = agent.state.pos + agent.state.vel
        # Check if the expected position is within the bounds and not colliding with obstacles
        return self.is_collision(expected_pos, obstacles)
    
    # Using vector cross product to check if the predicted position is on the same side of the obstacle as the start and end points
    def is_agent_blocked(self, agent, obstacles):
        # calculate the predicted position of the agent after 0.1 seconds
        predicted_position = agent.state.pos[0] + agent.state.vel[0] * 0.1 
        for obstacle_start, obstacle_end in obstacles:
            # transform the tensors to numpy arrays for easier manipulation
            obstacle_start_np = obstacle_start.numpy()
            obstacle_end_np = obstacle_end.numpy()
            predicted_position_np = predicted_position.numpy()

            d1 = np.cross(obstacle_end_np - obstacle_start_np, predicted_position_np - obstacle_start_np)
            d2 = np.cross(obstacle_end_np - obstacle_start_np, predicted_position_np + agent.state.vel.numpy() * 0.1 - obstacle_start_np)
            if (d1 * d2 < 0).any():
                # If the predicted position is on the opposite side of the obstacle as the start and end points, the agent is considered blocked
                return True
        return False

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
    
    def bug_algorithm(self, agent, target_pos, obstacles, scenario):
        # Check if the agent is currently following an obstacle
        if not agent.following_obstacle:
            # Move directly towards the target
            direct_movement = move_directly_towards(agent.state.pos, target_pos)

            # Check for collision with any obstacle
            collision, obstacle = self.check_collision(agent.state.pos + direct_movement, obstacles)
            if collision:
                # Start following the obstacle's boundary
                agent.following_obstacle = True
                agent.current_obstacle = obstacle
                self.follow_boundary(agent, obstacle)
            else:
                # Continue moving towards the target
                agent.state.pos += direct_movement
        else:
            # Continue following the obstacle's boundary
            continue_following_boundary(agent, agent.current_obstacle, target_pos, obstacles, scenario)

    def move_directly_towards(self, current_pos, target_pos):
        # Calculate the vector towards the target
        direction = target_pos - current_pos
        # Normalize the direction
        direction_norm = direction / torch.norm(direction)
        # Define the movement step
        step_size = 0.01
        return direction_norm * step_size

    def check_collision(self, pos, obstacles):
        # Iterate over obstacles to check for collision
        for obstacle in obstacles:
            if self.is_collision(pos, obstacle):
                return True, obstacle
        return False, None
        
    def follow_boundary(self, agent, obstacles):     # When the agent encounters an obstacle, call this function to start walking along the boundary of the obstacle.
        # Assume the obstacles is a list of line segments, each segment is represented by a start point and an end point
        # Find the closest point on the obstacle's boundary to the agent
        closest_point, closest_segment = None, None
        min_distance = float('inf')
        for segment in obstacles:
            for point in segment:
                distance = torch.norm(agent.state.pos - point).item()
                if distance < min_distance:
                    closest_point = point
                    closest_segment = segment
                    min_distance = distance
        
        # Due to the simplification of this example, I assume the agent always walks clockwise along the boundary
        # Next point depends on the relative position and direction of the agent to the obstacle boundary
        # Here I assume the agent always walks clockwise along the boundary
        if closest_point is not None and closest_segment is not None:
            next_point = closest_segment[1] if closest_point.equal(closest_segment[0]) else closest_segment[0]
            agent.goal_pos = next_point

    def continue_following_boundary(self, agent, obstacles):        # Assume this function is called when the agent is already moving along the boundary of the obstacle

        # check if there is a direct path from the current position to the goal without intersecting any obstacles
         # This function checkss if there is a direct path from the current position to the goal without intersecting any obstacles
        direct_path_available = self.is_agent_will_be_block(agent.state.pos, agent.goal_pos, obstacles)

        if direct_path_available:
            # if a direct path exists, update the agent's goal position to move towards the final goal
            if agent.current_line_segment is not None:
                agent.goal_pos = agent.current_line_segment  #  update the agent's goal position to move towards the final goal
                print("Direct path found. Agent can move towards the goal.")
                return True
        else:
            # If a direct path does not exist, the agent should continue moving along the obstacle's boundary
            print("Continuing to follow the boundary.")
            
            next_boundary_point = self.find_next_boundary_point(agent.state.pos, obstacles)
            agent.goal_pos = next_boundary_point  # Update the agent's goal position to move towards the next boundary point
            return False
        
    def find_next_boundary_point(self, current_position, obstacles): # To find the next boundary point, the agent needs to find the closest point on the obstacle's boundary to its current position
        closest_point = None
        min_distance = float('inf')

        for segment in obstacles:
            for endpoint in [segment[0], segment[1]]:
                # calculate the distance between the agent's current position and sny endpoint
                distance = torch.norm(current_position - endpoint)

                # IF the distance is less than the minimum distance, update the closest point and the minimum distance
                if distance < min_distance:
                    closest_point = endpoint
                    min_distance = distance
        return closest_point



class SimplePolicy:
    def compute_action(self, observation: torch.Tensor, agent: Agent, u_range: float) -> torch.Tensor:
        if agent.current_line_segment is not None:
            if not agent.at_start:
                # If the agent is not at the start of the line segment, the target position is the start point
                target_pos = agent.current_line_segment[0]  
                if torch.norm(agent.state.pos - target_pos) < 0.04:
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