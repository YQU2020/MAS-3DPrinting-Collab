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
        self.agents_radius = 0.03
        self.landmarks_radius = 0.03
        agent_colors = [Color.RED, Color.BLUE, Color.LIGHT_GREEN]  # Define the colors for the agents

        # Create agents with different colors
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(name=f"agent_{i}", u_multiplier=0.5, shape=Sphere(self.agents_radius), collide=False, color=agent_colors[i])
            self.agents.append(agent)
            self._world.add_agent(agent)
            agent.is_printing = False
            agent.current_line_segment = None
            agent.at_start = False
            agent.goal_pos = torch.tensor([0.0, 0.0])
            agent.completed_tasks = []  # List to store completed tasks
            agent.completed_total_dis = 0.0
            agent.task_queue = []  # List to store the tasks to be executed
            agent.previus_pos = None
            
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

    def update_trail(self):
        need_reassignment = False  # Flag to indicate whether task reassignment is needed
        
        for agent in self.agents:
            if self.check_for_collisions(agent):
                print(f"Agent {agent.name} is avoiding collision.")
                self.plan_paths_for_agents()
                break
                
            if agent.current_line_segment:
                start_point, end_point = agent.current_line_segment
                print(f"Agent {agent.name} is printing segment {agent.current_line_segment} at {agent.state.pos[0]}, goalpoint is {agent.goal_pos}.New direction: {agent.state.vel[0]}. Distance to goal: {torch.norm(agent.state.pos[0] - end_point)}")
                
                if agent.is_printing and torch.norm(agent.state.pos[0] - end_point) < 0.04: # Assume the radius of the agent is 0.03
                    agent.is_printing = False  # Finished printing
                    agent.at_start = False  # Reset the flag
                    agent.completed_tasks.append(agent.current_line_segment)  # Add the completed task to the list
                    agent.completed_total_dis += torch.norm(start_point - end_point)
                    self.printed_segments.append(agent.current_line_segment)  # Add the completed task to the records
                    self.add_line_to_world(end_point, start_point, color=agent.color, collide=False)
                    print(f"Agent {agent.name} finished segment{agent.current_line_segment}. Moving to next segment.printed {len(agent.completed_tasks)} segments")
                    agent.current_line_segment = None  # erase current line segment
                    agent.goal_pos = agent.state.pos[0]  # if no more line segments, stay at current position
                    
                    need_reassignment = True  # Flag that task reassignment is needed
                    
        # If there are unprinted segments and some agents are not printing, reassign tasks
        if need_reassignment and any(agent.current_line_segment is None for agent in self.agents):
            self.execute_tasks_allocation()
            
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
                            self.add_line_to_world(task[0], task[1], color=Color.GREEN, collide=True)
                            self.unprinted_segments.pop(index_to_remove)
                            print(f"Agent {agent.name} assigned to segment {task}.")
                        break  # Task assigned, exit loop

                # Update task queue after assignment
                agent.task_queue = [
                    task for task in agent.task_queue
                    if not operator.eq(task[0], agent.current_line_segment)  # Assuming both are tuples
                ]

    # ============== Obstacle Avoidence ================
    
    def a_star_pathfinding(self, start, goal, obstacles):
        # Make sure the start and goal positions are rounded to two decimal places
        start = [round(s, 2) for s in start.tolist()]
        goal = [round(g, 2) for g in goal.tolist()]
        open_set = set([tuple(start)])
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.euclidean_distance(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score[x])
            if current == tuple(goal):
                return self.reconstruct_path(came_from, tuple(goal))

            open_set.remove(current)
            for neighbor in self.get_neighbors(torch.tensor(current, dtype=torch.float32), obstacles):
                # Convert neighbor to tuple and round to two decimal places
                neighbor_rounded = [round(n, 2) for n in neighbor.tolist()]
                neighbor_tuple = tuple(neighbor_rounded)
                tentative_g_score = g_score[current] + self.euclidean_distance(list(current), neighbor_rounded)

                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score[neighbor_tuple] = tentative_g_score + self.euclidean_distance(neighbor_rounded, goal)
                    if neighbor_tuple not in open_set:
                        open_set.add(neighbor_tuple)
                        #print(f"neighbor_tuple: {neighbor_tuple}")
            #print(f"len(open_set): {len(open_set)}, open_set: {open_set}")

        # If the open set is empty, no path was foundW
        return []

    def euclidean_distance(self, point1, point2):
        # Ensure point1 and point2 are torch.Tensor types
        point1 = torch.tensor(point1, dtype=torch.float32)
        point2 = torch.tensor(point2, dtype=torch.float32)
        
        # Calculate Euclidean distance
        return torch.norm(point1 - point2).item()

    def calculate_new_direction(self, agent, obstacle):
        # Calculate direction perpendicular to the current direction
        current_direction = agent.goal_pos - agent.state.pos[0]
        perpendicular_direction = torch.tensor([-current_direction[1], current_direction[0]])
        direction_45_degree = (current_direction + perpendicular_direction) / torch.sqrt(torch.tensor(2.0))

        # Check which side is open
        if not self.is_collision(agent.state.pos[0] + perpendicular_direction, obstacle):
            print(f"perpendicular_direction, {perpendicular_direction}")
            return direction_45_degree
        elif not self.is_collision(agent.state.pos[0] - perpendicular_direction, obstacle):
            print(f"minus perpendicular_direction, {-perpendicular_direction}")
            return -direction_45_degree
        else:
            # If obstacles are on both sides, return the original direction
            return current_direction

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        # Tensorize the path
        return [torch.tensor(point, dtype=torch.float32) for point in total_path]

    def get_neighbors(self, position, obstacles, step_size=0.5):
        # Simplified neighbor generation for explanation
        directions = [torch.tensor([step_size, 0]), torch.tensor([-step_size, 0]),
                    torch.tensor([0, step_size]), torch.tensor([0, -step_size])]
        neighbors = []
        for d in directions:
            np = position + d
            # Check if the neighbor is within the bounds and not colliding with obstacles
            if torch.all(np <= 2.5) and torch.all(np >= -2.5):
                if not self.is_collision(np, obstacles):
                    neighbors.append(np)
        return neighbors
    
    def point_to_line_segment_distance(self, point, segment_start, segment_end):
        # Make sure all parameters are torch.Tensor types
        segment_start = torch.tensor(segment_start).view(-1)
        segment_end = torch.tensor(segment_end).view(-1)
        point = torch.tensor(point).view(-1)

        # Calculate the vector from the segment start to end and from the segment start to the point
        line_vec = segment_end - segment_start
        point_vec = point - segment_start

        # Check if the input vectors are one-dimensional
        if line_vec.dim() != 1 or point_vec.dim() != 1:
            raise ValueError("line_vec and point_vec must be one-dimensional")

        # Project the point onto the line
        proj_length = torch.dot(point_vec, line_vec) / torch.dot(line_vec, line_vec)
        proj_length = torch.clamp(proj_length, 0, 1)  # Should be between 0 and 1

        # Calculate the nearest point on the line
        nearest = segment_start + proj_length * line_vec
        distance = torch.norm(point - nearest)

        return distance

    def is_collision(self, position, obstacles):
        for segment_start, segment_end in obstacles:
            distance = self.point_to_line_segment_distance(position[0], segment_start, segment_end)
            if distance < self.agents_radius + 0.01:  # Assume the radius of the agent is 0.03
                return True
        return False
    
    def plan_paths_for_agents(self):
        for agent in self.agents:
            print(f"Planning path for agent {agent.name}")
            if agent.current_line_segment is not None:  # A new task has been assigned
                start_point = agent.current_line_segment[0]
                # Use the A* algorithm to plan a path around obstacles to reach the start point of the task
                #print(f"Printed segments: {self.printed_segments}, Unprinted segments: {self.unprinted_segments}")
                obstacles = self.printed_segments
                #print(len(obstacles))
                path = self.a_star_pathfinding(agent.state.pos[0], start_point, obstacles)
                #print(path)
                if path:
                    # IF a path was found, set the goal position to the next point on the path
                    # This assumes that `path[0]` is the agent's current position and `path[1]` is the next target on the path
                    agent.goal_pos = path[1] if len(path) > 1 else start_point
                else:
                    # IF no path was found, set the goal position to the start point of the task
                    agent.goal_pos = start_point
        
    def calculate_new_direction(self, agent, obstacle):
        # Calculate direction perpendicular to the current direction
        current_direction = agent.goal_pos - agent.state.pos[0]
        perpendicular_direction = torch.tensor([-current_direction[1], current_direction[0]])
        
        # Calculate 45 degree rotation by averaging the current direction and perpendicular direction
        direction_45_degree = (current_direction + perpendicular_direction) / torch.sqrt(torch.tensor(2.0))
        
        # Check which side is open
        if not self.is_collision(agent.state.pos[0] + perpendicular_direction, obstacle):
            print(f"perpendicular_direction, {perpendicular_direction}")
            return direction_45_degree
        elif not self.is_collision(agent.state.pos[0] - perpendicular_direction, obstacle):
            print(f"minus perpendicular_direction, {-perpendicular_direction}")
            return -direction_45_degree
        else:
            # If obstacles are on both sides, return the original direction
            return current_direction
        
    def check_for_collisions(self, agent):
        # Check for intersections between the agent and printed segments
        for segment in self.printed_segments:
            if self.line_circle_intersection(segment[0], segment[1], agent.state.pos[0], self.agents_radius):
                print()
                # If an intersection is found, handle the collision
                # This could involve changing the agent's direction or setting a flag to trigger other responses
                # If an obstacle is detected, adjust the direction
                # Calculate the intersection point between the current movement direction and the obstacle, and determine the direction to bypass the obstacle
                new_direction = self.calculate_new_direction(agent, segment)
                move_distance = min(torch.norm(new_direction), 0.03)
                # Apply the new direction to the agent's position to avoid movement
                agent.state.pos += new_direction * move_distance + move_distance * (segment[1] - segment[0]) # Move the agent along the line segment
                print("Intersection detected!")
                
                break

    def line_circle_intersection(self, line_start, line_end, circle_center, circle_radius):
        # Transform all parameters to torch.Tensor
        line_start = torch.tensor(line_start, dtype=torch.float32)
        line_end = torch.tensor(line_end, dtype=torch.float32)
        circle_center = torch.tensor(circle_center, dtype=torch.float32)

        # Calculate the vector from the line start to end and from the line start to the circle center
        line_vec = line_end - line_start
        to_start_vec = line_start - circle_center

        # Use the quadratic formula to solve for the intersection points
        a = torch.dot(line_vec, line_vec)
        b = 2 * torch.dot(to_start_vec, line_vec)
        c = torch.dot(to_start_vec, to_start_vec) - circle_radius ** 2

        # Calculate the discriminant
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            # Calculate the two possible intersection points t1 and t2
            t1 = (-b - torch.sqrt(discriminant)) / (2 * a)
            t2 = (-b + torch.sqrt(discriminant)) / (2 * a)
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                return True  # If at least one intersection point is within the line segment, return True
        return False
    
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