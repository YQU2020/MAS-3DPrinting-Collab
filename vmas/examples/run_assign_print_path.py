import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
sys.path.append('D:\\VMAS_yqu_implementation') # Add the abstract path of the VMAS implementation HERE
    
from vmas.scenarios.assign_print_path import Scenario, SimplePolicy
import time
from typing import Type

import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video, Color


def run_assign_print_path(
    scenario_name: str,
    heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 32,
    env_kwargs: dict = {},
    render: bool = False,
    save_render: bool = False,
    device: str = "cpu",
):

    assert not (save_render and not render), "To save the video you have to render it"

    # Scenario specific variables
    policy = SimplePolicy(scenario=Scenario)

    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()

    obs = env.reset() # This line is not needed because the line segments are already in the world
    
    total_reward = 0

    # Add the line segments to the world, mannually
    for start_point, end_point in env.scenario.unprinted_segments:
            env.scenario.add_line_to_world(start_point, end_point, color=Color.GRAY)
            #env.scenario.visulalize_endpoints(start_point, end_point, color=Color.GRAY)
            
    env.scenario.execute_tasks_allocation()  # Allocate tasks to agents
           
    # Update the goal position in the observation
    for s in range(n_steps):
        #print(f"Step {s} of {n_steps}")
        # Loop through each agent and execute the corresponding action
        actions = []
        for agent in env.world.agents:
            # If there are unprinted segments, check if the agent is blocked or has encountered an obstacle
            # If not, that means the agent is stopped and has no path needed
            if env.scenario.unprinted_segments:
                # Check if the agent is blocked or has encountered an obstacle
                if env.scenario.is_path_obstructed(agent.state.pos[0], agent.goal_pos, obstacles=env.scenario.printed_segments) and len(agent.path) == 0 and not agent.path:
                    # IF the agent is blocked and has no path, find a new path
                    print(f"Agent {agent.name} is blocked and will attempt to find a new path.")
                    env.scenario.on_collision_detected(agent)
                #elif len(agent.path) > 0:
                    # IF the agent has a path, follow it
                    #print(f"Agent {agent.name} is following its path.")
                    # --Here can add extra code to check if the agent is moving in the right direction--
                #else:
                    #print(f"Agent {agent.name} is moving freely without the need for a new path.")
                    # If the agent is not blocked and has no path, move freely
                agent_observation = env.scenario.observation(agent)
            
            if agent.current_line_segment is None:
                env.scenario.execute_tasks_allocation()     # Allocate tasks to agents if any agent has no task
                
                if not agent.at_start:       # Remember to only use the first element of the tensor
                    
                    if torch.norm((agent.state.pos - agent.current_line_segment[0])[0]) < env.scenario.agents_radius: 
                        print("*****************************************************************")
                        print(f"Agent {agent.name} has reached the start of the line segment {agent.current_line_segment}.")
                        agent.at_start = True  # agent has reached the start of the line segment
                        agent.goal_pos = agent.current_line_segment[1]  # update the goal position
                        agent.is_printing = True
                        
            need_reassignment = False
                    
            if agent.current_line_segment:
                env.scenario.on_collision_detected(agent)
            else:
                print(f"Agent {agent.name} does not have a current task assigned.")
                            
            if agent.current_line_segment:
                start_point, end_point = agent.current_line_segment
                if agent.is_printing and torch.norm(agent.state.pos[0] - end_point) < env.scenario.agents_radius: # Assume the radius of the agent
                    agent.is_printing = False  # Finished printing
                    agent.at_start = False  # Reset the flag
                    agent.completed_tasks.append(agent.current_line_segment)  # Add the completed task to the list
                    agent.completed_total_dis += torch.norm(start_point - end_point)
                    
                    env.scenario.printed_segments.append(agent.current_line_segment)  # Add the completed task to the records
                    env.scenario.add_line_to_world(end_point, start_point, color=agent.color, collide = True)
                    agent.current_line_segment = None  # erase current line segment
                    agent.goal_pos = agent.state.pos[0]  # if no more line segments, stay at current position
                                
                    need_reassignment = True  # Flag that task reassignment is needed

            #print("=================================================================")
            #print(f"Agent {agent.name} is printing segment {agent.current_line_segment} at {agent.state.pos[0]}, goalpoint is {agent.goal_pos}.New direction: {agent.state.vel[0]}. Distance to goal: {torch.norm(agent.state.pos[0] - agent.goal_pos)}. ")
            #print(f"Agent {agent.name} is print status: {agent.is_printing}, path: {agent.path}, task queue: {agent.task_queue}, completed tasks: {agent.completed_tasks}, current line segment: {agent.current_line_segment}.")
            agent_index = env.world.agents.index(agent)  
            agent_observation = obs[agent_index]
            action = policy.compute_action(agent_observation, agent, u_range=agent.u_range)
            actions.append(action)

        # Execute the environment step
        obs, rews, dones, info = env.step(actions)

        
        # check if all line segments have been printed
        all_lines_printed = all(agent.current_line_segment is None and not agent.is_printing for agent in env.world.agents)
        if all_lines_printed:
            print("All line segments have been printed. Terminating scenario.")
            break

        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward
        
        if render:
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )

    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward}"
    )

if __name__ == "__main__":
    run_assign_print_path(
        scenario_name="assign_print_path",
        heuristic=SimplePolicy,
        n_envs=200,
        n_steps=2000,
        render=True,
        save_render=False,
    )
