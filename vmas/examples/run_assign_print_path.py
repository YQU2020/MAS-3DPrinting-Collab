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
            # Only try to assign new tasks if the agent is not currently executing a task and the task queue is empty
            if len(agent.path) > 0:
                agent.goal_pos = agent.path.pop(0)  
                agent_index = env.world.agents.index(agent)  
                agent_observation = obs[agent_index]
                action = policy.compute_action(agent_observation, agent, u_range=agent.u_range)
                actions.append(action)
                continue
            if agent.current_line_segment is None or len(agent.task_queue) == 0:
                env.scenario.assign_tasks_based_on_bids()

            if agent.current_line_segment is None and len(agent.task_queue) > 0:
                next_task = agent.task_queue.pop(0)
                if not env.scenario.is_task_being_executed_by_another_agent(next_task, agent):
                    agent.current_line_segment = next_task
                    agent.is_printing = False  # ensure that the agent is not printing
                    # Attempt to find a path to the next task
                    agent.path = env.scenario.a_star_pathfinding(agent.state.pos[0], next_task[0], env.scenario.all_segments)
                    
                    if agent.path:
                        agent.goal_pos = agent.path[0]  
                    else:
                        agent.goal_pos = agent.current_line_segment[0]  # if no path is found, set the goal to the start of the line segment
                    
            # If the agent is at the start of the line segment, start printing    
            if agent.current_line_segment and torch.norm(agent.state.pos[0] - agent.current_line_segment[0]) < env.scenario.agents_radius and not agent.is_printing:
                agent.is_printing = True
                agent.goal_pos = agent.current_line_segment[1]
                    
            if agent.is_printing and torch.norm(agent.state.pos[0] - agent.current_line_segment[1]) < env.scenario.agents_radius:
                # Mark the task as completed, clear the current task, and prepare to accept the next task
                agent.is_printing = False
                agent.completed_tasks.append(agent.current_line_segment)  # Add the completed task to the list
                agent.completed_total_dis += torch.norm(agent.current_line_segment[0] - agent.current_line_segment[1])  # Add the completed task distance to the total distance
                env.scenario.printed_segments.append(agent.current_line_segment)
                env.scenario.add_line_to_world(agent.current_line_segment[0], agent.current_line_segment[1], color=agent.color)
                agent.current_line_segment = None
                
                # Check if the agent has completed all tasks
                if len(agent.task_queue) > 0:
                    next_task = agent.task_queue.pop(0)
                    agent.current_line_segment = next_task
                    agent.path = env.scenario.a_star_pathfinding(agent.state.pos[0], next_task[0], env.scenario.all_segments)
                    if agent.path:
                        agent.goal_pos = agent.path[0]  
                        print(f"Agent {agent.name} assigned to segment {next_task}.")
                        print(f"Agent {agent.name} is printing segment {agent.current_line_segment} at {agent.state.pos[0]}, goalpoint is {agent.goal_pos}.New direction: {agent.state.vel[0]}. Distance to goal: {torch.norm(agent.state.pos[0] - agent.goal_pos)}. ")
                    else:
                        agent.goal_pos = next_task[0]  
                    agent.is_printing = False  
                else:
                    agent.current_line_segment = None
                    agent.goal_pos = agent.state.pos[0] 

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
