import time
from typing import Type

import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video, Color
from vmas.scenarios.assign_print_path import Scenario, SimplePolicy

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
    policy = heuristic()

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

    #obs = env.reset() # This line is not needed because the line segments are already in the world
    
    total_reward = 0
    '''
    # Because reset_world_at will be called once before the first line segment is printed, 
    # it will pop off the first group of N line segments (N is the number of agents)
    env.scenario.unprinted_segments.insert(0, (env.scenario.print_path_points[0], env.scenario.print_path_points[1]))
    env.scenario.unprinted_segments.insert(1, (env.scenario.print_path_points[2], env.scenario.print_path_points[3]))
    '''
    # Add the line segments to the world, mannually
    for start_point, end_point in env.scenario.unprinted_segments:
            env.scenario.add_line_to_world(start_point, end_point, color=Color.GRAY)
            #env.scenario.visulalize_endpoints(start_point, end_point, color=Color.GRAY)
            
    env.scenario.execute_tasks_allocation()  # Allocate tasks to agents
           
    # Update the goal position in the observation
    for s in range(n_steps):

        # Loop through each agent and execute the corresponding action
        actions = []
        for agent in env.world.agents:
            agent_observation = env.scenario.observation(agent)
            #print(f"Agent {agent}  printing segment {agent.current_line_segment[0]}")
            if agent.current_line_segment is None:
                env.scenario.execute_tasks_allocation()
            if agent.current_line_segment is not None and not agent.at_start:
                # Remember to only use the first element of the tensor
                if torch.norm((agent.state.pos - agent.current_line_segment[0])[0]) < 0.01: 
                    agent.at_start = True  # agent has reached the start of the line segment
                    agent.goal_pos = agent.current_line_segment[1]  # update the goal position
                    agent.is_printing = True
            
            need_reassignment = False
            if env.scenario.is_agent_blocked(agent, env.scenario.printed_segments):
                print(f"Agent {agent.name} is blocked and will attempt to find a new path.")
                env.scenario.attempt_new_path(agent)
            
            if agent.current_line_segment:
                start_point, end_point = agent.current_line_segment
                print(f"Agent {agent.name} is printing segment {agent.current_line_segment} at {agent.state.pos[0]}, goalpoint is {agent.goal_pos}.New direction: {agent.state.vel[0]}. Distance to goal: {torch.norm(agent.state.pos[0] - end_point)}")

                if agent.is_printing and torch.norm(agent.state.pos[0] - end_point) < env.scenario.agents_radius: # Assume the radius of the agent
                    agent.is_printing = False  # Finished printing
                    agent.at_start = False  # Reset the flag
                    
                    # Agent finished printing, find a new safe position
                    safe_pos = env.scenario.find_safe_position(agent.state.pos[0], env.scenario.printed_segments)
                    if safe_pos is not None:
                        # If a safe position is found, update the goal position
                        agent.goal_pos = safe_pos
                    else:
                        # If no safe position is found, raise an exception
                        raise Exception("NO safe place founded!")
                    
                    agent.completed_tasks.append(agent.current_line_segment)  # Add the completed task to the list
                    agent.completed_total_dis += torch.norm(start_point - end_point)
                    
                    env.scenario.printed_segments.append(agent.current_line_segment)  # Add the completed task to the records
                    env.scenario.add_line_to_world(end_point, start_point, color=agent.color, collide = True)
                    print("=================================================================")
                    print(f"Agent {agent.name} finished segment{agent.current_line_segment}.")
                    agent.current_line_segment = None  # erase current line segment
                    agent.goal_pos = agent.state.pos[0]  # if no more line segments, stay at current position
                                
                    need_reassignment = True  # Flag that task reassignment is needed
                    
            agent.previus_pos = agent.state.pos
            if need_reassignment and any(agent.current_line_segment is None for agent in env.scenario.agents):
                env.scenario.execute_tasks_allocation()
            
            agent_action = policy.compute_action(agent_observation, agent, u_range=agent.u_range)
            actions.append(agent_action)
            #print(f"action: {len(actions)}")

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
        n_envs=400,
        n_steps=2000,
        render=True,
        save_render=False,
    )
