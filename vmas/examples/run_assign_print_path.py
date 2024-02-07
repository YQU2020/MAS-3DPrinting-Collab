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
    #print("Unprinted Segments:", env.scenario.unprinted_segments)
    # Because reset_world_at will be called once before the first line segment is printed, 
    # it will pop off the first group of N line segments (N is the number of agents)
    env.scenario.unprinted_segments.insert(0, (env.scenario.print_path_points[0], env.scenario.print_path_points[1]))
    env.scenario.unprinted_segments.insert(1, (env.scenario.print_path_points[2], env.scenario.print_path_points[3]))
    
    # Add the line segments to the world, mannually
    for start_point, end_point in env.scenario.unprinted_segments:
            env.scenario.add_line_to_world(start_point, end_point, color=Color.GRAY)
            env.scenario.visulalize_endpoints(start_point, end_point)
    # Show the endpoints of the line segments      
           
    #env.scenario.assign_print_paths()
    # Update the goal position in the observation
    for s in range(n_steps):
        #print(f"Unprinted Segments: {env.scenario.unprinted_segments}")
        # Loop through each agent and execute the corresponding action
        actions = []
        for agent in env.world.agents:
            agent_observation = env.scenario.observation(agent)
            
            if agent.current_line_segment is not None and not agent.at_start:
                # Remember to only use the first element of the tensor
                if torch.norm((agent.state.pos - agent.current_line_segment[0])[0]) < 0.05: 
                    agent.at_start = True  # agent has reached the start of the line segment
                    agent.goal_pos = agent.current_line_segment[1]  # update the goal position
                    agent.is_printing = True
            agent_action = policy.compute_action(agent_observation, agent, u_range=agent.u_range)
            actions.append(agent_action)

        # Execute the environment step
        obs, rews, dones, info = env.step(actions)

        # Update the print path of each agent
        env.scenario.update_trail()
        
        # check if all line segments have been printed
        all_lines_printed = all(agent.current_line_segment is None for agent in env.world.agents)
        if all_lines_printed:
            print("All line segments have been printed. Terminating scenario.")
            break

        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward

        # check if all agents have completed their line segments
        if all(agent.current_line_segment is None for agent in env.world.agents):
            print("All agents have completed their segments. Terminating simulation. In step: ", s)
            break
        
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
        n_steps=1000,
        render=True,
        save_render=False,
    )
