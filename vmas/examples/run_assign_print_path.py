import time
from typing import Type

import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
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
    step = 0
    obs = env.reset()
    total_reward = 0
    
    env.scenario.assign_print_paths()
    
    # Update the goal position in the observation
    for s in range(n_steps):
        print("********************************************************************************************************************")
        print(f"Step: {s + 1}")

        # Loop through each agent and execute the corresponding action
        actions = []
        for agent in env.world.agents:
            agent_observation = env.scenario.observation(agent)  
            agent_pos = agent_observation[:,:2]  
            current_goal_pos = env.scenario.print_path_points[env.scenario.current_segment_index]  # current_goal_pos是当前目标点

            # check the current and next goal points
            current_goal_pos = env.scenario.print_path_points[env.scenario.current_segment_index]
            next_goal_pos = env.scenario.print_path_points[min(env.scenario.current_segment_index + 1, len(env.scenario.print_path_points) - 1)]

            print(f"Step {s}: Agent Position: {agent_pos}, Current Goal Position: {current_goal_pos}, Next Goal Position: {next_goal_pos}")

            # calculate the agent's action
            agent_action = policy.compute_action(agent_observation, u_range=agent.u_range, current_goal_pos=current_goal_pos)
            actions.append(agent_action)

            # check if the agent has reached the current goal point
            if torch.norm(agent_pos - current_goal_pos) < 0.01:
                print(f"Agent has reached the current goal point: {current_goal_pos}.")
                # update the goal position to the next point in the print path
                if env.scenario.current_segment_index < len(env.scenario.print_path_points) - 1:
                    env.scenario.current_segment_index += 1
                    print(f"Moving to the next goal point: {next_goal_pos}.")
                else:
                    print("Agent has reached the final goal. Terminating simulation.")
                    break

        # Execute the environment step
        obs, rews, dones, info = env.step(actions)

        # chcek if all agents have completed their line segments
        all_done = all(agent.current_line_segment is None for agent in env.world.agents)
        if all_done:
            print("All agents have completed their line segments. Terminating simulation.")
            break

        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward

        # check if all agents have completed their line segments
        if all(agent.current_line_segment is None for agent in env.world.agents):
            print("All agents have completed their segments. Terminating simulation.")
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
        n_envs=300,
        n_steps=500,
        render=True,
        save_render=False,
    )
