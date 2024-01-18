import time
from typing import Type

import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
from vmas.scenarios.print_path import Scenario, SimplePolicy

def run_print_path(
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

    # Initialize the trail
    if scenario_name == "twin_tracking":
        twin_scenario = Scenario()
        twin_scenario._world = env.world  # Set the world attribute
        twin_scenario.make_world(n_envs, device)  # Explicitly call make_world
        twin_scenario.start_trail()
    
    last_position = None
    trail_interval = 10  # Number of steps after which to add a static agent
    
    # Update the goal position in the observation
    for s in range(n_steps):
        step += 1
        print("********************************************************************************************************************")
        print(f"Step: {step}")
        
        # Check the current number of agents
        num_agents = len(env.world.agents)
        actions = [None] * num_agents
        print(f"Number of agents in environment: {num_agents}, Number of actions before: {len(actions)}")
        
        actions = [torch.tensor([0.0])] * len(obs)
        
        # Update the goal position in the observation
        for i in range(len(obs)):
            obs[i][0][4:6] = env.scenario.goal_pos.unsqueeze(0)

        # Check if agent has reached the goal
        goal_pos = env.scenario.print_path_points[env.scenario.current_segment_index]  #  From print_path_points get goal_pos directly

        agent_pos = obs[0][0][:2]
        print(f"Step {s}: Agent Position: {agent_pos}, Current Goal Position: {env.scenario.goal_pos}")
        
        # Update the trail，检查agent是否到达当前目标点并更新到下一个目标点 
        env.scenario.update_trail(env.world.agents[0])
        
        print(f"Current Segment Index: {env.scenario.current_segment_index}, Goal Pos: {env.scenario.goal_pos}")

        # GET CURRENT AND NEXT GOAL POSITION
        current_goal_pos = env.scenario.print_path_points[env.scenario.current_segment_index]
        next_goal_pos = env.scenario.print_path_points[min(env.scenario.current_segment_index + 1, len(env.scenario.print_path_points) - 1)]
        
        
        print(f"Step {s}: Agent Position: {agent_pos}, Current Goal Position: {current_goal_pos}, Next Goal Position: {next_goal_pos}")
        
        # check if agent has reached the current goal point
        if torch.norm(agent_pos - goal_pos) < 0.01:
            print(f"Agent has reached the current goal point: {goal_pos}.")
            
            # update the goal position to the next point in the print path
            if env.scenario.current_segment_index < len(env.scenario.print_path_points) - 1:
                env.scenario.current_segment_index += 1
                env.scenario.goal_pos = next_goal_pos  # 获取新的目标点 SET GLOBAL GOAL POSITION,(NOT GOAL_POS IN AGENT)
                print(f"Moving to the next goal point: {goal_pos}.")
            else:
                print("Agent has reached the final goal. Terminating simulation.")
                break

    
        while len(actions) > 1:
            actions.pop(0)  # Remove the static agent
            
        # Generate action only for the main agent
        main_agent = env.world.agents[0]
        main_agent_action = policy.compute_action(obs[0], u_range=main_agent.u_range, current_goal_pos=env.scenario.goal_pos)
        print(f"Main agent action: {main_agent_action[0]}, u_range: {main_agent.u_range}")
        for i, agent in enumerate(env.world.agents):
            # Initialize the action tensor with the correct shape
            if agent.action.u is None:
                agent.action.u = torch.zeros((300, 2), dtype=torch.float32)
            #print(f"i is :  {i}, i-drop_time is: {i-drop_time}")
            #print(f"Number of dropstime after: {drop_time}")
            # Assign action only to the main agent
            if i == 0:
                actions[i] = main_agent_action
            else:
                actions[0] = torch.zeros_like(agent.action.u)
                

        print(f"Number of actions being generated: {len(actions)}")
        obs, rews, dones, info = env.step(actions)
        
        if s % trail_interval == 0:
            current_position = env.world.agents[0].state.pos.clone()
            if last_position is not None:
                print(last_position.shape)
                env.scenario.add_landmark(last_position[:][0])
            #    twin_scenario.add_static_agent(last_position)
            last_position = current_position

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

    # Stop the trail
    if scenario_name == "twin_tracking":
        twin_scenario.stop_trail()

    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward}"
    )

if __name__ == "__main__":
    run_print_path(
        scenario_name="print_path",
        heuristic=SimplePolicy,
        n_envs=300,
        n_steps=500,
        render=True,
        save_render=False,
    )
