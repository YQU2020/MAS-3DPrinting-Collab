import time
from typing import Type

import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
from vmas.scenarios.twin_tracking import Scenario, SimplePolicy

def run_twin_tracking(
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
    
    
    iteration_since_first_two_actions = 0
    has_two_actions_occurred = False
    drop_time = 1
    
    for s in range(n_steps):
        step += 1
        print("********************************************************************************************************************")
        print(f"Step: {step}")
        
        # Check the current number of agents
        num_agents = len(env.world.agents)
        actions = [None] * num_agents
        print(f"Number of agents in environment: {num_agents}, Number of actions before: {len(actions)}")
        
        # Check if the number of actions is 2 and update the flag
        if len(actions) == 2 and not has_two_actions_occurred:
            has_two_actions_occurred = True

        # Increment the counter if the flag is true
        if has_two_actions_occurred:
            iteration_since_first_two_actions += 1
            
        main_agent_pos = env.world.agents[0].state.pos
        goal_pos = twin_scenario.goal_pos  # Access the goal position from the scenario instance
        print(f"Main agent position: {main_agent_pos[0]}, Goal position: {goal_pos}")
        
        distance_to_goal = torch.norm(main_agent_pos - goal_pos)
        print(f"Distance to goal: {distance_to_goal}")
        
        if distance_to_goal < 1e-6:  # some_threshold is a small value
            print("Main agent has reached the goal position. Terminating simulation.")
            break  # Terminate the simulation
        
        # Increment drop_time every "trail_interval" iterations after the first occurrence
        print(f"iteration_since_first_two_actions: {iteration_since_first_two_actions}")
        if iteration_since_first_two_actions > 0 and iteration_since_first_two_actions % trail_interval == 0:
            drop_time += 1
            
        while len(actions) > 1:
            actions.pop(0)  # Remove the static agent
            
        
        # Generate action only for the main agent
        main_agent = env.world.agents[0]
        main_agent_action = policy.compute_action(obs[0], u_range=main_agent.u_range)
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


        # Update the trail for twin_tracking scenario
        if scenario_name == "twin_tracking":
            twin_scenario.update_trail(env.agents[0])
            
        if s % trail_interval == 0:
            current_position = env.world.agents[0].state.pos.clone()
            if last_position is not None:
                twin_scenario.add_static_agent(last_position)
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
    run_twin_tracking(
        scenario_name="twin_tracking",
        heuristic=SimplePolicy,
        n_envs=300,
        n_steps=40,
        render=True,
        save_render=False,
    )
