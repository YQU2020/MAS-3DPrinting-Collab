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

    for s in range(n_steps):
        step += 1
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
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
        n_steps=200,
        render=True,
        save_render=False,
    )
