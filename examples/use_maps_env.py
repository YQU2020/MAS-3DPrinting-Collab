#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.
import os
import time

import numpy as np
import torch
from PIL import Image

from maps import make_env


def use_maps_env(render: bool = False):

    scenario_name = "football"

    # Scenario specific variables
    n_agents = 3

    num_envs = 1000
    continuous_actions = False
    device = "cuda"  # or cuda or any other torch device
    wrapped = False
    n_steps = 100

    simple_2d_action = (
        [0, 0.5] if continuous_actions else [3]
    )  # Sample action tell each agent to go down

    env = make_env(
        scenario_name=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        rllib_wrapped=wrapped,
        # Environment specific variables
        n_agents=n_agents,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    for s in range(n_steps):
        actions = []
        step += 1
        print(f"Step {step}")
        if wrapped:  # Rllib interface
            for i in range(num_envs):
                actions_per_env = []
                for j in range(n_agents):
                    actions_per_env.append(np.array(simple_2d_action))
                actions.append(actions_per_env)
            obs, rews, dones, info = env.vector_step(actions)
            if render:
                frame_list.append(
                    Image.fromarray(
                        env.try_render_at(
                            mode="rgb_array",
                            agent_index_focus=None,
                            visualize_when_rgb=True,
                        )  # Can give the camera an agent index to focus onß
                    )
                )
        else:  # Same as before, with faster MAPS interface
            for i in range(n_agents):
                actions.append(
                    torch.tensor(
                        simple_2d_action,
                        device=device,
                    ).repeat(num_envs, 1)
                )
            obs, rews, dones, info = env.step(actions)
            if render:
                frame_list.append(
                    Image.fromarray(
                        env.render(
                            mode="rgb_array",
                            agent_index_focus=None,
                            visualize_when_rgb=True,
                        )
                    )
                )  # Can give the camera an agent index to focus on

    if render:
        gif_name = scenario_name + ".gif"

        # Produce a gif
        frame_list[0].save(
            gif_name,
            save_all=True,
            append_images=frame_list[1:],
            duration=3,
            loop=0,
        )
        # Requires image magik to be installed to convert the gif in faster format
        os.system(f"convert -delay 1x30 -loop 0 {gif_name} {scenario_name}_fast.gif")

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device}"
        f" for {'wrapped' if wrapped else 'unwrapped'} simulator"
    )


if __name__ == "__main__":
    use_maps_env(render=False)
