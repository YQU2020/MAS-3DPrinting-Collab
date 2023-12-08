import time
import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
from typing import Type
from vmas.scenarios.single_agent_scenario import Scenario, SimplePolicy

def run_single_agent(
    scenario_name: str,
    heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 1,
    env_kwargs: dict = {},
    render: bool = True,
    save_render: bool = False,
    device: str = "cpu",
):
    assert not (save_render and not render), "To save the video you have to render it"

    policy = heuristic()
    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,a
        device=device,
        continuous_actions=True,
        wrapper=None,
        **env_kwargs,
    )

    frame_list = []
    init_time = time.time()
    step = 0
    obs = env.reset()
    total_reward = 0

    for s in range(n_steps):
        step += 1
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)

        obs, rews, dones, info = env.step(actions)
        obs[0][0][4:6] = env.scenario.goal_pos.unsqueeze(0)

        # Assuming the goal position is static and known

        print("********************************************************************************************************************")
        print(f"Step: {step}, Agent Position: {obs[0][0][:2]}, Goal Position: {env.scenario.goal_landmark.state.pos[0]}, Agent Velocity: {obs[0][0][2:4]}, relative_goal_pos: {obs[0][0][4:6]}")

        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward
        print(f"reward: {mean_global_reward}, total_reward: {total_reward}")
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
    run_single_agent(
        scenario_name="single_agent_scenario",
        heuristic=SimplePolicy,
        n_envs=1,
        n_steps=100,
        render=True,
        save_render=False,
    )
