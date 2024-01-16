import torch,time
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
from typing import Type
from vmas.scenarios.double_agent_trails_lines import Scenario, SimplePolicy


def run_double_agent_trails_lines(
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

    # Scenario specific variables
    policy = heuristic()

    # Initialize the environment
    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        **env_kwargs,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    obs = env.reset()
    total_reward = 0
    
    current_position = [None] * len(env.world.agents)
    last_position = [None]*len(env.world.agents)
    time_interval = 1  # Number of steps after which to add a static agent

    for s in range(n_steps):
        step += 1
        actions = [None] * len(env.world.agents)
        print("********************************************************************************************************************")
        print(f"Step: {step}")
        agent_pos = torch.zeros((len(env.world.agents), 2))
        
        for i, agent in enumerate(env.world.agents):
            actions[i] = policy.compute_action(obs[i], u_range=agent.u_range)
            obs[i][0][4:6] = env.scenario.nth_goal_pos[i]
            relative_goal_pos = obs[i][0]  # Assuming the relative goal position is part of the observation
            print(f"Agent {i} relative goal position: {relative_goal_pos}")
            goal_pos = env.scenario.nth_goal_pos[i]
            print(f"Agent {i} goal position: {goal_pos}")
            agent_pos[i] = obs[i][0][:2]
            if torch.norm(agent_pos[i] - goal_pos) < 0.05:
                print(f"Agent {i} has reached the goal. Terminating simulation.")
                break
        obs, rews, dones, info = env.step(actions)
        
        
        if s % time_interval == 0:
            for i, agent in enumerate(env.world.agents):
                current_position[i] = env.world.agents[i].state.pos.clone()
                # update the last position
                if last_position[i] is not None:
                    env.scenario.add_landmark(last_position[i][:][0], current_position[i])
                last_position[i] = current_position[i]   
        
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
    run_double_agent_trails_lines(
        scenario_name="double_agent_trails_lines",
        heuristic=SimplePolicy,
        n_envs=1,
        n_steps=40,
        render=True,
        save_render=False,
    )
