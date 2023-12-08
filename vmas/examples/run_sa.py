import torch, time
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.utils import save_video
from typing import Type
from vmas.scenarios.single_agent_scenario import Scenario, SimplePolicy

def run_single_agent_scenario(
    scenario_name: str,
    heuristic: Type[BaseHeuristicPolicy] = SimplePolicy,
    n_steps: int = 200,
    n_envs: int = 1,
    render: bool = True,
    save_render: bool = False,
    device: str = "cpu",
):
    assert not (save_render and not render), "To save the video you have to render it"

    policy = heuristic()
    
    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
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
        
        # Extracting specific parts of the observation
        agent_pos = obs[0][0][:2]  # Agent's current position
        agent_vel = obs[0][0][2:4]  # Agent's current velocity
        relative_goal_pos = obs[0][0][4:6]  # Relative position to the goal

        # Assuming the goal position is static and known
        goal_pos = env.scenario.goal_pos

        print("********************************************************************************************************************")
        print(f"Step: {step}, Agent Position: {agent_pos}, Goal Position: {goal_pos}, Agent Velocity: {agent_vel}, relative_goal_pos: {relative_goal_pos}")
        
        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward
        
        # Check if the agent has reached the goal
        if dones[0]:
            print(f"Agent has reached the goal at step {step}")
            break

        #if render:
        #    frame_list.append(env.render(mode="rgb_array"))

    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(f"It took: {total_time}s for {step} steps on device {device}\n"
          f"The average total reward was {total_reward}")

if __name__ == "__main__":
    run_single_agent_scenario(
        scenario_name="single_agent_scenario",
        heuristic=SimplePolicy,
        n_steps=200,
        render=True,
        save_render=False,
    )
