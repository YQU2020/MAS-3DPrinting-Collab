import time
import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
from typing import Type
from vmas.scenarios.twin_tracking import Scenario

# Define the policy
class SimplePolicy(BaseHeuristicPolicy):
    def __init__(self, continuous_actions=True):
        super().__init__(continuous_actions)
        self.target_pos = None

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        if self.target_pos is None:
            self.target_pos = torch.rand(2) * 2 - 1

        pos_agent = observation[:, :2]
        action = torch.clamp(self.target_pos - pos_agent, min=-u_range, max=u_range)
        return action

# Run the heuristic with the custom scenario
def run_twin_tracking(
    scenario_class,
    heuristic: Type[BaseHeuristicPolicy] = SimplePolicy,
    n_steps: int = 200,
    n_envs: int = 2,
    render: bool = True,
    save_render: bool = False,
    device: str = "cpu",
):
    assert not (save_render and not render), "To save the video you have to render it"

    policy = heuristic()
    
    twin_scenario = scenario_class()
    # Initialize the environment with the custom scenario
    env = make_env(
        scenario=twin_scenario,
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

    # Start laying the trail
    twin_scenario.start_trail()

    for s in range(n_steps):
        step += 1
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
        obs, rews, dones, info = env.step(actions)
        twin_scenario.update_trail(env.agents[0])  # Update the trail based on the agent's movement
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

    # Stop laying the trail
    twin_scenario.stop_trail()

    total_time = time.time() - init_time
    if render and save_render:
        save_video("custom_scenario", frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward}"
    )

if __name__ == "__main__":

    run_twin_tracking(
        scenario_class= Scenario,
        heuristic=SimplePolicy,
        n_envs=2,
        n_steps=600,
        render=True,
        save_render=False,
    )
