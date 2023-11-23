import torch
from vmas import make_env, render_interactively
from single_agent_scenario import Scenario, SimplePolicy

def main():
    # Initialize the environment with the Single Agent Scenario
    env = make_env(scenario=Scenario(), num_envs=1, device="cpu")
    obs = env.reset()

    # Initialize the own policy
    policy = SimplePolicy()

    # Main loop 
    running = True
    while running:
        try:
            # Compute action based on the current observation
            action = policy.compute_action(obs, env.action_space.high[0])

            # Take a step in the environment
            obs, rewards, done, info = env.step(action)

            # Render the current state of the environment
            render_interactively(env)

            # Check if the episode is done
            if done.any():
                obs = env.reset()

        except KeyboardInterrupt:
            # Stop t
            running = False

if __name__ == "__main__":
    main()

