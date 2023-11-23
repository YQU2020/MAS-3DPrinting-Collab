import torch
from vmas.simulator import VMASEnv
from vmas.scenarios.single_agent_scenario import SingleAgentScenario


def main():
    device = "cpu"
    env = VMASEnv(SingleAgentScenario, device=device, render=True, batch_dim=1)

    for _ in range(1000):  # Number of steps to simulate
        action = torch.tensor([[0.5, 0.5]], device=device)  # Example action, adjust as needed
        env.step(action)

        if env.done().any():
            env.reset()

if __name__ == "__main__":
    main()
