# Multi-Agent Systems for Enhanced 3D Printing

## Introduction

This project explores the application of collaborative robotics in 3D printing to improve scalability and efficiency. The use of Multi-Agent Systems (MAS) allows for dynamic task allocation and real-time synchronization among autonomous robots. The primary aim is to develop and refine a cooperative 3D printing scenario using multiple robots that can efficiently collaborate on complex print tasks. Specifically, my goal is to create a scenario in which at least two robots cooperate to complete at least 20 randomly generated printing tasks in environments with and without initial obstacles, ensuring that there are no collisions between the robots. Additionally, I aim to evaluate the VMAS performance in environments with over 100 random print tasks and more than 3 agents working simultaneously to complete these tasks.

## Abstract

Despite advancements in 3D printing technology, the integration of MAS presents opportunities for further improvements. Base on the Vectorized Multi-Agent Simulator (VMAS), this research investigates the coordination of multiple robots in a simulated 3D printing environment. The results show that MAS can decrease printing times by up to 30% and increase scalability, effectively managing more complex tasks compared to traditional single-agent systems.

## Key Features

- **Robotic Coordination**: Enhanced synchronization and task allocation among multiple robots, provides new scenarios that support mulitple (2+) agent to collabrative print 10+ tasks.
- **Dynamic Task Allocation**: Using Auction algorithm to implement Real-time distribution of tasks based on current conditions and agent statuses.
- **Obstacle Avoidance**: Efficient navigation strategies to avoid collisions and maintain smooth operations between multiple agents.

## Methodology

The project utilizes VMAS for simulations, focusing on developing and testing algorithms for dynamic task allocation and real-time synchronization. The custom scenarios include path planning, obstacle avoidance, and collaborative task execution, providing insights into the scalability and efficiency of MAS in 3D printing.

## Results

The experimental results demonstrate significant improvements in operational times and scalability:
- **Printing Time Reduction**: Up to 30% decrease in printing times.
- **Scalability**: Effective management of more complex tasks with increased number of robots.
- **Efficiency**: Enhanced performance in terms of task allocation and synchronization.

## Future Work

Future research aims to implement these MAS in real-world settings to validate the simulation results. Additionally, exploring advanced algorithms and integrating machine learning techniques could further refine task allocation and synchronization processes, such as providing task distribution or 3D visualization of the scenarios.

## Acknowledgements

Special thanks to my supervisors, Dr. Yuzuko Nakamura and Prof. Simon Julier, for their invaluable guidance, and to Prof. Mirco Musolesi and Dr. Yunda Yan for their expert advice. Appreciation is also extended to Prof. Lewis Griffin for his extraordinary support during challenging times.

## License

This project is licensed under the MIT License.

## References

For more detailed information, please refer to  my thesis document and the [source code repository](https://github.com/YQU2020/MAS-3DPrinting-Collab). The scenario files is under (https://github.com/YQU2020/MAS-3DPrinting-Collab/tree/main/vmas/scenarios) and the script files (movement visualization) is under (https://github.com/YQU2020/MAS-3DPrinting-Collab/tree/main/vmas/examples)

---