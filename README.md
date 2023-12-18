# Exploration of the Min-Max Q-learning Reward Machines in Competitive Two-Agent Zero-Sum Games

**Authors**: Kateryna Nekhomiazh and Bohdan Naida

This repository hosts our course project for CSC2542: Topics in Knowledge Representation and Reasoning at the University of Toronto. 

## Components
- **Two-Agent OfficeWorld Environment**: Based on the [PettingZoo](https://pettingzoo.farama.org/) library. [Explore here](https://github.com/iamKateryna/minimaxproject/tree/main/envs/grids).
- **RL Agents**: [Find the agents here](https://github.com/iamKateryna/minimaxproject/tree/main/rl_agents).
- **Reward Machines**: Includes Reward Machine implementation and environment wrappers. [More details](https://github.com/iamKateryna/minimaxproject/tree/main/reward_machines).

The foundational code for this project, particularly the single-agent OfficeWorld environment and Reward Machine implementation for the single-agent setting, was sourced from [Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning by Icarte et al](https://github.com/RodrigoToroIcarte/reward_machines/tree/master).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iamKateryna/minimaxproject.git
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
### Usage

To play the game, run:
```bash
    python human_mode.py
