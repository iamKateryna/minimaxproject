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

To play the human version of the game, run:
```bash
    python run.py
```

To modify map or objects placement, refer to `map_collection.py`, to modify Reward Machines type, refer to `rm_constants.py`. Make sure to choose correspondimg options. 

Note: MAP_2 RMs refer to RMs for maps with decorations, MAP_3 RMs refer to RMs for maps without decorations. MapCollection's map numbers have nothing to do with MAP_2 or MAP_3 in RM files, will be changed in the future. Sorryyyy

To run the experiments, run:
```bash
    python train_agents.py
```

Training configurations are in `training_configurations.py`.

To evaluate policies against other agents, run:
```bash
    python evaluate_agents.py
```
Evaluation configurations are in `evaluation_configurations.py`.


### Configurations details:
- `predator_prey: bool`, True - predator_prey option (where agent 2's task is to catch agent 1 while it delivers coffee to the office), False - both agents' task is to deliver coffee to the office.
- `map_type: MapType.SIMPLIFIED or MapType.BASE`, simplified - 6 by 9, base - 12 by 9
- `use_crms: (bool, bool)`, True - add crm, False - do not add crm
- `can_be_in_same_cell: bool`, True - can, False - cannot
- `coffee_type: CoffeeType.SINGLE or CoffeeType.UNLIMITED`, single - one coffee per coffee machine
- `agent_types = (AgentType, AgentType)`, AgentType.MINMAX or AgentType.QLEARNING or AgentType.RANDOM
- `total_timesteps: int`, total number of steps
- `max_episode_length: int`, upper booundary for the episode length (in timesteps)
- `print_freq: int`, print/log frequency of the progress

- `q_init: float`
- `learning_rate: float`
- `discount_factor: float`
- `exploration_rate: float`
- `exploration_decay_after: ExplorationDecay.EPISODE or ExplorationDecay.STEP`, specifies when the exploration rate should decay: either after each episode (ExplorationDecay.EPISODE) or after each step (ExplorationDecay.STEP)
- `n_episodes_for_decay: int`, the number of episodes over which to exponentially decay the exploration rate. If exploration_decay_after is set to ExplorationDecay.STEP, then the exploration rate is decayed exponentially over 70% of the total_timesteps.
- `group: str`, details, used to group runs in wandb
- `details: str`, details of the training to capture in .log file name and run name in wandb
