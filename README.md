# **Data Center Energy Optimization with Reinforcement Learning**

This repository contains a reinforcement learning (RL) framework for optimizing energy consumption in a data center with limited storage and flexible energy consumption. The project utilizes a **Q-learning agent (QAgent)** and a **Heuristic agent** to learn and evaluate energy optimization strategies based on a simulated environment.

## **Project Overview**

The project implements an RL-based approach for energy management in a data center. It features:

- A **Q-learning agent** (`QAgent`) that learns an optimal policy through experience.
- A **Heuristic agent** (`Heuristic`) that follows predefined rules.
- A **simulation environment** (`DataCenterEnv`) to model energy consumption and price variations.
- **Training and evaluation pipelines** to test agent performance.

## **Project Structure**

```
├── agent.py           # Implements the Q-learning agent and heuristic agent
├── main.py            # Handles training and validation of agents
├── env.py             # Defines the data center environment (not provided here)
├── Data/              # Contains training and validation datasets
│   ├── train-cleaned-features.xlsx
│   ├── validate.xlsx
├── QTables/           # Stores learned Q-tables for reinforcement learning
├── Results/           # Stores evaluation results
├── Plots/             # Stores generated plots
├── utils.py           # Utility functions (argument parsing)
```

## **Installation & Requirements**

Ensure you have the required dependencies installed:

```bash
pip install numpy pandas gym matplotlib openpyxl
```

## **How to Use**

### **1. Train the Q-learning Agent**

Run the following command to train the QAgent:

```bash
python main.py --mode train --agent QAgent --agent_nr 1 --show_plot True --save_plot True --use_rewardshaping True
```

- `--agent_nr` specifies the agent ID.
- `--use_rewardshaping` enables reward shaping for better learning.
- `--show_plot` and `--save_plot` control result visualization.

### **2. Evaluate the Agent**

After training, you can evaluate the agent:

```bash
python main.py --mode validate --agent QAgent --agent_nr 1 --show_plot True
```

### **3. Run the Heuristic Agent**

To compare with a heuristic approach:

```bash
python main.py --mode validate --agent Heuristic --agent_nr 1 --show_plot True
```

## **Agent Details**

### **QAgent (Reinforcement Learning)**

- Uses **Q-learning** to optimize energy management.
- Discretizes the state space based on:
  - Storage level
  - Electricity price
  - Hour of the day
  - Day of the week
  - Season
- Implements **reward shaping** to guide learning.
- Stores learned policies in a **Q-table**.

### **Heuristic Agent**

- Uses a **predefined rule-based approach** for decision-making.
- Doesn't require training.

## **Customization**

Modify hyperparameters in `main.py`:

- `small_reward`, `large_reward`: Adjust reward shaping.
- `learning_rate`: Controls learning updates.
- `n_episodes`: Sets the number of training iterations.
- `state_bin_size`: Changes state discretization granularity.

## **Results & Visualization**

- Training generates **Q-tables** stored in `QTables/`
- Evaluations produce **action profiles** in `Results/`
- **Plots** visualizing agent performance are saved in `Plots/`

## **Troubleshooting**

- Ensure **data files** (`train-cleaned-features.xlsx`, `validate.xlsx`) are present in the `Data/` folder.
- Check **dependencies** (`numpy`, `pandas`, `gym`, etc.).
- Use `--show_plot True` to debug agent behavior visually.

## **Future Improvements**

- Implement **Deep Q-Networks (DQN)** for improved performance.
- Incorporate **real-time price forecasting** for better decisions.
- Optimize hyperparameters using **Bayesian optimization**.

## **License**

This project is open-source and available for educational and research purposes.