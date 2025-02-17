# Task Offloading with Deep Reinforcement Learning

## System Overview
This project implements a distributed task offloading system using deep reinforcement learning for edge computing environments.

## Key Components

### Environment
- Simulates distributed computing environment
- Manages task scheduling and resource allocation
- Tracks system performance metrics

### Decision Makers 
- Deep reinforcement learning agents
- Rule-based strategies
- Baseline comparison methods

### Bookkeeping
- Manages experiment logging and metrics tracking
- Handles checkpointing and resuming of training
- Generates performance visualizations and plots

### Championship
- Evaluates and compares different agent strategies
- Maintains top performing models
- Enables champion model reuse

### Meta Plots
- Generates comparative visualizations
- Analyzes performance trends
- Creates publication-ready figures

## Features
- PyTorch-based deep learning implementation
- Extensive logging and visualization tools
- Flexible configuration system
- Multiple decision-making strategies



## Example Usage

### Basic Training Run

1. Configure hyperparameters in a JSON file running
```
python hyperparameters

```


## Run training
```
python main.py \
    --log_folder logs \
    --epochs 1000 \
    --average_window 500
```
## Resume Training
```
python main.py \
    --log_folder logs \
    --resume_run run_1 \
    --epochs 500
```

# Different Versions

In order to run the different versions, you just need to modify the hyperparameters file,
## Cooler 
just run it with the ```delay_to_energy_ratio ```  hyperparameter set to 0. Meaning only delay will be taken into account in the reward.
## Priority 
In order to run pdppnet, with **priority**, one needs to change the priority distributions from the hyperparameters.
## Deccoffe.
Modify the  ```delay_to_energy_ratio ```   to be non-zero.