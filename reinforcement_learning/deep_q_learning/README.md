# Deep Q-Learning for Atari Breakout

This project implements a Deep Q-Network (DQN) agent that learns to play
Atari's Breakout using reinforcement learning.

## Overview

Deep Q-Learning combines Q-Learning with deep neural networks to approximate
the action-value function. Originally introduced by DeepMind in 2013, this
approach enables agents to learn optimal policies directly from high-dimensional
sensory inputs such as raw pixels.

### Key Components

- **Experience Replay**: Stores agent experiences in a replay buffer to break
  correlation between consecutive samples and improve learning stability.
- **Target Network**: Uses a separate network for generating Q-value targets,
  updated periodically to stabilize training.
- **Frame Stacking**: Stacks 4 consecutive frames to capture temporal dynamics
  and provide velocity information.
- **Reward Clipping**: Clips rewards to [-1, 1] range to normalize gradients.

## Environment

| Package     | Version  |
|-------------|----------|
| Python      | 3.9      |
| numpy       | 1.25.2   |
| gymnasium   | 0.29.1   |
| keras       | 2.15.0   |
| keras-rl2   | 1.0.4    |
| tensorflow  | 2.15.0   |
| Pillow      | 10.3.0   |
| h5py        | 3.11.0   |
| pygame      | latest   |

## Installation

```bash
pip install -r requirements.txt
```

## Files

| File                      | Description                                    |
|---------------------------|------------------------------------------------|
| [train.py](./train.py)    | Train a DQN agent on Atari Breakout            |
| [play.py](./play.py)      | Visualize a trained agent playing the game     |
| [requirements.txt](./requirements.txt) | Python dependencies              |

## Usage

### Training

Train the DQN agent for 1 million steps:

```bash
python3 train.py
```

This produces:
- `policy.h5`: Trained model weights
- `training_history.pkl`: Training metrics history
- `training_performance.png`: Reward progression plot

### Playing

Watch the trained agent play:

```bash
python3 play.py
```

Requires `policy.h5` from training. Displays 5 episodes via Pygame.

## Model Architecture

The CNN architecture follows Mnih et al. (2015):

| Layer       | Filters | Kernel | Stride | Activation |
|-------------|---------|--------|--------|------------|
| Conv2D      | 32      | 8x8    | 4      | ReLU       |
| Conv2D      | 64      | 4x4    | 2      | ReLU       |
| Conv2D      | 64      | 3x3    | 1      | ReLU       |
| Dense       | 512     | -      | -      | ReLU       |
| Dense (out) | 4       | -      | -      | Linear     |

Input: 4 stacked 84x84 grayscale frames.

## Hyperparameters

| Parameter               | Value     |
|-------------------------|-----------|
| Replay buffer size      | 1,000,000 |
| Frame stack (window)    | 4         |
| Discount factor (gamma) | 0.99      |
| Learning rate           | 0.00025   |
| Epsilon start           | 1.0       |
| Epsilon end             | 0.1       |
| Epsilon decay steps     | 1,000,000 |
| Target update frequency | 10,000    |
| Training interval       | 4         |
| Warmup steps            | 50,000    |
| Batch size              | 32        |

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement
  learning. *Nature*, 518(7540), 529-533.
- [keras-rl2 Documentation](https://github.com/wau/keras-rl2)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
