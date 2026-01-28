# Temporal Difference Learning

Temporal Difference (TD) learning is a model-free reinforcement learning
approach that combines Monte Carlo sampling with dynamic programming
bootstrapping. Unlike Monte Carlo methods, TD learning updates value estimates
after each step rather than waiting for episode completion, enabling online
learning in continuing tasks.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **TD Error** | The difference between estimated and observed value: δ = r + γV(s') - V(s) |
| **Eligibility Traces** | Memory of recently visited states that receive credit for TD errors |
| **λ (Lambda)** | Trace decay parameter controlling the TD(0) to Monte Carlo spectrum |
| **Bootstrapping** | Using current value estimates to update other estimates |

## Environment

- **Python**: 3.9
- **NumPy**: 1.25.2
- **Gymnasium**: 0.29.1

## Tasks

| Task | File | Description |
|------|------|-------------|
| 0. Monte Carlo | [0-monte_carlo.py](./0-monte_carlo.py) | First-visit Monte Carlo for state value estimation using episode returns |
| 1. TD(λ) | [1-td_lambtha.py](./1-td_lambtha.py) | TD learning with eligibility traces for state value estimation |
| 2. SARSA(λ) | [2-sarsa_lambtha.py](./2-sarsa_lambtha.py) | On-policy TD control with eligibility traces for action-value estimation |

## Algorithm Comparison

| Algorithm | Update Timing | Trace Factor | Learns |
|-----------|---------------|--------------|--------|
| Monte Carlo | End of episode | N/A | V(s) |
| TD(λ) | Every step | λ ∈ [0,1] | V(s) |
| SARSA(λ) | Every step | λ ∈ [0,1] | Q(s,a) |

## Usage Example

```python
import gymnasium as gym
import numpy as np
from 0-monte_carlo import monte_carlo
from 1-td_lambtha import td_lambtha
from 2-sarsa_lambtha import sarsa_lambtha

# Create environment
env = gym.make('FrozenLake-v1', is_slippery=True)

# Initialize value/Q tables
V = np.zeros(env.observation_space.n)
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Define a simple policy
def random_policy(state):
    return np.random.randint(0, 4)

# Train using different algorithms
V_mc = monte_carlo(env, V.copy(), random_policy)
V_td = td_lambtha(env, V.copy(), random_policy, lambtha=0.9)
Q_sarsa = sarsa_lambtha(env, Q.copy(), lambtha=0.9)
```

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press.
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
