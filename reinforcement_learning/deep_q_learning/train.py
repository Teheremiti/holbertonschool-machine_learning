#!/usr/bin/env python3
"""Train an agent to play Atari's Breakout using Deep Q-Learning."""
from __future__ import division
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
import matplotlib.pyplot as plt
import pickle

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.util import *
from rl.core import Processor

# Training hyperparameters
seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 10000
max_episodes = 10


class CompatibilityWrapper(gym.Wrapper):
    """Wrapper ensuring compatibility with older gym API.

    Converts the new gymnasium (observation, reward, terminated, truncated,
    info) step return format to the older (observation, reward, done, info)
    format expected by keras-rl2.
    """

    def step(self, action):
        """Execute one environment step.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: Contains (observation, reward, done, info) where:
                - observation: Environment observation after action.
                - reward: Reward obtained from the action.
                - done: Boolean indicating episode termination.
                - info: Additional environment information.
        """
        observation, reward, terminated, truncated, info = (
            self.env.step(action))
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """Reset the environment.

        Args:
            **kwargs: Additional arguments passed to underlying reset.

        Returns:
            observation: Initial observation of the environment.
        """
        observation, info = self.env.reset(**kwargs)
        return observation


def create_atari_environment(env_name):
    """Create and configure an Atari environment for reinforcement learning.

    Applies standard Atari preprocessing including frame resizing to 84x84,
    grayscale conversion, frame skipping, and random no-ops at episode start.

    Args:
        env_name: Name of the Atari environment (e.g., 'ALE/Breakout-v5').

    Returns:
        gym.Env: Configured and wrapped Atari environment.
    """
    env = gym.make(env_name, render_mode='rgb_array')
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=1,
        noop_max=30
    )
    env = CompatibilityWrapper(env)
    return env


def build_model(window_length, shape, actions):
    """Build a CNN model for Deep Q-Learning.

    Constructs a convolutional neural network following the DQN architecture
    from Mnih et al. (2015) for processing stacked Atari frames.

    Args:
        window_length: Number of frames to stack as input.
        shape: Shape of input image as (height, width).
        actions: Number of possible actions in the environment.

    Returns:
        keras.models.Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length,) + shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


class AtariProcessor(Processor):
    """Processor for Atari observations and rewards.

    Handles preprocessing of observations, state batches, and rewards
    before passing them to the DQN agent.
    """

    def process_observation(self, observation):
        """Convert observation to uint8 numpy array.

        Args:
            observation: Raw observation from environment (may be tuple).

        Returns:
            np.ndarray: Processed observation as uint8 array.
        """
        if isinstance(observation, tuple):
            observation = observation[0]
        return np.array(observation, dtype='uint8')

    def process_state_batch(self, batch):
        """Normalize pixel values in a batch of states.

        Args:
            batch: Batch of states with pixel values in [0, 255].

        Returns:
            np.ndarray: Normalized batch with values in [0.0, 1.0].
        """
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        """Clip reward to [-1, 1] range.

        Args:
            reward: Raw reward from the environment.

        Returns:
            float: Clipped reward value.
        """
        return np.clip(reward, -1., 1.)


if __name__ == "__main__":
    # 1. CREATE ENV
    env = create_atari_environment('ALE/Breakout-v5')
    observation = env.reset()

    plt.imshow(observation, cmap='gray')
    plt.title("Initial Observation")
    plt.axis('off')
    plt.show()

    nb_actions = env.action_space.n

    # 2. BUILD MODEL
    window_length = 4
    model = build_model(window_length, observation.shape, nb_actions)

    # 3. DEFINE AGENT
    memory = SequentialMemory(limit=1000000,
                              window_length=window_length)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=1000000
    )

    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)
    dqn.compile(Adam(learning_rate=0.00025),
                metrics=['mae'])

    # 4. TRAIN MODEL
    history = dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

    # 5. SAVE MODEL & HISTORY
    dqn.save_weights('policy.h5', overwrite=True)

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['episode_reward'])
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.savefig('training_performance.png')
    plt.show()

    # 6. TEST
    test_env = gym.make('ALE/Breakout-v5', render_mode='human')
    test_env = AtariPreprocessing(test_env,
                                  screen_size=84,
                                  grayscale_obs=True,
                                  frame_skip=4,
                                  noop_max=30)

    scores = dqn.test(test_env,
                      nb_episodes=10,
                      visualize=True)
    print('Average score over 10 test episodes:',
          np.mean(scores.history['episode_reward']))

    # 7. Close env
    env.close()
