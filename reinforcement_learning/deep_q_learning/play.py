#!/usr/bin/env python3
"""Display a game played by the agent trained on Atari's Breakout."""
from __future__ import division

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
import time
import pygame
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.util import *
from rl.core import Processor
from rl.callbacks import Callback


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


class PygameCallback(Callback):
    """Callback to display game visualization using Pygame.

    Renders each frame to a Pygame window during agent testing,
    allowing visual observation of agent performance.
    """

    def __init__(self, env, delay=0.02):
        """Initialize the Pygame display.

        Args:
            env: Gymnasium environment instance.
            delay: Delay in seconds between frames (default: 0.02).
        """
        self.env = env
        self.delay = delay
        pygame.init()
        self.screen = pygame.display.set_mode((420, 320))
        pygame.display.set_caption("Atari Breakout - DQN Agent")

    def on_action_end(self, action, logs={}):
        """Render frame after each action.

        Args:
            action: Action taken by the agent.
            logs: Dictionary of logs from the training process.
        """
        frame = self.env.render()
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (420, 320))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
                pygame.quit()

        time.sleep(self.delay)

    def on_episode_end(self, episode, logs={}):
        """Pause briefly between episodes.

        Args:
            episode: Episode number that ended.
            logs: Dictionary of logs from the training process.
        """
        pygame.time.wait(1000)


if __name__ == "__main__":
    # 1. CREATE ENV
    env = create_atari_environment('ALE/Breakout-v5')
    nb_actions = env.action_space.n

    # 2. BUILD MODEL
    window_length = 4
    input_shape = (84, 84)
    model = build_model(window_length, input_shape, nb_actions)

    # 3. LOAD TRAINED WEIGHTS
    model.load_weights('policy.h5')

    # 4. CONFIGURE AGENT
    memory = SequentialMemory(limit=1000000,
                              window_length=window_length)
    processor = AtariProcessor()
    policy = GreedyQPolicy()

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
    dqn.compile(optimizer='adam',
                metrics=['mae'])

    # 5. TEST AGENT
    pygame_callback = PygameCallback(env, delay=0.02)
    scores = dqn.test(env, nb_episodes=5,
                      visualize=False,
                      callbacks=[pygame_callback])

    # 6. DISPLAY RESULT
    print('Average score over 5 test episodes:',
          np.mean(scores.history['episode_reward']))

    # 7. CLOSE ENV AND PYGAME
    env.close()
    pygame.quit()
