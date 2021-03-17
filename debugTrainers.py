import json
from typing import Any, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tqdm
from kaggle_environments import evaluate, make, utils
from tensorflow import keras
from tensorflow.keras import layers
from tf_agents.environments.random_py_environment import RewardFn

from ActorCritic import ActorCritic
from C4_Helpers import convert_state, render
from MCTS import SearchTree
from Training import C4Trainer

# PLAN
# Create a function which takes in functions and returns a function, specifically the train_step funciton.
# Tests could then proceed by specifying which functions to pass in, i,e which version of train_step you want

class Test_Overfitting(C4Trainer):
    def __init__(self, params, config, states, rewards, input_shape=(1,6,7,1)):
        super().__init__(params, config, input_shape)
        self.states = states
        self.rewards = rewards
    
    def run_episode(self, player=0):
        index = np.random.randint(len(self.states))
        state = self.states[index]
        reward = self.rewards[index]

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        tau = self.params["tau"]

        action_vals, state_val = self.model(state)

        action, prob = self.get_action(state, action_vals, tau)

        for t in tf.range(1):

            rewards = rewards.write(t, reward)

            values = values.write(t, tf.squeeze(state_val))

            action_probs = action_probs.write(t, prob)

        # convert tensor array to tensor
        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards
    
    """ def compute_loss(self, action_probs: tf.Tensor,  values: tf.Tensor,  returns: tf.Tensor, loss_function: tf.keras.losses.Loss) -> tf.Tensor:
        Computes the combined actor-critic loss.

        critic_loss = loss_function(values, returns)

        # total loss is actor-critic loss
        return critic_loss """


