import json
from typing import Any, List, Sequence, Tuple
import copy

import numpy as np
import tensorflow as tf
import tqdm
from kaggle_environments import evaluate, make, utils
from tensorflow import keras
from tensorflow.keras import layers
from tf_agents.environments.random_py_environment import RewardFn

from ActorCritic import ActorCritic
from C4_Helpers import convert_state, render, legal
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
        reward = tf.constant(self.rewards[index], dtype=tf.float32)

        render(state, [1, -1, 0])
        self.tree.set_state(state)

        initial_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        final_log_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        tau = self.params["tau"]

        # maximum number of steps
        for t in tf.range(1):
            action_values, state_val = self.call_model(tf.convert_to_tensor(state.reshape(1, self.rows, self.columns, 1), dtype=tf.float32))

            print('Action_values: ', action_values)
            print('State_val: ', state_val)

            init_probs = self.get_probs(action_values)
        
            action, probs, terminal, win = self.tree.MCTS()

            final_log_probs = self.get_log_probs(probs)

            # store value, and removes size 1 dimensions
            values = values.write(t, tf.squeeze(state_val))
            
            # store inital probability distribution
            initial_prob_list = initial_prob_list.write(t, init_probs)

            # store improved probability distribution
            final_log_prob_list = final_log_prob_list.write(t, final_log_probs)

        # convert tensor array to tensor
        initial_prob_list = initial_prob_list.stack()
        final_log_prob_list = final_log_prob_list.stack()
        values = values.stack()

        return initial_prob_list, final_log_prob_list, values, reward
    
    """ def compute_loss(self, action_probs: tf.Tensor,  values: tf.Tensor,  returns: tf.Tensor, loss_function: tf.keras.losses.Loss) -> tf.Tensor:
        Computes the combined actor-critic loss.

        critic_loss = loss_function(values, returns)

        # total loss is actor-critic loss
        return critic_loss """


