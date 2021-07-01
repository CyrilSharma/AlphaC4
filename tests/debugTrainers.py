import json
import copy
from C4 import C4
from MCTS import MCTS
import numpy as np
import tensorflow as tf
from Trainer import Trainer

# PLAN
# Create a function which takes in functions and returns a function, specifically the train_step funciton.
# Tests could then proceed by specifying which functions to pass in, i,e which version of train_step you want

class OverfitTrainer(Trainer):
    def __init__(self, model, params, config, states, rewards, input_shape=(1,6,7,1)):
        super().__init__(model, params, config, input_shape)
        self.states = states
        self.rewards = rewards
    
    def run_episode(self, player=1):
        self.tree.reset()
        game = C4()
        index = np.random.randint(len(self.states))

        game.state = copy.deepcopy(self.states[index])
        reward = tf.convert_to_tensor([self.rewards[index]], dtype=tf.float32)

        initial_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        final_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        turn = 0

        init_probs, final_probs, state_val = self.tree.final_probs(game, self.params["temp"])

        # store value, and removes size 1 dimensions
        values = values.write(turn, tf.squeeze(state_val))
        initial_prob_list = initial_prob_list.write(turn, init_probs)
        final_prob_list = final_prob_list.write(turn, final_probs)
        
        # convert tensor array to tensor
        initial_prob_list = initial_prob_list.stack()
        final_prob_list = final_prob_list.stack()
        values = values.stack()

        return initial_prob_list, final_prob_list, values, reward

