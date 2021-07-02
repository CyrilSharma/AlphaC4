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
    def __init__(self, model, params, config, states, rewards):
        super().__init__(model, params, config)
        self.states = states
        self.rewards = rewards
    
    def run_episode(self, player=1):
        game = C4()
        index = np.random.randint(len(self.states))
        game.state = copy.deepcopy(self.states[index])
        reward = self.rewards[index]

        init_probs, final_probs, state_val = self.tree.final_probs(game, self.params["temp"])

        # update memory
        return [(game.state.reshape(self.rows, self.columns, 1), final_probs, reward)]

