import json
import copy
from C4 import C4
from MCTS import MCTS
import numpy as np
import tensorflow as tf
from Trainer import Trainer
from tqdm import tqdm
from collections import deque
from random import shuffle

# PLAN
# Create a function which takes in functions and returns a function, specifically the train_step funciton.
# Tests could then proceed by specifying which functions to pass in, i,e which version of train_step you want

class OverfitTrainer(Trainer):
    def __init__(self, model, params, config, states, rewards):
        super().__init__(model, params, config)
        self.states = states
        self.rewards = rewards
    
    def run_episode(self):
        final_probs = np.zeros(self.rows)
        final_probs[0] = 1.0

        # update memory
        return [(copy.deepcopy(self.states[i]).reshape(self.rows, self.columns, 1), final_probs, self.rewards[i]) for i in range(len(self.states)) for j in range(self.params["training_args"]["batch_size"])]
    
    def training_loop(self, model_name, graphs=True):

        episodes = self.params["num_eps"]
        iterations = self.params["num_iters"]

        for j in range(iterations):
            iterMemory = deque([], maxlen=self.params["maxQueueLen"])

            for t in tqdm(range(episodes), desc="Self Play"):
                self.tree = MCTS(self.model, self.params)
                iterMemory += self.run_episode()
            
            self.history.append(iterMemory)

            if len(self.history) > self.params["numStoredIters"]:
                self.history.pop(0)
            
            trainingData = []
            for e in self.history:
                trainingData.extend(e)
            shuffle(trainingData)

            self.train(trainingData, self.params["training_args"])
        
        self.model.save(f"Models/{model_name}", save_format='tf')

