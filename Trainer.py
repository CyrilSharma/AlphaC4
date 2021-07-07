import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm
from tensorflow import keras
from ActorCritic import ActorCritic
from collections import deque
from helpers import render
from MCTS import MCTS
from Tournament import Tournament
from C4 import C4
from matplotlib import pyplot as plt
from Evaluate import Evaluate
import copy
import sys
import logging

logging.basicConfig(filename='trainer.log', filemode='w', level=logging.DEBUG)
class Trainer():
    def __init__(self, model: ActorCritic, params, config):
        self.rows = config['rows']
        self.columns = config['columns']
        self.params = params
        self.config = config
        self.model = compile_model(model, params)
        self.good_moves = []
        self.perfect_moves = []
        self.battle_outcomes = []
        self.history = []

    def run_episode(self):
        episode_memory = []
        game = C4()
        turn = 0
        actions = []
        terminal = False
        temp = self.params["temp"]

        while not terminal:
            # get data
            final_probs, state_val = self.tree.final_probs(game, temp)
            # store value, and removes size 1 dimensions
            episode_memory.append([copy.deepcopy(game.state).reshape((self.rows,self.columns,1)) * game.player, final_probs, None, game.player])

            # choose action and modify state accordingly
            action = np.random.choice([0,1,2,3,4,5,6], p=final_probs)
            actions.append(action)
            game.move(action)
            self.tree.shift_root(action)

            reward, terminal = game.is_terminal(action)

            # make agent play its best moves after x turns
            turn += 1
            if (turn > self.params["exp_turns"]):
                temp = 0

        logging.debug("Actions: " + str(actions))
        logging.debug("\n" + np.array_str(game.state))
        logging.debug(f"Winner: {-1 * reward * game.player}")
        # update memory
        return [(ep[0], ep[1], -1 * reward * game.player * ep[3]) for ep in episode_memory]


    def train(self, training_data, args):
        """Runs a model training step."""
        boards, probs, values = list(zip(*training_data))
        self.model.fit(x=np.stack(boards), y=[np.stack(probs), np.stack(values)], batch_size=args["batch_size"], 
        epochs=args["epochs"])
    
    def test(self):
        old_model = keras.models.load_model("Models/v0", compile=False)
        p2 = MCTS(old_model, self.params)
        p1 = MCTS(self.model, self.params)
        
        avg_reward = Tournament(p1, p2, self.params["numBattles"])
        self.battle_outcomes.append(avg_reward)

        print(f'New network won {avg_reward * 100}% of the matches against the old network.')

        p1.reset()

        # note that the dataset is of size 1000
        good_moves, perfect_moves, mse = Evaluate(samples=200, agent=p1, sample_spacing=3)
        print(f"good_moves: {good_moves * 100}% \nperfect_moves: {perfect_moves * 100}% \nstate mse: {mse}")
        self.good_moves.append(good_moves * 100)
        self.perfect_moves.append(perfect_moves * 100)

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

            self.model.save(f"Models/v0", save_format='tf')
            self.train(trainingData, self.params["training_args"])
            self.test()
        
        self.model.save(f"Models/{model_name}", save_format='tf')

        
        if graphs:
            plt.title("Battle Outcomes") 
            plt.xlabel("Iteration") 
            plt.ylabel("Win Percentage") 
            plt.plot(list(range(self.params["num_iters"])), self.battle_outcomes) 
            plt.show()

            plt.title("Good moves over time") 
            plt.xlabel("Iteration") 
            plt.ylabel("Percentage") 
            plt.plot(list(range(self.params["num_iters"])), self.good_moves) 
            plt.show()

            plt.title("Perfect moves over time") 
            plt.xlabel("Iteration") 
            plt.ylabel("Percentage") 
            plt.plot(list(range(self.params["num_iters"])), self.perfect_moves) 
            plt.show()

def compile_model(model, params):
    losses = {'output_1':prob_loss, 'output_2':value_loss}
    lossWeights={'output_1':0.5, 'output_2':0.5}
    adam = keras.optimizers.Adam(learning_rate=params["alpha"])
    model.compile(optimizer=adam, loss=losses, loss_weights=lossWeights)
    return model

def prob_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return loss

def value_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = mse(y_true, y_pred)
    return loss
