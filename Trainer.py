import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm
from tensorflow import keras
from ActorCritic import ActorCritic
from collections import deque
from helpers import render
from MCTS import MCTS
from Battle import battle
from C4 import C4
from matplotlib import pyplot as plt
import sys
class Trainer():
    def __init__(self, model: ActorCritic, params, config):
        self.rows = config['rows']
        self.columns = config['columns']
        self.params = params
        self.config = config
        self.model = compile_model(model, params)
        self.history = []

    def run_episode(self):
        player = np.random.randint(2) * 2 - 1
        episode_memory = []
        game = C4()

        turn = 0
        terminal = False

        while not terminal:
            # get data
            init_probs, final_probs, state_val = self.tree.final_probs(game, self.params["temp"])

            # choose action and modify state accordingly
            action = np.random.choice([0,1,2,3,4,5,6], p=final_probs.numpy())
            game.move(action)
            self.tree.shift_root(action)

            if game.player == player:
                # store value, and removes size 1 dimensions
                episode_memory.append([game.state.reshape((self.rows,self.columns,1), dtype=tf.float32), final_probs, None])
                turn += 1

            reward, terminal = game.is_terminal(action)

        if reward != 0:
            if game.player == player:
                reward = 1
            else:
                reward = -1

        # update memory
        return [(episode_memory[0], episode_memory[1], reward) for ep in episode_memory]


    def train(self, training_data, args):
        """Runs a model training step."""

        boards, probs, values = list(zip(*training_data))
        self.model.fit(x=np.stack(boards), y=[np.stack(probs), np.stack(values)], batch_size=args["batch_size"], 
        epochs=args["epochs"])
        
        print()

        self.model.evaluate(x=np.stack(boards), y=[np.stack(probs), np.stack(values)], batch_size=args["batch_size"])

        print(self.model.predict(np.stack(boards), batch_size=args["batch_size"]))

    def test_network(self, num, episodes=20):

        model_old = keras.models.load_model(f"Models/v{num}", compile=False)

        old_tree = MCTS(model_old, config["timeout"], params["c_puct"])

        avg_reward = 0
    
        for ep_num in range(episodes):
            # reset trees, and make tau non zero
            self.tree.reset()
            old_tree.reset()

            swap = np.random.choice([True, False])

            reward = battle(self.tree, old_tree, swap)

            avg_reward += (reward - avg_reward) / (ep_num + 1)
        
        self.tree.reset()

        print(f'New network scored an average reward of: {avg_reward} against the old network.')

    def training_loop(self, save=True, graphs=True):

        episodes = self.params["num_eps"]
        iterations = self.params["num_iters"]

        for j in range(iterations):
            iterMemory = deque([], maxlen=self.params["maxQueueLen"])

            for t in tqdm(range(episodes), desc="Self Play"):
                self.tree = MCTS(self.model, self.config["timeout"], self.params["c_puct"])
                iterMemory += self.run_episode()
            
            self.history.append(iterMemory)

            if len(self.history) > self.params["numStoredIters"]:
                self.history.pop(0)
            
            trainingData = []
            for e in self.history:
                trainingData.extend(e)
            shuffle(trainingData)

            self.train(trainingData, self.params["training_args"])

        
        if graphs:
            pass
                    
        if save:
            self.model.save("my_model")

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
