import json
import numpy as np
import tensorflow as tf
import tqdm

from tensorflow import keras
import sys

from ActorCritic import ActorCritic
from helpers import render
from MCTS import MCTS
from C4 import C4

def battle():
    player = float(sys.argv[1])

    if player == 0:
        player = np.random.randint(2) * 2 - 1
    
    # Opening JSON file 
    with open('parameters.json') as f:
        params = json.load(f) 

    with open('config.json') as f:
        config = json.load(f) 

    model = keras.models.load_model("my_model", compile=False)
    tree = MCTS(model, config["timeout"], params["c_puct"])

    game = C4()

    turn = 0
    terminal = False

    while not terminal:
        if (game.player == player):
            render(game.state)
            action = int(input("Which column?? "))
            game.move(action)
            tree.shift_root(action)
            reward, terminal = game.is_terminal(action)
        else:
            render(game.state)
            # get data
            init_probs, final_probs, state_val = tree.final_probs(game, 0)
            print(f"\nInit probs: {init_probs.numpy()}")
            print(f"Final probs: {final_probs.numpy()}")

            # choose action and modify state accordingly
            action = np.random.choice([0,1,2,3,4,5,6], p=final_probs.numpy())

            game.move(action)
            tree.shift_root(action)
            reward, terminal = game.is_terminal(action)
    
    render(game.state)


        
if __name__ == '__main__':
    battle()