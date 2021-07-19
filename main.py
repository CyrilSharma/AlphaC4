
import numpy as np
import tensorflow as tf
import sys

sys.path.append("/kaggle_simulations/agent")

from utils.C4 import C4
from utils.MCTS import MCTS

def convert_state(state):
    new_state = np.array(state)
    new_state.shape = (6, 7)
    new_state[new_state == 2] = -1
    return new_state

params = {
    "c_puct": 3,
    "epsilon": 0.25,
    "dirichlet": 2.0,
    "timeout": 1
}
modelpath = "model"
model = tf.keras.models.load_model(f"/kaggle_simulations/agent/utils/{modelpath}", compile=False)
tree = MCTS(model, params)
game = C4()

def my_agent(obs, config):
    tree = MCTS(model, params)
    tree.timeout = config.timeout - 0.5
    game.state = convert_state(obs.board)
    game.player = 1 if obs.mark == 1 else -1
    probs, _ = tree.final_probs(game, 1)
    action = np.argmax(probs)
    return action.item()
