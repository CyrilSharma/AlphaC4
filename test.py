from kaggle_environments import evaluate, make, utils
import numpy as np
import tensorflow as tf
import sys
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

modelpath = "utils/model"
model = tf.keras.models.load_model(f"{modelpath}", compile=False)
tree = MCTS(model, params)
game = C4()

def my_agent(obs, config):
    print(obs)
    tree = MCTS(model, params)
    tree.timeout = config.timeout - 0.5
    print(config.timeout)
    print(tree.timeout)
    game.state = convert_state(obs.board)
    game.player = 1 if obs.mark == 1 else -1
    probs, _ = tree.final_probs(game, 1)
    action = np.argmax(probs)
    return action.item()

def main():
    env = make("connectx", debug=True)
    env.run([my_agent, my_agent])
    print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
    print(env.state[0])
    print(convert_state(env.state[0].observation.board))

if __name__ == "__main__":
    main()

