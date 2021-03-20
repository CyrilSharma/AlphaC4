import json
from typing import Any, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tqdm
from kaggle_environments import evaluate, make, utils
from tensorflow import keras
from tensorflow.keras import layers

from ActorCritic import ActorCritic
from C4_Helpers import state_to_action
from MCTS import SearchTree

# Opening JSON file 
with open('parameters.json') as f:
    params = json.load(f) 

with open('C4_Config.json') as f:
    config = json.load(f) 

model = keras.models.load_model("my_model", compile=False)
tree = SearchTree(model, params, config)
env = make("connectx", debug=True)
env.render()


def battle(mode):
    running_reward = 0

    with tqdm.trange(500) as t:
        for i in t:
            if np.random.randint(2) == 0:
                config = [mode, None]
                player = 2
            else:
                config = [None, mode]
                player = 1

            trainer = env.train(config)

            while not env.done:

                action, _, _, _ = tree.MCTS()

                observation, reward, done, _ = trainer.step(action)

                state_to_action(tree.state * -1, np.array(observation).reshape(6, 7))

            running_reward = reward * 0.01 + running_reward * 0.99

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                print(f'Episode {i}: average reward: {running_reward}')
                print(f'Agent is player {player}')
                env.render()
        
if __name__ == '__main__':
    battle("random")