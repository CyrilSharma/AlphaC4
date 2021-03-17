from typing import Any, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tqdm
from kaggle_environments import evaluate, make, utils
from tensorflow import keras
from tensorflow.keras import layers

from ActorCritic import ActorCritic
from Training import get_action, tf_env_reset

model = keras.models.load_model("my_model", compile=False)
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
            state = tf_env_reset()

            while not env.done:

                action_values, state_val = model(state)

                print(state_val.numpy())

                tau = 0.5
                
                action, prob = get_action(state, action_values, tau)

                action = int(action.numpy())

                observation, reward, done, _ = trainer.step(action)
                state = tf.convert_to_tensor(np.array(observation['board'], dtype=np.float32).reshape(1, 6, 7, 1))

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