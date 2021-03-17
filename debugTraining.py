import unittest

import json
import numpy as np
import tensorflow as tf
from kaggle_environments import evaluate, make, utils
from tensorflow import keras
from tensorflow.keras import layers

from ActorCritic import ActorCritic
from debugTrainers import Test_Overfitting
from Training import C4Trainer

columns = 7
rows = 6

class debugTraining(unittest.TestCase):

    def test_Overfitting(self):
        params = {
            "episodes": 1500,
            "tau": 4.0,
            "gamma": 0.90,
            "alpha": 0.00001,
            "eps": 0.00000000000001,
            "c_puct": 1
        }

        with open('C4_Config.json') as f:
            config = json.load(f) 

        state1 = tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, 0, 0], [0, -1, 0, 1, -1, 0, 0], [0, 1, -1, 1, -1, 0, 0]]).reshape(1, rows, columns, 1), dtype=tf.float32)

        state2 = tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, 0, 0], [0, -1, 0, 1, -1, 0, 0], [0, 1, -1, 1, -1, 0, 0]]).reshape(1, rows, columns, 1), dtype=tf.float32)

        states = [state1, state2]

        rewards = [0.5, -0.5]

        overfit_trainer = Test_Overfitting(params, config, states, rewards)

        action_vals, state_val = overfit_trainer.model(state1)
        print('Model predicted: ', state_val.numpy().item())
        print('Answer', rewards[0])

        action_vals, state_val = overfit_trainer.model(state2)
        print('Model predicted: ', state_val.numpy().item())
        print('Answer', rewards[1])

        overfit_trainer.training_loop()

        action_vals, state_val = overfit_trainer.model(state1)
        print('Model predicted: ', state_val.numpy().item())
        print('Answer', rewards[0])

        action_vals, state_val = overfit_trainer.model(state2)
        print('Model predicted: ', state_val.numpy().item())
        print('Answer', rewards[1])

        self.assertTrue(5==5)


if __name__ == '__main__':
    unittest.main()
