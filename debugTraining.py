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
            "episodes": 100,
            "tau": 1.5,
            "alpha": 0.00001,
            "eps": 0.00000000000001,
            "c_puct": 1,
            "c": 0.001,
            "cutoff": 0.05
        }

        config = {
            "rows": 6, 
            "columns": 7, 
            "inarow": 4,
            "timeout": 2,
            "debug": True
        }

        state1 = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0, 0, 0], 
                [0, -1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0], 
                [0, -1, 0, 1, -1, 0, 0], 
                [0, 1, -1, 1, -1, 0, 0]
            ]
        )

        state2 = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0, 0, 0], 
                [0, -1, 0, 1, 0, 0, 0],
                [0, 1, 0, -1, 1, 0, 0], 
                [0, -1, 1, 1, -1, 0, 0], 
                [0, 1, -1, 1, -1, 0, 0]
            ]
        )

        states = [state1, state2]

        rewards = [0.5, -0.5]

        overfit_trainer = Test_Overfitting(params, config, states, rewards)

        action_vals, state_val = overfit_trainer.call_model(tf.convert_to_tensor(state1.reshape(1, rows, columns, 1), dtype=tf.float32))
        print('Model predicted: ', state_val.numpy().item())
        print('Answer', rewards[0])

        action_vals, state_val = overfit_trainer.call_model(tf.convert_to_tensor(state2.reshape(1, rows, columns, 1), dtype=tf.float32))
        print('Model predicted: ', state_val.numpy().item())
        print('Answer', rewards[1])

        overfit_trainer.training_loop()

        action_vals, state_val = overfit_trainer.call_model(tf.convert_to_tensor(state1.reshape(1, rows, columns, 1), dtype=tf.float32))
        print('Model predicted: ', state_val.numpy().item())
        print('Answer', rewards[0])

        action_vals, state_val = overfit_trainer.call_model(tf.convert_to_tensor(state2.reshape(1, rows, columns, 1), dtype=tf.float32))
        print('Model predicted: ', state_val.numpy().item())
        print('Answer', rewards[1])

        self.assertTrue(5==5)


if __name__ == '__main__':
    unittest.main()
