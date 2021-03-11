import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kaggle_environments import evaluate, make, utils
from ActorCritic import ActorCritic
columns = 7
rows = 6

class Test_MCTS(unittest.TestCase):

    def test_opposite_state(self):

        board = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 1, 2, 0, 0], [0, 1, 2, 1, 2, 0, 0]])

        state = tf.convert_to_tensor(np.array(board).reshape(1, rows, columns, 1))
        state_opp = opposite_state(state)
        board_opp = state_opp.numpy().squeeze()

        board_answer = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 1, 0, 0], [0, 2, 1, 2, 1, 0, 0]])

        self.assertTrue((board_opp == board_answer).all())
    
    def test_legal(self):

        state = tf.convert_to_tensor(np.array([[0, 2, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 0, 0], [0, 2, 0, 1, 2, 0, 0], [0, 1, 2, 1, 2, 0, 0]]))
        action_values = tf.convert_to_tensor(np.array([0, 2, 5, 0, 4, 2, 0]).reshape(1, columns))
        legal_action_values, legal_actions = legal(state, action_values)

        legal_action_answers = np.array([0, 2, 3, 4, 6])
        legal_action_value_answers = np.array([0, 5, 0, 4, 0])

        self.assertTrue((legal_actions == legal_action_answers).all())
        self.assertTrue((legal_action_values.numpy() == legal_action_value_answers).all())

    
    def test_get_action(self):

        tf.random.set_seed(42)
        state = tf.convert_to_tensor(np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 0, 0], [0, 2, 0, 1, 2, 0, 0], [0, 1, 2, 1, 2, 0, 0]]))
        action_values = tf.convert_to_tensor(np.array([0, 2, 5, 0, 4, 2, 0]).reshape(1, columns))

        action_probs = tf.convert_to_tensor(np.array([0.000001, 1, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]).reshape(1, columns))
        action_values = tf.math.log(action_probs)


        tau = 1.0

        action, prob = get_action(state, action_values, tau)

        self.assertTrue(action.numpy() == 1)
        self.assertTrue(0.99 <= prob.numpy() <= 1)

        action_probs = tf.convert_to_tensor(np.array([6, 1, 1, 1, 1, 1, 1], dtype=np.float32).reshape(1, columns))
        action_values = tf.math.log(action_probs)
        tau = 0.5

        action, prob = get_action(state, action_values, tau)

        self.assertTrue(action.numpy() == 0)
        self.assertTrue(0.855 <= prob.numpy() <= 0.860)

if __name__ == '__main__':
    unittest.main()