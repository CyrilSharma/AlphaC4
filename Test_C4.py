import unittest
import numpy as np
from kaggle_environments import evaluate, make, utils
from C4_Helpers import is_win, is_terminal, move, legal, render

columns = 7
rows = 6

class Test_C4(unittest.TestCase):

    def test_is_win(self):
        state = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 0, 0], [0, 2, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])

        render(state)

        action = 5

        self.assertFalse(is_win(state, action))
        print('Is not win.')

        state = np.array([[0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])

        action = 5

        render(state)

        self.assertTrue(is_win(state, action))
        print('Is win.')

        state = np.array([[0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 1, 1, 2, 1, 0]])

        action = 2

        render(state)

        self.assertTrue(is_win(state, action))
        print('Is win.')

        state = np.array([[0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 2, 1, 0], [1, 1, 1, 1, 2, 1, 0]])

        action = 0

        render(state)

        self.assertTrue(is_win(state, action))
        print('Is win.')

    def test_is_terminal(self):
        state = np.array([[0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])

        action = 5

        render(state)

        reward, terminal = is_terminal(state, action)

        self.assertTrue(reward == 1)
        self.assertTrue(terminal)
        print('Is terminal.')

        state = np.array([[2, 2, 2, 1, 1, 1, 2], [1, 1, 1, 2, 2, 2, 1], [2, 2, 2, 1, 1, 1, 2],
        [1, 1, 1, 2, 2, 2, 1], [2, 2, 2, 1, 1, 1, 2], [1, 1, 1, 2, 2, 2, 1]])

        action = 5

        render(state)

        reward, terminal = is_terminal(state, action)

        self.assertTrue(reward == 0)
        self.assertTrue(terminal)
        print('Is terminal.')

    def test_move(self):
        state = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])

        render(state)

        action = 5

        move(state, action, 2)

        print('Action = 5')

        render(state)

        state_answer = np.array([[0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])

        self.assertTrue((state == state_answer).all())
    
    def test_legal(self):
        state = np.array([[0, 2, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])

        render(state)

        legal_actions = legal(state)

        answer = [0, 2, 3, 4, 6]

        self.assertTrue(legal_actions == answer)



if __name__ == '__main__':
    unittest.main()