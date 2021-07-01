import unittest
import numpy as np
from C4 import C4
from helpers import render

columns = 7
rows = 6

class Test_C4(unittest.TestCase):

    def test_is_win(self):
        game = C4(6,7,4)


        game.state = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, 0, 0], [0, -1, 0, 1, -1, 1, 0], [0, 1, -1, 1, -1, 1, 0]])

        render(game.state)

        action = 5

        self.assertFalse(game.is_win(action))
        print('Is not win.')

        game.state = np.array([[0, 0, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, -1, 0], [0, 1, 0, 1, -1, 1, 0], [0, 1, -1, 1, -1, 1, 0]])

        action = 5

        render(game.state)

        self.assertTrue(game.is_win(action))
        print('Is win.')

        game.state = np.array([[0, 0, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, -1, 0], [0, 1, 0, 1, -1, 1, 0], [0, 1, 1, 1, -1, 1, 0]])

        action = 2

        render(game.state)

        self.assertTrue(game.is_win(action))
        print('Is win.')

        game.state = np.array([[0, 0, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, -1, 1, 0], [1, 1, 1, 1, -1, 1, 0]])

        action = 0

        render(game.state)

        self.assertTrue(game.is_win(action))
        print('Is win.')

        game.state = np.array(
        [[-0., -0., -0., -0., -0., -0., -0.],
        [ 1., -1., -0., -1.,  1., -0., -0.],
        [-1.,  1., -0.,  1., -1., -1.,  0.],
        [ 1., -1., -0., -1.,  1., -1., -1.],
        [ 1.,  1., -0., -1.,  1., -1., -1.],
        [-1.,  1.,  1.,  1.,  1., -1., -1.]]
        )

        action = 2
        self.assertTrue(game.is_win(action))
        print('Is win.')

    def test_terminal(self):
        game = C4(6,7,4)

        game.state = np.array([[0, 0, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, -1, 0], [0, 1, 0, 1, -1, 1, 0], [0, 1, -1, 1, -1, 1, 0]])

        action = 5

        render(game.state)

        reward, terminal = game.is_terminal(action)

        self.assertTrue(reward == 1)
        self.assertTrue(terminal)
        print('Is terminal.')

        game.state = np.array([[0, 0, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, -1, 0], [0, 1, 0, 1, -1, 1, 0], [0, 1, -1, 1, -1, 1, 0]])

        action = 5

        render(game.state, [1, -1, 0])

        reward, terminal = game.is_terminal(action)

        self.assertTrue(reward == 1)
        self.assertTrue(terminal)
        print('Is terminal.')

        game.state = np.array([[-1, -1, -1, 1, 1, 1, 0], [1, 1, 1, -1, -1, -1, 0], [-1, -1, -1, 1, 1, 1, 0],
        [1, 1, 1, -1, -1, -1, 0], [-1, -1, -1, 1, 1, 1, 0], [1, 1, 1, -1, -1, -1, 0]])

        action = 5

        render(game.state)

        reward, terminal = game.is_terminal(action)

        self.assertTrue(reward == 0)
        self.assertFalse(terminal)
        print('Is not terminal.')

        game.state = np.array([[-1, -1, -1, 1, 1, 1, -1], [1, 1, 1, -1, -1, -1, 1], [-1, -1, -1, 1, 1, 1, -1],
        [1, 1, 1, -1, -1, -1, 1], [-1, -1, -1, 1, 1, 1, -1], [1, 1, 1, -1, -1, -1, 1]])

        action = 5

        render(game.state)

        reward, terminal = game.is_terminal(action)

        self.assertTrue(reward == 0)
        self.assertTrue(terminal)
        print('Is terminal.')

        game.state = np.array(
        [[-1.,  1., -1., -0., -0., -0., -0.],
        [-1.,  1., -1., -0., -0., -0., -0.],
        [ 1., -1.,  1., -0., -0., -0., -0.],
        [-1.,  1., -1.,  0., -0., -0., -0.],
        [ 1., -1.,  1.,  0., -0., -0., -0.],
        [-1.,  1., -1.,  1.,  0.,  0.,  0.]]
        )

        action = 3

        render(game.state)

        reward, terminal = game.is_terminal(action)

        self.assertTrue(reward == 1)
        self.assertTrue(terminal)

    def test_move(self):
        game = C4(6,7,4)

        game.state = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, -1, 0], [0, 1, 0, 1, -1, 1, 0], [0, 1, -1, 1, -1, 1, 0]])

        render(game.state)

        action = 5

        game.move(action)

        print('Action = 5')

        render(game.state)

        state_answer = np.array([[0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, -1, 0], [0, 1, 0, 1, -1, 1, 0], [0, 1, -1, 1, -1, 1, 0]])

        self.assertTrue((game.state == state_answer).all())
    
    def test_legal(self):
        game = C4(6,7,4)
        game.state = np.array([[0, -1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, -1, 0], [0, -1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, -1, 0], [0, 1, 0, 1, -1, 1, 0], [0, 1, -1, 1, -1, 1, 0]])

        render(game.state)

        legal_actions = game.legal()
        answer = np.array([1., 0., 1., 1., 1., 0., 1.])

        self.assertTrue((legal_actions == answer).all())

if __name__ == '__main__':
    unittest.main()