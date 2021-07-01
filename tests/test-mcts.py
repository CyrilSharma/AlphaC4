import unittest
import json
import numpy as np
from C4 import C4
from MCTS import MCTS
from helpers import render
from ActorCritic import ActorCritic

params = {
    "episodes": 1500,
    "tau": 1.0,
    "gamma": 0.90,
    "alpha": 0.00001,
    "eps": 0.00000000000001,
    "c_puct": 3,
    "cutoff": 0.05
}

config = {
    "rows": 6, 
    "columns": 7, 
    "inarow": 4,
    "timeout": 2,
    "debug": True
}

class tester(unittest.TestCase):
    
    def test1(self):
        print('\nTest 2')
        print('_' * 30, '\n')

        config = {
            "rows": 6, 
            "columns": 7, 
            "inarow": 4,
            "timeout": 2,
            "debug": True
        }

        state = np.array(
            [[0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 1, 1, -1, 1],
            [0, 0, -1, 1, -1, 1, -1], 
            [0, 0, -1, -1, 1, 1, 1]], dtype=np.float32
        )

        print('Initial State: ')
        render(state)

        c4 = C4()
        c4.state = state

        model = ActorCritic()

        tree = MCTS(model, config["timeout"], c_puct=params["c_puct"])

        # Note that it's always assumed to be player one's turn
        result = tree.final_probs(c4, 1)

        print(f"\n{result[0]}", end="\n\n")
        print(result[1])

        children = tree.root.children

        actions = list(range(config['columns']))

        action = np.random.choice(actions, p=result[1])

        c4.move(action)
        render(c4.state)

        print('Algorithmn chose: ', action)
        print('_' * 30)
        self.assertTrue(action == 3)
    
    def test2(self):
        print('\nTest 3')
        print('_' * 30, '\n')
        np.random.seed(42)

        config = {
            "rows": 6, 
            "columns": 7, 
            "inarow": 4,
            "timeout": 2,
            "debug": True
        }

        state = np.array(
        [[-0., -0., -0., -0., -0., -0., -0.],
        [-0., -0., -0., -0., -0., -0., -0.],
        [ 1., -0., -0., -0.,  1., -0., -0.],
        [-1., -0., -0., -0.,  1.,  1., -0.],
        [-1., -0., -0., 0, -1., -1., -1.],
        [ 1.,  1.,  1., -1.,  1., -1., -1.]], dtype=np.float32
        )

        print('Initial State: ')
        render(state, [1, -1, 0])

        c4 = C4()
        c4.state = state

        model = ActorCritic()

        tree = MCTS(model, config["timeout"], c_puct=params["c_puct"])

        # Note that it's always assumed to be player one's turn
        result = tree.final_probs(c4, 1)

        print(result[0])
        print(result[1])

        actions = list(range(config['columns']))

        action = np.random.choice(actions, p=result[1])

        c4.move(action)
        render(c4.state)

        print('Algorithmn chose: ', action)
        print('_' * 30)

        self.assertTrue(action == 3)
    
    def test3(self):
        print('\nTest 3')
        print('_' * 30, '\n')
        np.random.seed(42)

        config = {
            "rows": 6, 
            "columns": 7, 
            "inarow": 4,
            "timeout": 2,
            "debug": True
        }

        state = np.array(
        [[-0., -0., -0., -0., -0., -0., -0.],
        [-0., -0., -0., -0., -0., -0., -0.],
        [ 1., -0., -0., -0.,  1., -0., -0.],
        [-1., -0., -0., -0.,  1.,  1., -0.],
        [-1.,  1.,   1., 0,   -1., -1., -1.],
        [ 1.,  1.,  -1., 0.,  -1., 1., -1.]], dtype=np.float32
        )

        print('Initial State: ')
        render(state, [1, -1, 0])

        c4 = C4()
        c4.state = state

        model = ActorCritic()

        tree = MCTS(model, config["timeout"], c_puct=params["c_puct"])

        # Note that it's always assumed to be player one's turn
        result = tree.final_probs(c4, 1)

        print(result[0])
        print(result[1])

        actions = list(range(config['columns']))

        action = np.random.choice(actions, p=result[1])

        c4.move(action)
        render(c4.state)

        print('Algorithmn chose: ', action)
        print('_' * 30)

        self.assertTrue(action == 1)


if __name__ == '__main__':
    unittest.main()