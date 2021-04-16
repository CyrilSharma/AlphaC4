import unittest
import json
import numpy as np
from utils.bindings import MCTS, C4
from C4_Helpers import render

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

class Test_MCTS(unittest.TestCase):
    
    def test1_MCTS(self):
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
            [0, 0, -1, -1, 1, 1, 1]]
        )

        print('Initial State: ')
        render(state, [1, -1, 0])

        c4 = C4()
        c4.state = state

        tree = MCTS(num_threads=4)

        # Note that it's always assumed to be player one's turn
        probs = tree.final_probs(c4, 1)

        print(probs)

        children = tree.root.children

        for i in range(len(children)):
            print(children[i].visits)

        actions = list(range(config['columns']))

        action = np.random.choice(actions, p=probs)

        print('Algorithmn chose: ', action)
        print('_' * 30)
        self.assertTrue(action == 3)
    
    def test2_MCTS(self):
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
        [ 1.,  1.,  1., -1.,  1., -1., -1.]]
        )

        print('Initial State: ')
        render(state, [1, -1, 0])

        c4 = C4()

        c4.state = state

        tree = MCTS()

        # Note that it's always assumed to be player one's turn
        probs = tree.final_probs(c4, 1)

        print(probs)

        actions = list(range(config['columns']))

        action = np.random.choice(actions, p=probs)

        print('Algorithmn chose: ', action)
        print('_' * 30)

        self.assertTrue(action == 3)


if __name__ == '__main__':
    unittest.main()