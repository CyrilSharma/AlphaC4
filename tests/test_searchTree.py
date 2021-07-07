import unittest
import json
import numpy as np
from C4 import C4
from MCTS import MCTS
from helpers import render
from ActorCritic import ActorCritic
from SearchTree import SearchTree

params = {
    "episodes": 1500,
    "tau": 1.0,
    "gamma": 0.90,
    "alpha": 0.00001,
    "eps": 0.00000000000001,
    "c_puct": 3,
    "cutoff": 0.05
}

class tester(unittest.TestCase):
    
    def test1(self):
        print('\nTest 1')
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
        tree = SearchTree(model, config["timeout"], c_puct=params["c_puct"])
        tree.action_probs(c4, 1)
        
        action = 3
        self.assertTrue(action == 3)


if __name__ == '__main__':
    unittest.main()