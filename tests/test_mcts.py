import unittest
import json
import numpy as np
from C4 import C4
from MCTS import MCTS
from helpers import render
from ActorCritic import ActorCritic

params = {
    "temp": 1,
    "numBattles": 5,
    "alpha": 0.00001,
    "c_puct": 4,
    "c": 0.001,
    "dirichlet": 1.75,
    "epsilon": 0.25,
    "exp_turns": 16,
    "cutoff": 0.05,
    "timeout": 1,
    "num_iters": 10,
    "num_eps": 10,
    "maxQueueLen": 10000,
    "numStoredIters": 3,
    "training_args": {
        "epochs": 10,
        "batch_size": 64
    }
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
            [0, 0, 0, 0, -1, -1, 0],
            [0, 0, 0, 1, 1, -1, 1],
            [0, 0, -1, 1, -1, 1, -1], 
            [0, 0, -1, -1, 1, 1, 1]], dtype=np.float32
        )

        print('Initial State: ')
        render(state)

        c4 = C4()
        c4.state = state

        model = ActorCritic()

        tree = MCTS(model, params)

        # Note that it's always assumed to be player one's turn
        final_probs, _ = tree.final_probs(c4, 1)

        print(f"\n{final_probs}", end="\n\n")

        children = tree.root.children

        action = np.argmax(final_probs)

        c4.move(action)
        render(c4.state)

        print('Algorithmn chose: ', action)
        print('_' * 30)
        self.assertTrue(action == 3)
    
    def test2(self):
        print('\nTest 2')
        print('_' * 30, '\n')
        np.random.seed(42)

        params["timeout"] = 2.0

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

        tree = MCTS(model, params)

        # Note that it's always assumed to be player one's turn
        final_probs, _ = tree.final_probs(c4, 1)

        print(final_probs)

        action = np.argmax(final_probs)

        c4.move(action)
        render(c4.state)

        print('Algorithmn chose: ', action)
        print('_' * 30)

        self.assertTrue(action == 3)
    
    def test3(self):
        print('\nTest 3')
        print('_' * 30, '\n')
        np.random.seed(42)
        params["timeout"] = 5.0

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
        tree = MCTS(model, params)
        final_probs, _ = tree.final_probs(c4, 1)

        print(final_probs)

        action = np.argmax(final_probs)

        c4.move(action)
        render(c4.state)

        print('Algorithmn chose: ', action)
        print('_' * 30)

        self.assertTrue(action == 1)

    def test4(self):
        print('\nTest 4')
        print('_' * 30, '\n')
        np.random.seed(42)

        params["timeout"] = 4.0

        state = np.array(
        [[-0., -0., -0., -0., -0., -0., -0.],
        [-0., -0., -0., -0., -0., -0., -0.],
        [ 0., -0., -0., -0., 0., 0., 0.],
        [ 0., -0., -0., -0., 0., 1., 0.],
        [ 0.,  0.,  0.,  0., 0., 1., 0.],
        [ 0.,  0.,  0.,  0., 0., 1., 0.]], dtype=np.float32
        )

        print('Initial State: ')
        render(state, [1, -1, 0])

        c4 = C4()
        c4.state = state

        model = ActorCritic()
        tree = MCTS(model, params)
        final_probs, _ = tree.final_probs(c4, 1)
        action = np.argmax(final_probs)

        print(f'Final probabilities: {final_probs}')
        print(f'Action: {action}')

        render(c4.state)
        tree.shift_root(action)

        # Note that it's always assumed to be player one's turn
        final_probs, _ = tree.final_probs(c4, 1)
        action = np.argmax(final_probs)

        print(f'Final probabilities: {final_probs}')
        print(f'Action: {action}')
        
        self.assertTrue(action == 5)
    
    def test5(self):
        print('\nTest 5')
        print('_' * 30, '\n')
        np.random.seed(42)

        params["timeout"] = 2.0

        state = np.array(
        [[-0., -0., -0., -0., -0., -0., -0.],
        [-0., -0., -0., -0., -0., -0., -0.],
        [ 0., -0., -0., -0., 0., 0., 0.],
        [ 0., -0., -0., -0., 0., -1., 0.],
        [ 0.,  0.,  0.,  0., 1., -1., 0.],
        [ 1.,  1.,  0.,  0., 1., -1., 0.]], dtype=np.float32
        )

        print('Initial State: ')
        render(state, [1, -1, 0])

        c4 = C4()
        c4.state = state
        c4.player = -1

        model = ActorCritic()
        tree = MCTS(model, params)
        final_probs, _ = tree.final_probs(c4, 1)
        action = np.argmax(final_probs)

        print(f'Final probabilities: {final_probs}')
        print(f'Action: {action}')

        c4.move(0)
        tree.shift_root(0)
        render(c4.state)

        # Note that it's always assumed to be player one's turn
        final_probs, _ = tree.final_probs(c4, 1)
        action = np.argmax(final_probs)

        print(f'Final probabilities: {final_probs}')
        print(f'Action: {action}')
        
        self.assertTrue(action == 5)


if __name__ == '__main__':
    unittest.main()