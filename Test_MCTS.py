import unittest
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kaggle_environments import evaluate, make, utils
from ActorCritic import ActorCritic
from C4_Helpers import convert_state, is_terminal, legal, move, unmove, render
from MCTS import SearchTree, Node

params = {
    "episodes": 1500,
    "tau": 1.0,
    "gamma": 0.90,
    "alpha": 0.00001,
    "eps": 0.00000000000001,
    "c_puct": 1,
    "cutoff": 0.05
}

config = {
    "rows": 6, 
    "columns": 7, 
    "inarow": 4,
    "timeout": 2,
    "debug": True
}

model = keras.models.load_model("my_model", compile=False)
class Test_MCTS(unittest.TestCase):

    def test_node(self):
        print('Original state: ')
        state = np.array([[0, 2, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])
        render(state)

        node = Node(state=state, action=1)
        node.append_child(state=state, prob=0.7, action=2)
        child = node.get_child(action=2)

        move(state, child.action, 1)

        answer = np.array([[0, 2, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 1, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])

        print('Method returns: ')
        render(state)
        print('Answer: ')
        render(answer)

        self.assertTrue((state == answer).all())
    
    def test_MCTS_expand(self):
        np.random.seed(42)

        print('Original state: ')
        state = np.array([[0, 2, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])
        render(state)

        tree = SearchTree(state=state, model=model, params=params, config=config)

        node = Node(state=state, prob=0.7, action=1)
        tree.expand(node)

        num_children = len(list(node.children.values()))
        
        self.assertTrue(len(node.actions) == num_children)
    

    def test_MCTS_backup(self):
        state = np.array([[0, 2, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2, 0],
        [0, 1, 0, 0, 1, 2, 0], [0, 1, 0, 1, 2, 1, 0], [0, 1, 2, 1, 2, 1, 0]])

        tree = SearchTree(state=state, model=model, params=params, config=config)

        node = Node(state=state, prob=0.7, action=1)
        node.append_child(state=state, prob=0.7, action=1)
        child = node.get_child(action=1)
        child.append_child(state=state, prob=0.7, action=1)
        grandchild = child.get_child(action=1)

        tree.backup(grandchild, 1)

        self.assertTrue(node.reward == 1)
        self.assertTrue(child.reward == -1)
        self.assertTrue(grandchild.reward == 1)
    
    def test1_MCTS(self):
        np.random.seed(42)
        state = convert_state(np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 1, 1, 1, 0]]))

        print('Initial State: ')
        render(state, [1, -1, 0])

        tree = SearchTree(state=state, model=model, params=params, config=config)

        # Note that it's always assumed to be player one's turn
        action, prob, terminal, win = tree.MCTS()

        print(tree)

        self.assertTrue(action == 6)

        state = convert_state(np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],[0, 0, 2, 1, 0, 0, 0], [0, 0, 2, 2, 1, 0, 0], [0, 0, 2, 1, 2, 1, 0]]))

        print('Initial State: ')
        render(state, [1, -1, 0])

        tree = SearchTree(state=state, model=model, params=params, config=config)

        # Note that it's always assumed to be player one's turn

        action, prob, terminal, win = tree.MCTS()

        print(tree)
    
    def test2_MCTS(self):
        print('\nTest2')
        print('_' * 30, '\n')
        np.random.seed(42)

        config = {
            "rows": 6, 
            "columns": 7, 
            "inarow": 4,
            "timeout": 2,
            "debug": True
        }

        state = convert_state(np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0],[0, 0, 0, 1, 1, 2, 1], [0, 0, 2, 1, 2, 1, 2], [0, 0, 2, 2, 1, 1, 1]]))

        print('Initial State: ')
        render(state, [1, -1, 0])

        tree = SearchTree(state=state, model=model, params=params, config=config)

        # Note that it's always assumed to be player one's turn
        action, prob, terminal, win = tree.MCTS()

        print('Algorithmn chose: ', action)

        print(tree)
        print('_' * 30)
        self.assertTrue(action == 3)
    
    def test3_MCTS(self):
        print('\nTest2')
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

        tree = SearchTree(state=state, model=model, params=params, config=config)

        # Note that it's always assumed to be player one's turn
        action, probs, terminal, win = tree.MCTS()

        print('Probs: ', probs)

        print('Algorithmn chose: ', action)

        print(tree)
        print('_' * 30)
        self.assertTrue(action == 3)
    
            

if __name__ == '__main__':
    unittest.main()