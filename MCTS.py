import numpy as np
from scipy import special
import tensorflow as tf
import time
import copy
from C4_Helpers import convert_state, is_terminal, legal, move, unmove, render

class SearchTree():
    def __init__(self, state, model, params, config):
        self.state = state
        self.model = model
        self.model_dims = (1, config['rows'], config['columns'], 1)
        self.root = Node(self.state)
        self.tau = params['tau']
        self.c = params['c_puct']
        self.steptime = config['timeout'] - 0.5

    def model_call(self, state):
        action_vals, state_val = self.model(tf.convert_to_tensor(state.reshape(self.model_dims), dtype=tf.float32))
        legal_action_vals = action_vals.numpy()[0][legal(state)]
        probs = special.softmax(legal_action_vals)
        return probs, state_val[0].numpy().item()
    
    def MCTS(self):
        if not self.root.terminal:
            t_end = time.time() + self.steptime
            self.expand(self.root)
            while (time.time() < t_end):
                v = self.tree_policy(self.root)
                delta = self.estimate_value(v)
                self.backup(v, delta)
                #print(self.root.get_child(1).reward, self.root.get_child(1).visits)
            return self.final_action()
    
    def estimate_value(self, v):
        move(self.state, v.action, 1)

        if v.terminal == True:
            delta = 1
        else:
            _, delta = self.model_call(self.state * -1)
        
        unmove(self.state, v.action)

        return delta
    
    def expand(self, node):
        probs, _ = self.model_call(self.state)

        i = 0
        for action in node.actions:
            move(self.state, action, 1)
            node.append_child(self.state * -1, probs[i], action)
            unmove(self.state, action)
            i += 1

        return node
    
    def tree_policy(self, node):
        while node.terminal == False:
            if len(node.children) == 0:
                return self.expand(node)
            else:
                node = self.choose_action(node)
        return node
    
    def final_action(self):
        denom = 0
        rectified_visits = np.zeros(len(self.root.actions))
        for action in self.root.actions:
            child = self.root.children[action]
            x = child.visits ** (1 / self.tau)
            rectified_visits[action] = x
            denom += x
        
        probs = rectified_visits / denom

        # print('Probs: ', probs)

        index = np.random.choice(len(self.root.actions), p=probs)

        action = self.root.actions[index]

        self.shift_root(action)

        return action, probs[index], self.root.terminal

    def choose_action(self, v):
        Q_a = np.array([v.children[action].q for action in v.children], dtype=np.float32)
        visits = np.array([v.children[action].visits for action in v.children], dtype=np.float32)
        probs = np.array([v.children[action].prob for action in v.children], dtype=np.float32)
        total_visits = np.sum(visits)
        U_a = self.c * probs * np.sqrt(total_visits)/ (1 + visits)
        index = np.argmax(Q_a + U_a)
        return v.children[index]

    def shift_root(self, action):
        move(self.state, action, 1)
        self.state *= -1
        
        actions = copy.deepcopy(self.root.actions)
        actions.remove(action)

        for a in actions:
            self.trim(self.root.get_child(a))

        self.root = self.root.get_child(action)

    def trim(self, node):
        for action in node.children:
            child = node.children[action]

            if len(child.children) > 0:
                self.trim(child)
            else:
                del child
            
    def backup(self, v, delta):
        while v is not None:
            v.q += (delta - v.q) / (v.visits + 1)
            v.visits += 1
            v.reward += delta
            delta = -1 * delta
            v = v.parent
    
    def __str__(self):
        print('\nNew State')
        render(self.state, [1, -1, 0])
        print('Actions')
        print('_' * 30)
        for action in self.root.parent.actions:
            print('\nAction ', action)
            print('_' * 20)
            child = self.root.parent.get_child(action)
            print('\nReward: ', child.reward)
            print('Visits: ', child.visits)
            print('Average Reward: ', child.q)
            print('Initial Probability ', child.prob)
            print('_' * 20, '\n')
        return ''




class Node():
    def __init__(self, state, prob=None, action=None, parent=None):
        self.reward = 0
        _, self.terminal = is_terminal(state, action)
        self.visits = 0
        self.q = 0
        self.prob = prob

        self.action = action

        if self.terminal:
            self.actions = []
        else:
            self.actions = legal(state)

        self.parent = parent
        self.children = {} 
    
    def append_child(self, state, prob, action):
        self.children[action] = Node(state, prob, action, self)
    
    def get_child(self, action):
        return self.children[action]
