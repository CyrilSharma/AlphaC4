import numpy as np
from scipy import special
import tensorflow as tf
import time
import copy
from C4_Helpers import convert_state, is_terminal, legal, move, unmove, render
from threading import Thread
from Exceptions import InvalidActionError

class SearchTree():
    def __init__(self, model, params, config, state=None):
        if state is None:
            self.state = np.zeros((config['rows'], config['columns'])) 
        else:
            self.state = np.copy(state)

        self.model = model
        self.columns = config['columns']
        self.rows = config['rows']
        self.model_dims = (1, config['rows'], config['columns'], 1)
        self.root = Node(self.state)
        self.parent = None
        self.tau = params['tau']
        self.c = params['c_puct']
        self.steptime = config['timeout'] - 0.5
        self.cutoff = params['cutoff']
        self.expand(self.root)
    
    def set_state(self, state):
        self.trim(self.root)
        self.state = np.copy(state)
        self.root = Node(np.copy(state))
    
    def move(self, action):
        if action is not None:
            move(self.state, action, 1)
            self.state *= -1

    def unmove(self, action):
        if action is not None:
            self.state *= -1
            unmove(self.state, action)

    @tf.function
    def call_model(self, state: tf.Tensor):
        action_vals, state_val = self.model(state, training=False)
        return action_vals, state_val

    def get_estimates(self, state):
        input_data = tf.convert_to_tensor(state.reshape(self.model_dims), dtype=tf.float32)
        action_vals, state_val = self.call_model(input_data)
        legal_action_vals = action_vals.numpy()[0][legal(state)]
        probs = special.softmax(legal_action_vals)

        return probs, state_val[0].numpy().item()
    
    def MCTS(self):
        if not self.root.terminal:
            t_0 = time.time()
            t_end = time.time() + self.steptime
            while (time.time() < t_end):
                v = self.tree_policy(self.root)
                delta = self.estimate_value(v)
                self.backup(v, delta)
            return self.final_action()
        else:
            raise InvalidActionError(message='Action was taken after the game ended')
    
    def estimate_value(self, v):
        if v.terminal == True:
            if v.win:
                delta = 1
            else:
                delta = 0
        else:
            _, delta = self.get_estimates(self.state)

        return delta
    
    def expand(self, node):
        probs, _ = self.get_estimates(self.state)

        i = 0
        for action in node.actions:
            self.move(action)
            node.append_child(self.state, probs[i], action)
            self.unmove(action)
            i += 1
        
        return node
    
    def tree_policy(self, node):
        while node.terminal == False:
            if len(node.children) == 0:
                return self.expand(node)
            else:
                node = self.choose_action(node)
            
            self.move(node.action)

        return node
    
    def final_action(self):
        if self.tau != 0:
            denom = 0
            rectified_visits = np.zeros(self.columns)

            for action in range(self.columns):
                if action in self.root.children:
                    child = self.root.children[action]
                    x = child.visits ** (1 / self.tau)

                    if x != 0:
                        rectified_visits[action] = x
                    else:
                        rectified_visits[action] = 0.01
                        x = 0.01

                    denom += x
                else:
                    rectified_visits[action] = 0
            
            probs = rectified_visits / denom

            action = np.random.choice(self.columns, p=probs)

        else:
            visits = []
            for action in range(self.columns):
                if action in self.root.children:
                    visits.append(self.root.children[action].visits)
                else:
                    visits.append(0)
            
            action = np.argmax(visits)
            probs = None

        self.shift_root(action)

        return action, probs, self.root.terminal, self.root.win

    def choose_action(self, v):
        Q_a = np.array([v.children[action].q for action in v.children], dtype=np.float32)
        visits = np.array([v.children[action].visits for action in v.children], dtype=np.float32)
        probs = np.array([v.children[action].prob for action in v.children], dtype=np.float32)
        total_visits = np.sum(visits)
        U_a = self.c * probs * np.sqrt(total_visits)/ (1 + visits)
        index = np.argmax(Q_a + U_a)

        return list(v.children.values())[index]

    def shift_root(self, action):
        self.move(action)
        
        actions = copy.deepcopy(self.root.actions)
        actions.remove(action)


        for a in actions:
            self.trim(self.root.get_child(a))
        
        self.parent = copy.deepcopy(self.root)

        self.root = self.root.get_child(action)
        self.root.parent = None
        self.root.action = None

    def trim(self, node):
        for action in node.children:
            child = node.children[action]

            if len(child.children) > 0:
                self.trim(child)
            else:
                del child
            
    def backup(self, v, delta):
        while v is not None:
            self.unmove(v.action)
            v.q += (delta - v.q) / (v.visits + 1)
            v.visits += 1
            v.reward += delta
            delta = -1 * delta
            v = v.parent
    
    def set_tau(self, tau):
        self.tau = tau

    def reset(self):
        self.trim(self.root)
        self.state = np.zeros((self.rows, self.columns)) 
        self.root = Node(self.state)
        self.expand(self.root)

    
    def __str__(self):
        print('\nNew State')
        render(self.state, [1, -1, 0])
        print('Actions')
        print('_' * 30)
        for action in self.parent.actions:
            print('\nAction ', action)
            print('_' * 20)
            child = self.parent.get_child(action)
            print('\nReward: ', child.reward)
            print('Visits: ', child.visits)
            print('Average Reward: ', child.q)
            print('Initial Probability ', child.prob)
            print('Terminal ', child.terminal)
            print('_' * 20, '\n')
        return ''

class Node():
    def __init__(self, state, prob=None, action=None, parent=None):
        self.reward = 0
        self.visits = 0
        self.q = 0
        self.prob = prob
        self.action = action
        self.win, self.terminal = is_terminal(state, action)

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
