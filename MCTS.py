import numpy as np
import time
import tensorflow as tf
import copy
import sys
from tqdm import tqdm
from C4 import C4

class Node():
    def __init__(self, num_actions=7, parent=None, action=-1, prob=0):
        self.parent = parent
        self.children = [None] * num_actions
        self.visits = 0
        self.prob = prob
        self.action = action
        self.expanded = False
        self.q = 0

    def best_child(self, c_puct):

        best_value = -10000000000
        best_move = 0

        for i in range(len(self.children)):
            # empty node
            if (self.children[i] is None):
                continue

            total_child_visits = self.visits + 1
            cur_value = self.children[i].PUCT(c_puct, self.visits + 1)

            if (cur_value > best_value):
                best_value = cur_value
                best_move = i
                best_node = self.children[i]

        return best_move

    def expand(self, child_probs):
        if (not self.expanded):
            for i in range(len(self.children)):
                # illegal action
                if abs(child_probs[i]) < 0.000001:
                    continue

                self.children[i] = Node(parent=self, action=i, prob=child_probs[i], num_actions=len(self.children))

            self.expanded = True

    def backup(self, value):
        # If it is not root, this node's parent should be updated first
        if (self.parent is not None):
            self.parent.backup(-value)

        # update visits
        visits = self.visits
        self.visits += 1

        # update q
        # std::lock_guard<std::mutex> lock(self.lock)
        self.q += (value - self.q) / (visits + 1)

    def PUCT(self, c_puct, total_child_visits):
        # u
        visits = self.visits
        u = (c_puct * self.prob * np.sqrt(total_child_visits) / (1 + visits))

        if (visits <= 0):
            return u
        else:
            # average reward with virtual loss
            return u + self.q

class MCTS():
    def __init__(self, model, t, c_puct, board_dims=[6,7]):
        self.model = model
        self.model_dims = (1, board_dims[0], board_dims[1], 1)
        self.dims = board_dims
        self.c_puct = c_puct
        self.num_actions = board_dims[1]
        self.root = Node()
        self.timeout = t

        # initial call to function to compile it
        _, _ = call_model(self.model, tf.convert_to_tensor(np.zeros(self.model_dims), dtype=tf.float32))

    def shift_root(self, last_action):

        # reuse the child tree
        if (last_action >= 0 and self.root.children[last_action] is not None):
            # unlink
            new_node = self.root.children[last_action]
            # so that the new tree isn't deleted
            self.root.children[last_action] = None
            new_node.parent = None

            # assigns root to new node and deletes the rest of the tree
            self.tree_deleter(self.root)
            self.root = new_node
    
    def reset(self):
        self.tree_deleter(self.root)
        self.root = Node()

    def final_probs(self, C4, temp):
        init_probs, init_val = self.predict(C4)
        # initialize probs
        action_probs = np.zeros(self.num_actions)

        t_0 = time.time()
        t_end = time.time() + self.timeout

        i = 0

        while time.time() < t_end:
            game = C4.clone()
            self.update(game)
            i += 1

        children = self.root.children
    
        # this chooses whatever action had the most visits if the temp was 0
        if (temp - 1e-3 < 0.0000001):
            max_count = 0
            best_action = 0

            for i in range(len(children)):
                # if child exists and number of child visits is greater then the max
                if (children[i] and children[i].visits > max_count):
                    max_count = children[i].visits
                    best_action = i

            action_probs[best_action] = 1.

        else:
            # explore
            sum = 0
            for i in range(len(children)):
                if (children[i] is not None and children[i].visits > 0):
                    action_probs[i] = pow(children[i].visits, 1 / temp)
                    sum += action_probs[i]

            # renormalization
            action_probs = action_probs / sum


        return (init_probs, action_probs, init_val)

    def update(self, game):

        node = self.root
        action = node.action

        # search down the tree until you reach a state that has not been explored
        while (True):
            if (not node.expanded):
                break
            
            # select
            action = node.best_child(self.c_puct)
            game.move(action)
            node = node.children[action]
        
        # get game status
        if (action == -1):
            reward, terminal = (0,0)
        
        else:
            reward, terminal = game.is_terminal(action)

        # if not terminal
        if (terminal == 0):

            result = self.predict(game)
            probs = result[0]
            value = result[1][0]

            # expand
            node.expand(probs)

        # value(parent . node) = -value
        node.backup(reward)
        return

    def predict(self, game):
        input_data = tf.convert_to_tensor(game.state.reshape(self.model_dims) * game.player, dtype=tf.float32)
        probs_tensor, value_tensor = call_model(self.model, input_data)

        probs = probs_tensor[0].numpy()
        value = value_tensor[0].numpy()

        # mask invalid actions
        legal_moves = game.legal()

        # sum legal action values
        sum = 0
        for i in range(self.num_actions):
            if (legal_moves[i] == 1):
                sum += probs[i]
            else:
                probs[i] = 0
            
        

        # renormalization
        if (sum > 0.00000001):
            probs = probs / sum
        else:
            # number of legal moves
            sum = np.sum(legal_moves)

            # initialize probs of legal moves to random distribution
            probs = legal_moves / sum

        return (probs, value)
    
    def tree_deleter(self, n: Node):
        if (n is None):
            return

        # remove children
        for i in range(len(n.children)):
            if (n.children[i] is not None):
                self.tree_deleter(n.children[i])

        # remove self
        del n

@tf.function
def call_model(model, input_data: tf.Tensor):
    action_value_tensor, value_tensor = model(input_data, training=False)
    # NOTE: This was changed! NN no longer outputs a softmax of probabilities!
    probs_tensor = tf.nn.softmax(action_value_tensor)
    return (probs_tensor, value_tensor)

