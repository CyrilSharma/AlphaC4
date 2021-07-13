import numpy as np
from numpy.random import default_rng
import time
import tensorflow as tf
import copy
import sys
from tqdm import tqdm
from C4 import C4
import logging
import copy
class Node():
    def __init__(self, num_actions=7, parent=None, action=-1, prob=0):
        self.parent = parent
        self.children = [None] * num_actions
        self.visits = 0
        self.prob = prob
        self.action = action
        self.expanded = False
        # for debugging
        self.terminal = 0.0
        self.q = 0.0

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

        return best_move

    def expand(self, child_probs):
        if (not self.expanded):
            for i in range(len(self.children)):
                # illegal action
                if abs(child_probs[i]) < np.finfo(np.float32).eps:
                    continue

                self.children[i] = Node(parent=self, action=i, prob=child_probs[i], num_actions=len(self.children))

            self.expanded = True
        else:
            for i in range(len(self.children)):
                # illegal action
                if abs(child_probs[i]) < np.finfo(np.float32).eps:
                    continue

                if (self.children[i] is not None):
                    self.children[i].prob = child_probs[i]
                    
    def backup(self, value):
        # If it is not root, this node's parent should be updated first
        if (self.parent is not None):
            self.parent.backup(-value)
        
        # update visits
        self.visits += 1

        # update q
        self.q += (value - self.q) / self.visits


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
    def __init__(self, model, params, board_dims=[6,7]):
        self.model = model
        self.model_dims = (1, board_dims[0], board_dims[1], 1)
        self.dims = board_dims
        self.c_puct = params["c_puct"]
        self.epsilon = params["epsilon"]
        self.dirichlet = params["dirichlet"] 
        self.num_actions = board_dims[1]
        self.root = Node()
        self.timeout = params["timeout"]
        self.num_sims = params["num_sims"]
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
        # adds noise to the root node.
        # initial reading of state value is stored for testing purposes.
        init_probs, init_val = self.predict(C4)
        probs = self.add_noise(C4, init_probs)
        
        # add noise to root node only
        self.root.expand(probs)

        # initialize probs
        action_probs = np.zeros(self.num_actions)

        t_0 = time.time()
        t_end = time.time() + self.timeout
        i = 0

        # don't spend more then self.timeout seconds on this and don't do more then simulations than self.num_sims
        while time.time() < t_end and i < self.num_sims:
            game = copy.deepcopy(C4)
            self.update(game)
            i += 1

        children = self.root.children

        # for logging purposes
        probs = []
        visits = []
        qs = []
        terminals = []

        # explore
        sum = 0
        for i in range(self.dims[1]):
            probs.append("{:.2f}".format(round(children[i].prob,2)) if children[i] is not None else "Illegal")
            visits.append(children[i].visits if children[i] is not None else "Illegal")
            qs.append("{:.2f}".format(round(children[i].q, 2)) if children[i] is not None else "Illegal")
            terminals.append(children[i].terminal if children[i] is not None else "Illegal")

            if (children[i] is not None and children[i].visits > 0):
                action_probs[i] = pow(children[i].visits, 1 / temp)
                sum += action_probs[i]

        # renormalization
        action_probs = action_probs / sum

        logging.debug("Probs: " + str(probs))
        logging.debug("Visits: " + str(visits))
        logging.debug("Qs: " + str(qs))
        logging.debug("Terminals: " + str(terminals))
        logging.debug("_" * 20)

        return (action_probs, init_val)
    
    def add_noise(self, C4, init_probs):
        legal_moves = C4.legal()
        # dirichlet distribution over the legal moves
        noise = self.epsilon * default_rng().dirichlet(self.dirichlet * np.ones(np.count_nonzero(legal_moves)))
        probs = (1 - self.epsilon) * init_probs
        
        # add dirichlet noise to init_probs, then normalize
        j = 0
        for i in range(len(probs)):
            if legal_moves[i] == 1:
                probs[i] += noise[j]
                j += 1

        probs /= np.sum(probs)
        
        return probs

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
        
        reward, terminal = game.is_terminal(action)
        node.terminal = terminal

        # if not terminal
        if (terminal == 0):
            probs, value = self.predict(game)
            node.expand(probs)
            reward = value
        else:
            print("", end="")

        node.backup(reward)
        return

    def predict(self, game):
        input_data = tf.convert_to_tensor(copy.deepcopy(game.state).reshape(self.model_dims) * game.player, dtype=tf.float32)
        probs_tensor, value_tensor = call_model(self.model, input_data)

        probs = probs_tensor[0].numpy()
        value = value_tensor[0].numpy().item()

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

