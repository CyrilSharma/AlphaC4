# %%
import json
from typing import Any, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tqdm

from tensorflow import keras
from ActorCritic import ActorCritic
from C4_Helpers import render
from MCTS import SearchTree
from matplotlib import pyplot as plt

class C4Trainer():
    def __init__(self, model, params, config, input_shape=(1,6,7,1)):
        self.rows = config['rows']
        self.columns = config['columns']
        self.params = params
        self.config = config
        self.model = model
        self.tree = SearchTree(self.model, params, config)
        self.input_dims = input_shape
        self.optimizer = keras.optimizers.Adam(learning_rate=params["alpha"])

    @tf.function
    def call_model(self, state: tf.Tensor):
        action_vals, state_val = self.model(state, training=False)
        return action_vals, state_val
    
    def get_probs(self, action_values):
        idx = tf.constant(legal(self.tree.state), tf.int32)
        legal_action_vals = tf.gather(tf.squeeze(action_values), idx)
        legal_probs = tf.nn.softmax(legal_action_vals)

        probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        i = tf.constant(0, dtype=tf.int32)
        for action in tf.range(self.columns):
            if action in idx:
                probs = probs.write(action, legal_probs[i])
                i = i + tf.constant(1, dtype=tf.int32)
            else:
                probs = probs.write(action, tf.constant(0, dtype=tf.float32))
        return probs.stack()
    
    def get_log_probs(self, probs):
        idx = legal(self.tree.state)

        log_probs = []

        i = 0
        for prob in probs:
            if i in idx:
                try:
                    log_probs.append(np.log(prob))
                except RuntimeWarning:
                    print('Hehe got em')
            else:
                log_probs.append(0.0)
            
            i += 1
        
        return tf.convert_to_tensor(log_probs, dtype=tf.float32)

    def run_episode(self, player=0):
        self.tree.reset()
        initial_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        final_log_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        max_steps = self.rows * self.columns

        terminals = []

        # maximum number of steps
        for t in tf.range(max_steps):
            # for each of the two players...
            for i in range(2):
                
                input_data = tf.convert_to_tensor(self.tree.state.reshape(1, self.rows, self.columns, 1), dtype=tf.float32)
                action_values, state_val = self.call_model(input_data)

                init_probs = self.get_probs(action_values)
            
                action, probs, terminal, win = self.tree.MCTS()

                final_log_probs = self.get_log_probs(probs)

                if player == i:
                    # store value, and removes size 1 dimensions
                    values = values.write(t, tf.squeeze(state_val))
                    
                    # store inital probability distribution
                    initial_prob_list = initial_prob_list.write(t, init_probs)

                    # store improved probability distribution
                    final_log_prob_list = final_log_prob_list.write(t, final_log_probs)

                # exit loop when the episode is over
                if terminal:
                    turn = i
                    break

            if terminal:  
                if not win:
                    reward = tf.constant(0, dtype=tf.float32)
                else:
                    if turn == player:
                        reward = tf.constant(1, dtype=tf.float32)
                    else:
                        reward = tf.constant(-1, dtype=tf.float32)
                break
        
        # convert tensor array to tensor
        initial_prob_list = initial_prob_list.stack()
        final_log_prob_list = final_log_prob_list.stack()
        values = values.stack()

        return initial_prob_list, final_log_prob_list, values, reward

    def compute_loss(self, initial_prob_list: tf.Tensor, final_log_prob_list: tf.Tensor, values: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        c = self.params["c"]

        variables = self.model.trainable_variables

        loss_L2 = tf.add_n([ tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name ]) * c

        cross_entropy = tf.math.reduce_sum(initial_prob_list * final_log_prob_list)

        mse = tf.math.reduce_sum(tf.math.square(values - reward))
        
        loss = mse + cross_entropy + loss_L2

        # total loss is actor-critic loss
        return loss


    def train_step(self, player: int) -> tf.Tensor:

        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            initial_prob_list, final_log_prob_list, values, reward = self.run_episode(player)

            # Calculating loss values to update our network
            loss = self.compute_loss(initial_prob_list, final_log_prob_list, values, reward)
        
        # Where the magic happens...

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return reward.numpy(), loss.numpy()
    
    def test_network(self, num, episodes=20):

        model_old = keras.models.load_model(f"Models/v{num}", compile=False)

        old_tree = SearchTree(model_old, self.params, self.config)

        self.tree.reset()

        trees = [self.tree, old_tree]

        avg_reward = 0
    
        for ep_num in range(episodes):
            # reset trees, and make tau non zero
            self.tree.reset()
            old_tree.reset()
            self.tree.set_tau(self.params['tau'])
            old_tree.set_tau(self.params['tau'])

            player = np.random.randint(2)

            if player != 0:
                action, _, terminal, win = old_tree.MCTS()
                self.tree.shift_root(action)

            terminal = False
            i = 0
            
            while not terminal:
                # after a few moves, play greedily
                if i == self.params['exploratory_turns']:
                    self.tree.set_tau(0)
                    old_tree.set_tau(0)

                action, _, terminal, win = trees[i % 2].MCTS()

                if terminal:
                    break

                trees[(i + 1) % 2].shift_root(action)

                i += 1
            
            if win:
                if (i % 2) == 0:
                    reward = 1
                else:
                    reward = -1
            else:
                reward = 0

            render(trees[i%2].state * -1, [1, -1, 0])

            avg_reward += (reward - avg_reward) / (ep_num + 1)
        
        self.tree.set_tau(self.params['tau'])
        print(f'New network scored an average reward of: {avg_reward} against the old network.')

    def training_loop(self, display=True, save=True, graphs=True):
        episodes = self.params["episodes"]

        running_reward = 0
        running_loss = 0

        episode_nums = list(range(1, episodes + 1))
        losses = [0 for num in range(episodes)]

        version = 1

        with tqdm.trange(episodes) as t:
            for i in t:

                player = np.random.randint(2)

                episode_reward, loss = self.train_step(player)

                losses[i] = loss

                running_reward = episode_reward*0.01 + running_reward*.99

                running_loss = loss * 0.01 + running_loss * .99

                t.set_description(f'Episode {i}')
                t.set_postfix(episode_loss=loss, running_loss=running_loss)

                if i == self.params["test_every"] - 1:
                    self.model.save(f"Models/v{version}", save_format='tf')
                    version += 1

                if (((i + 1) % self.params["test_every"]) == 0 and i > (self.params["test_every"] - 1)):
                    self.model.save(f"Models/v{version}", save_format='tf')
                    self.test_network(version - 1)
                    version += 1

                # Show average episode reward every 10 episodes
                if (i % 5 == 0 and display):
                    print(f'Episode {i}: average loss: {running_loss}')
                    # print(self.tree)
                    render(self.tree.state * -1, [1, -1, 0])

        
        if graphs:
            plt.title("Loss over time") 
            plt.xlabel("Episode") 
            plt.ylabel("Loss") 
            plt.plot(episode_nums,losses) 
            plt.show()
                    
        if save:
            self.model.save("my_model", save_format='tf')

if __name__ == '__main__':
    main()
# %%
