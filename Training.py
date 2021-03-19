import json
from typing import Any, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tqdm
from kaggle_environments import evaluate, make, utils
from tensorflow import keras
from tensorflow.keras import layers

from ActorCritic import ActorCritic
from C4_Helpers import convert_state, render
from MCTS import SearchTree

def main():
    # Opening JSON file 
    with open('parameters.json') as f:
        params = json.load(f) 

    with open('C4_Config.json') as f:
        config = json.load(f) 

    trainer = C4Trainer(params, config)
    trainer.training_loop()

class C4Trainer():
    def __init__(self, params, config, input_shape=(1,6,7,1)):
        self.env = make("connectx", config)
        self.rows = config['rows']
        self.columns = config['columns']
        self.params = params
        self.model = ActorCritic()
        self.tree = SearchTree(self.model, params, config)
        self.input_dims = input_shape
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=params["alpha"])

    def run_episode(self, player=0):
        self.tree.reset()
        initial_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        final_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        max_steps = self.rows * self.columns
        tau = self.params["tau"]

        terminals = []

        # maximum number of steps
        for t in tf.range(max_steps):
            # for each of the two players...
            for i in range(2):

                init_probs, state_val = self.model(tf.convert_to_tensor(self.tree.state.reshape(1, self.rows, self.columns, 1), dtype=tf.float32))
            
                action, probs, terminal, win = self.tree.MCTS()

                """ if (i == 0):
                    render(self.tree.state, [1, -1, 0])
                else:
                    render(self.tree.state * -1, [1, -1, 0]) """

                terminals.append(terminal)

                final_probs = tf.convert_to_tensor(probs, dtype=tf.float32)

                if player == i:

                    # store value, and removes size 1 dimensions
                    values = values.write(t, tf.squeeze(state_val))
                    
                    # store inital probability distribution
                    initial_prob_list = initial_prob_list.write(t, init_probs[0])

                    # store improved probability distribution
                    final_prob_list = final_prob_list.write(t, final_probs)

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
        final_prob_list = final_prob_list.stack()
        values = values.stack()

        return initial_prob_list, final_prob_list, values, reward

    def compute_loss(self, initial_prob_list: tf.Tensor, final_prob_list: tf.Tensor, values: tf.Tensor,  reward: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        c = self.params["c"]

        variables = self.model.trainable_variables

        loss_L2 = tf.add_n([ tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name ]) * c

        cross_entropy = tf.math.reduce_sum(initial_prob_list * tf.math.log(final_prob_list))

        mse = tf.math.square(values - reward)
        
        loss = mse + cross_entropy + loss_L2

        # total loss is actor-critic loss
        return loss


    def train_step(self, player: int) -> tf.Tensor:

        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            initial_prob_list, final_prob_list, values, reward = self.run_episode(player)

            # Calculating loss values to update our network
            loss = self.compute_loss(initial_prob_list, final_prob_list, values, reward)
        
        # Where the magic happens...

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return reward

    def training_loop(self, save=True):
        np.random.seed(42)

        episodes = self.params["episodes"]

        running_reward = 0

        with tqdm.trange(episodes) as t:
            for i in t:

                player = np.random.randint(2)

                episode_reward = int(self.train_step(player))

                running_reward = episode_reward*0.01 + running_reward*.99

                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    print(f'Episode {i}: average reward: {running_reward}')
                    render(self.tree.state, [1, -1, 0])
                    
        if save:
            self.model.save("my_model", save_format='tf')

if __name__ == '__main__':
    main()