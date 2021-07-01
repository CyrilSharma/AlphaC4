import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras
from ActorCritic import ActorCritic
from collections import deque
from helpers import render
from MCTS import MCTS
from Battle import battle
from C4 import C4
from matplotlib import pyplot as plt
import sys



class Trainer():
    def __init__(self, model, params, config, input_shape=(1,6,7,1)):
        self.rows = config['rows']
        self.columns = config['columns']
        self.params = params
        self.config = config
        self.model = model
        self.tree = MCTS(model, config["timeout"], params["c_puct"])
        self.input_dims = input_shape
        self.optimizer = keras.optimizers.Adam(learning_rate=params["alpha"])

    def run_episode(self, player=1):
        self.tree.reset()
        game = C4()

        initial_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        final_prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        turn = 0
        terminal = False
        
        episode_memory = []

        while not terminal:
            # get data
            init_probs, final_probs, state_val = self.tree.final_probs(game, self.params["temp"])

            # choose action and modify state accordingly
            action = np.random.choice([0,1,2,3,4,5,6], p=final_probs.numpy())
            game.move(action)
            self.tree.shift_root(action)

            if game.player == player:
                # store value, and removes size 1 dimensions
                values = values.write(turn, tf.squeeze(state_val))
                initial_prob_list = initial_prob_list.write(turn, init_probs)
                final_prob_list = final_prob_list.write(turn, final_probs)
                episode_memory.append([game.state, final_probs, None])
                turn += 1
                
            reward, terminal = game.is_terminal(action)

        if reward == 0:
            reward = tf.constant(0, dtype=tf.float32)
        else:
            if game.player == player:
                reward = tf.constant(1, dtype=tf.float32)
            else:
                reward = tf.constant(-1, dtype=tf.float32)
        
        # convert tensor array to tensor
        initial_prob_list = initial_prob_list.stack()
        final_prob_list = final_prob_list.stack()
        values = values.stack()

        # return initial_prob_list, final_prob_list, values, reward

    def compute_loss(self, initial_prob_list: tf.Tensor, final_prob_list: tf.Tensor, values: tf.Tensor, rewards: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        c = self.params["c"]

        variables = self.model.trainable_variables

        loss_L2 = tf.add_n([ tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name ]) * c

        cross_entropy = tf.reduce_sum(tf.multiply(-final_prob_list, tf.log(initial_prob_list))) / len(final_prob_list)
        print(f"final_prob_list {final_prob_list}")
        print(f"initial_prob_list {initial_prob_list}")
        print(f"cross entropy {cross_entropy}")
        mse = tf.keras.losses.MeanSquaredError()
        mse_loss = mse(rewards, values)
        print(f"rewards {rewards}")
        print(f"values {values}")
        print(f"mse {mse_loss}")
        loss = mse_loss + cross_entropy + loss_L2

        # total loss is actor-critic loss
        return loss


    def train_step(self) -> tf.Tensor:

        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # player is either +- 1
            player = np.random.randint(2) * 2 - 1

            # Run the model for one episode to collect training data
            initial_prob_list, final_prob_list, values, reward = self.run_episode(player)

            # THIS NEEDS TO CHANGE, REPLAY BUFFER NEEDS TO BE IMPLEMENTED
            # Calculating loss values to update our network
            loss = self.compute_loss(initial_prob_list, final_prob_list, values, reward)
        
        # Where the magic happens...

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return reward.numpy(), loss.numpy()
    
    def test_network(self, num, episodes=20):

        model_old = keras.models.load_model(f"Models/v{num}", compile=False)

        old_tree = MCTS(model_old, config["timeout"], params["c_puct"])

        avg_reward = 0
    
        for ep_num in range(episodes):
            # reset trees, and make tau non zero
            self.tree.reset()
            old_tree.reset()

            swap = np.random.choice([True, False])

            reward = battle(self.tree, old_tree, swap)

            avg_reward += (reward - avg_reward) / (ep_num + 1)
        
        self.tree.reset()

        print(f'New network scored an average reward of: {avg_reward} against the old network.')

    def training_loop(self, save=True, graphs=True):

        episodes = self.params["episodes"]
        iters = self.params["iterations"]

        running_reward = 0
        running_loss = 0

        episode_nums = list(range(1, episodes + 1))
        losses = [0 for num in range(episodes)]

        version = 1

        for j in range(iters):
            with tqdm.trange(episodes) as t:

                iterationTrainExamples = deque([], maxlen=self.params["maxlenOfQueue"])

                for i in t:
                    episode_reward, loss = self.train_step()

                    if ((i % self.params["test_every"] - 1) == 0):
                        #self.model.save(f"Models/v{version}", save_format='tf')
                        #self.test_network(version - 1)
                        version += 1

                    losses.append(loss)

                    running_reward = episode_reward*0.01 + running_reward*.99
                    running_loss = loss * 0.01 + running_loss * .99

                    t.set_description(f'Episode {i}')
                    t.set_postfix(episode_loss=loss, running_loss=running_loss)

        
        if graphs:
            plt.title("Loss over time") 
            plt.xlabel("Episode") 
            plt.ylabel("Loss") 
            plt.plot(episode_nums,losses) 
            plt.show()
                    
        if save:
            self.model.save("my_model")