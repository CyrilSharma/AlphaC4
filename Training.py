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
        self.input_dims = input_shape
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=params["alpha"])
        self.loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    
    def opposite_state(self, state):
        state_opp = state.numpy() * -1
        return tf.convert_to_tensor(state_opp) 

    def env_reset(self) -> np.ndarray:
        self.env.reset()
        observation = self.env.state[0].observation
        return convert_state(np.array(observation['board'], dtype=np.float32).reshape(self.rows, self.columns))

    def tf_env_reset(self) -> tf.Tensor:
        return tf.numpy_function(self.env_reset, [], tf.float32)

    def env_step(self, action: np.ndarray, player:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""

        # weird format that self.env.step wants actions
        actions = [None, None]
        if self.env.state[0].status == "ACTIVE":
            active = 0
            actions[0] = action.item()
        else:
            active = 1
            actions[1] = action.item()
        
        self.env.step(actions)
        
        # gives the board state from the active player's perspective
        observation = self.env.state[active].observation

        if not self.env.done:
            reward = float(0)
            done = False
        else:
            # custom reward scheme
            if self.env.state[player].reward == 1:
                reward = 1
            elif self.env.state[player].reward == 0.5:
                reward = 0
            else:
                reward = -1
                # print(f'Player {player} lost')
            done = True

        return convert_state(np.array(observation['board'], dtype=np.float32).reshape(self.rows, self.columns)), np.array(reward, dtype=np.float32), np.array(done, dtype=np.int32)

    # tensorflow wrapper
    def tf_env_step(self, action: tf.Tensor, player=0) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action, player], [tf.int32, tf.float32, tf.int32])

    def legal(self, state: tf.Tensor, action_vals: tf.Tensor):
        # fill the legal set of actions
        legal_actions = []
        legal_action_values = []
        board = state.numpy().flatten()

        for i in tf.range(self.columns):
            if board[i] == 0:
                legal_actions.append(i)
                legal_action_values.append(action_vals[0, i])
    
        legal_action_values = tf.reshape(tf.convert_to_tensor(legal_action_values, dtype=tf.float32), shape=(1, len(legal_actions)))
        
        return legal_action_values, legal_actions

    def get_action(self, state: tf.Tensor, action_values: tf.Tensor, tau: float):

        # action set needs to be limited to legal actions
        legal_action_values, legal_actions = self.legal(state, action_values)

        tau_tensor = tf.fill(legal_action_values.shape.as_list(), tau)

        legal_action_values = tf.math.divide(legal_action_values, tau_tensor)
        
        # action legal refers to the action index for the legal set
        # output of function is of type [num_batches, num_samples]
        action_legal = tf.random.categorical(legal_action_values, 1)[0, 0]

        # action general refers to the action index for the general set
        action = legal_actions[int(action_legal.numpy().item())]

        # apply softmax over action values to get action probabilities
        action_probs = tf.nn.softmax(legal_action_values)

        prob = action_probs[0, action_legal]

        return action, prob

    def run_episode(self, player=0):
        state = self.tf_env_reset()

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        max_steps = self.rows * self.columns
        tau = self.params["tau"]

        # maximum number of steps
        for t in tf.range(max_steps):
            # for each of the two players...
            for i in range(2):

                if i == 1:
                    state = self.opposite_state(state)

                action_vals, state_val = self.model(tf.reshape(state, self.input_dims))

                action, prob = self.get_action(state, action_vals, tau)

                state, reward, terminal = self.tf_env_step(action, player)

                if player == i:
                    rewards = rewards.write(t, reward)

                    # store value, and removes size 1 dimensions
                    values = values.write(t, tf.squeeze(state_val))
                    
                    # store action values
                    action_probs = action_probs.write(t, prob)

                # exit loop when the episode is over
                if tf.cast(terminal, tf.bool):
                    turn = i
                    break

            if tf.cast(terminal, tf.bool):
                
                # overwrite the last reward the agent received with the reward of the terminal state
                if turn != player:
                    if turn == 0:
                        rewards = rewards.write(tf.math.add(t, - 1), reward)
                    else:
                        rewards = rewards.write(t, reward)
                break

        # convert tensor array to tensor
        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def compute_TD_returns(self, rewards: tf.Tensor, standardize = False) -> tf.Tensor:
        """Compute expected returns per timestep."""

        # batches
        eps = self.params["eps"]
        discount = self.params["gamma"]

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        # ::-1 reverses list, as updating from the terminal state backwards will speed up learning
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)

        # terminal state's value is always zero; no more reward can be obtained
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape


        for i in tf.range(n):
            reward = rewards[i]
            # TD(0) equation for calculating value of state
            discounted_sum = reward + discount * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1]

        # center and scale distibution
        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / 
                    (tf.math.reduce_std(returns) + eps))

        return returns

    def compute_loss(self, action_probs: tf.Tensor,  values: tf.Tensor,  returns: tf.Tensor, loss_function: tf.keras.losses.Loss) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        # TD error
        advantage = tf.math.subtract(returns, values)

        action_log_probs = tf.math.log(action_probs)
        # just computes a sum
        actor_loss = -tf.math.reduce_sum(tf.math.multiply(action_log_probs, advantage)) * 0.001

        critic_loss = loss_function(values, returns)

        # total loss is actor-critic loss
        return [actor_loss, critic_loss]


    def train_step(self, player: int) -> tf.Tensor:

        """Runs a model training step."""

        discount = self.params["gamma"]

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(player)

            # Calculate expected returns
            returns = self.compute_TD_returns(rewards)

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns, self.loss_function)
        
        # Where the magic happens...

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward

    def training_loop(self, save=True):
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
                    render(np.array(self.env.state[0].observation['board']).reshape(self.rows, self.columns))
                    
        if save:
            self.model.save("my_model", save_format='tf')

if __name__ == '__main__':
    main()