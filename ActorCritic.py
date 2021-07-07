from typing import Any, List, Sequence, Tuple
import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras
from tensorflow.keras import layers

class ActorCritic(tf.keras.Model):
  """ Combined Actor-Critic network.
  The idea here is the information the common layers extract can and should be shared with both the actor and the critic """

  def __init__(self, filters=32):
    """Initialize."""
    super().__init__()
    self.tower_height = 15
    self.common = [ResidualBlock() for i in range(self.tower_height)]
    self.actor = PolicyBlock()
    self.critic = ValueBlock()

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = inputs
    for i in range(self.tower_height):
        x = self.common[i](x)
    return self.actor(x), self.critic(x)

class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters=32):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=(4,4), padding="same", data_format="channels_last", input_shape=(1,6,7,1))
        # self.batch_n1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size=(4,4), padding="same", data_format="channels_last")


    def call(self, inputs):
        x = self.conv1(inputs)
        # x = self.batch_n1(x)
        x = tf.keras.activations.relu(x)
        x = self.conv2(x)
        # x = self.batch_n1(x)
        x = x + inputs
        x = tf.keras.activations.relu(x)
        return x

class PolicyBlock(keras.layers.Layer):
    def __init__(self, columns=7, filters=32):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=(2,2), padding="same", data_format="channels_last")
        # self.batch_n1 = layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = layers.Dense(columns)


    def call(self, inputs):
        x = self.conv1(inputs)
        # x = self.batch_n1(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class ValueBlock(keras.layers.Layer):
    def __init__(self, filters=32, hidden_units=128):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=(1,1), padding="same", data_format="channels_last")
        # self.batch_n1 = layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = layers.Dense(hidden_units, activation="relu")

        # state values can only be between -1 (game lost) and 1 (game won)
        self.dense2 = layers.Dense(1, activation="tanh")

    def call(self, inputs):
        x = self.conv1(inputs)
        # x = self.batch_n1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x