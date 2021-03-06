from typing import Any, List, Sequence, Tuple
import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras
from tensorflow.keras import layers

class ActorCritic(tf.keras.Model):
  """ Combined Actor-Critic network.
  The idea here is the information the common layers extract can and should be shared with both the actor and the critic """

  def __init__(self, filters=128):
    """Initialize."""
    super().__init__()
    self.tower_height = 20
    self.common = [ResidualBlock(filters) for i in range(self.tower_height)]
    self.actor = PolicyBlock()
    self.critic = ValueBlock()

  def call(self, inputs: tf.Tensor, training=False) -> Tuple[tf.Tensor, tf.Tensor]:
    x = inputs # tf.add(inputs, 0.5)
    for i in range(self.tower_height):
        x = self.common[i](x, training)
    return self.actor(x, training), self.critic(x, training)

class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters=128, c=0.00001):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=(3,3), padding="same", data_format="channels_last", bias_regularizer=keras.regularizers.l2(c))
        self.batch_n1 = layers.BatchNormalization(momentum=0.900)
        self.batch_n2 = layers.BatchNormalization(momentum=0.900)
        self.conv2 = layers.Conv2D(filters, kernel_size=(3,3), padding="same", data_format="channels_last", bias_regularizer=keras.regularizers.l2(c))


    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.batch_n1(x, training)
        x = self.conv2(x)
        x = self.batch_n2(x, training)
        x = x + inputs
        x = tf.nn.relu(x)
        return x

class PolicyBlock(keras.layers.Layer):
    def __init__(self, columns=7, filters=32, hidden_units=256, c=0.001):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=(2,2), padding="same", data_format="channels_last", bias_regularizer=keras.regularizers.l2(c))
        self.batch_n1 = layers.BatchNormalization(momentum=0.900)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = layers.Dense(columns, bias_regularizer=keras.regularizers.l2(c))


    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.batch_n1(x, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class ValueBlock(keras.layers.Layer):
    def __init__(self, filters=32, hidden_units=256, c=0.001):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=(1,1), padding="same", data_format="channels_last", bias_regularizer=keras.regularizers.l2(c))
        self.batch_n1 = layers.BatchNormalization(momentum=0.900)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = layers.Dense(hidden_units, activation="relu", bias_regularizer=keras.regularizers.l2(c))

        # state values can only be between -1 (game lost) and 1 (game won)
        self.dense2 = layers.Dense(1, activation="tanh", bias_regularizer=keras.regularizers.l2(c))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.batch_n1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x