from tests.debugTrainers import OverfitTrainer
from ActorCritic import ActorCritic
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

def main():
    with open('parameters.json') as file:
        params = json.load(file)
    with open('config.json') as file:
        config = json.load(file)
    
    state1 = np.array(
        [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0.,  0., 0., 0, 0., 0., 0.],
        [0.,  0., 0., 0., 0., 0., 0.]], dtype=np.float32
    )

    state2 = np.array(
        [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0.,  0., 0., 0, 0., 0., 0.],
        [-1.,  -1., -1., 1., -1., -1., -1.]], dtype=np.float32
    )

    state3 = np.array(
        [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0.,  0., 0., 0, 0., 0., 0.],
        [1.,  1., 1., -1., 1., 1., 1.]], dtype=np.float32
    )

    states = [state2, state3] #, state3]
    rewards = [-0.5, 0.2]

    model = ActorCritic()

    trainer = OverfitTrainer(model, params, config, states, rewards)
    trainer.training_loop()

    model = keras.models.load_model('my_model', custom_objects={"ActorCritic": ActorCritic}, compile=False)

    action_vals, val = model.predict(tf.convert_to_tensor(state2.reshape(1, 6, 7, 1), dtype=tf.float32))
    print(tf.nn.softmax(action_vals))
    print(val)

    action_vals, val = model.predict(tf.convert_to_tensor(state3.reshape(1, 6, 7, 1), dtype=tf.float32))
    print(tf.nn.softmax(action_vals))
    print(val)

main()