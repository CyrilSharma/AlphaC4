from tests.debugTrainers import OverfitTrainer
from ActorCritic import ActorCritic
import json
import numpy as np
import tensorflow as tf
import copy
from tensorflow import keras
from helpers import render
import logging

def main():
    with open('parameters.json') as file:
        params = json.load(file)

    params["num_iters"] = 1
    params["num_eps"] = 30
    params["maxQueueLen"] = 10000
    params["training_args"]["epochs"] = 4
    params["numStoredIters"] = 1
    params["states_per_ep"] = 256


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

    states = [copy.deepcopy(state2), copy.deepcopy(state3)]
    rewards = [-0.5, 0.2]

    model = ActorCritic()

    trainer = OverfitTrainer(model, params, config, states, rewards)
    trainer.training_loop('test')

    model = keras.models.load_model('Models/test', custom_objects={"ActorCritic": ActorCritic}, compile=False)
    input1 = np.stack([copy.deepcopy(states[0]).reshape(6, 7, 1) for j in range(1)]) # (params["training_args"]["batch_size"])])
    input2 = np.stack([copy.deepcopy(states[1]).reshape(6, 7, 1) for j in range(1)]) # params["training_args"]["batch_size"])])

    action_vals, val = model(input1)
    print(tf.nn.softmax(action_vals))
    print(val)

    action_vals, val = model(input2, training=False)
    print(tf.nn.softmax(action_vals))
    print(val)

main()