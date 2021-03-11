from kaggle_environments import evaluate, make, utils
import numpy as np
from time import sleep


def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


env = make("connectx", debug=True)
trainer = env.train([None, "negamax"])
env.reset()

while not env.done:
    env.render()
    action = int(input("Action: ")) - 1
    trainer.step(action)
    env.render()