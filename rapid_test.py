from kaggle_environments import evaluate, make, utils
import numpy as np
from time import sleep

rows = 6
columns = 7

def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

def main():
    env = make("connectx", debug=True)
    trainer = env.train([None, "negamax"])
    env.reset()

    i = -1
    while not env.done:
        env.render()
        action = int(input("Action: "))
        i += 1
        trainer.step(action)
        i += 1
        print(env.steps[i][0]['action'])
        env.render()

    # print(np.array(env.state[0].observation['board']).reshape(rows, columns))

main()