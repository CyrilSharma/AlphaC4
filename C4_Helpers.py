from typing import Any, List, Sequence, Tuple
import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from kaggle_environments import evaluate, make, utils
from ActorCritic import ActorCritic
from termcolor import cprint

COLUMN_HEIGHT = 6
ROW_LENGTH = 7
INAROW = 4


def count(state, column, row, mark, offset_row, offset_column):
    for i in range(1, INAROW):
        r = row + offset_row * i
        c = column + offset_column * i
        if (
            r < 0
            or r >= COLUMN_HEIGHT
            or c < 0
            or c >= ROW_LENGTH
            or state[r, c] != mark
        ):
            return i - 1
    return INAROW


def is_win(state: np.ndarray, action: int):
    column = state[:, action]
    zeros = np.count_nonzero(column == 0)
    row = zeros
    mark = state[row, action]

    return (
        (count(state, action, row, mark, 1, 0)) >= INAROW - 1 # vertical.
        or (count(state, action, row, mark, 0, 1) + count(state, action, row, mark, 0, -1)) >= (INAROW - 1)  # horizontal.
        or (count(state, action, row, mark, -1, -1) + count(state, action, row, mark, 1, 1)) >= (INAROW - 1)  # top left diagonal.
        or (count(state, action, row, mark, -1, 1) + count(state, action, row, mark, 1, -1)) >= (INAROW - 1)  # top right diagonal.
    )

def is_terminal(state: np.ndarray, action: int) -> [float, bool]:

    terminal = False

    if action is None:
        return 0.0, False

    if is_win(state, action):
        terminal = True
        reward = 1.0
        return reward, terminal
    
    top_row = state[0, :]

    if (np.count_nonzero(top_row) == ROW_LENGTH):
        terminal = True
        reward = 0.0
    else:
        terminal = False
        reward = 0.0

    return reward, terminal

def unmove(state: np.ndarray, action: int):
    column = state[:, action]
    zeros = np.count_nonzero(column == 0)
    row = zeros
    state[row, action] = 0

def move(state: np.ndarray, action: int, mark: int):
    column = state[:, action]
    zeros = np.count_nonzero(column == 0)
    row = zeros - 1
    state[row, action] = mark

def legal(state: np.ndarray):
    legal_actions = []
    row = state[0, :]

    for action in range(len(row)):
        if row[action] == 0:
            legal_actions.append(action)
        
    return legal_actions

def convert_state(state: np.ndarray):
    state_opp = state
    state_opp[state_opp == 2] = -1
    return state_opp

def render(state, token_nums=[1, 2, 0]):
    print()

    p1, p2, space = token_nums

    j = 5

    # print board
    for row in state:
        # print vertical labels.
        print(j, end = " ")
        for token in row:
            # Print terms in color.
            if token == p1:
                cprint('HHHH', 'red', end = " ")
            elif token == p2:
                cprint('HHHH', 'blue', end = " ")  
            else:
                print('HHHH', end = " ")
        j -= 1
        # place next row on new line.
        print()
    
    # align number with tokens on next line.
    print("  ", end = "")

    # print horizontal labels.
    for i in range(7):
        print(i, end = "    ")
    print("", end = "\n\n")








    