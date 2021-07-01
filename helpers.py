import numpy as np
from termcolor import cprint

def state_to_action(state_old: np.ndarray, state_new: np.ndarray):
    """ Takes in a new state and outputs the action that was taken) """
    for action in range(7):
        column_old = state_old[:, action]
        column_new = state_new[:, action]
        diff = np.count_nonzero(column_new) - np.count_nonzero(column_old)

        if diff == 1:
            return action
    
    return None

def render(state, token_nums=[1, -1, 0]):
    """ Renders a state
        token_nums = [Player 1 token, Player 2 token, empty token]
    """
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
                cprint('HH', 'red', end = " ")
            elif token == p2:
                cprint('HH', 'blue', end = " ")  
            else:
                print('HH', end = " ")
        j -= 1
        # place next row on new line.
        print()
    
    # align number with tokens on next line.
    print("  ", end = "")

    # print horizontal labels.
    for i in range(7):
        print(i, end = "  ")
    print("", end = "\n\n")








    