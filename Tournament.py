from C4 import C4
from MCTS import MCTS
from helpers import render
import numpy as np
from tqdm import tqdm
import logging

def match(p1: MCTS, p2: MCTS, swap):
    game = C4()
    trees = [p1, p2]

    turn = swap
    terminal = False

    while not terminal:
        # get data
        final_probs, state_val = trees[turn % 2].final_probs(game, 1)

        action = np.argmax(final_probs)
        game.move(action)

        for tree in trees:
            tree.shift_root(action)

        reward, terminal = game.is_terminal(action)
        turn += 1
    
    if swap == 0:
        outcome = reward * game.player * -1
    else:
        outcome = reward * game.player
    logging.debug("\n" + np.array_str(game.state))
    logging.debug(f"Outcome: {outcome}")
    logging.debug(f"Swap: {swap}")

    return outcome

def Tournament(p1: MCTS, p2: MCTS, episodes):
    avg_score = 0
    outcomes = []
    for t in tqdm(range(episodes), desc="Tournament"):
        p1.reset()
        p2.reset()
        swap = t % 2
        outcome = match(p1, p2, swap)
        outcomes.append(outcome)
        # shift reward from between [-1, 1] to [0, 1]
        score = (outcome + 1) / 2
        avg_score += (score - avg_score) / (t + 1)
    print(f"Outcomes: {outcomes}")
    return avg_score


