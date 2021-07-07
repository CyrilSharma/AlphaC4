import json
from C4 import C4
from MCTS import MCTS
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
from helpers import render
import logging

def win_loss_draw(score):
    if score>0: 
        return 'win'
    if score<0: 
        return 'loss'
    return 'draw'

def getData():
    data = []
    with open('c4-eval.txt') as f:
        for line in f:
            full_line = line
            dict = json.loads(full_line)
            data.append(dict)
    return data

def Evaluate(samples, agent: MCTS, sample_spacing=8, random=False):
    data = getData()

    if random:
        rng = default_rng()
        # grabs 100 unique samples
        subset = rng.choice(data, size=samples,replace=False)
    else:
        subset = data[0:samples*sample_spacing:sample_spacing]
    
    moves = 0
    state_error = 0
    goodMoves = 0
    perfectMoves = 0
    agent_moves = np.zeros(7)

    for t in tqdm(range(samples), desc="Evaluation"):
        agent.reset()

        datum = subset[t]
        game = C4()

        player = ((len([x for x in datum["board"] if x!=0]) + 1) % 2) * 2 - 1
        board = np.array(datum["board"])
        board = np.reshape(np.where(board == 2, -1, board), (6, 7))

        game.player = player
        game.state = board

        logging.debug("\n" + np.array_str(game.state))
        final_probs, state_val = agent.final_probs(game, 0)
        #agent_move = np.random.choice([0,1,2,3,4,5,6])
        agent_move = np.argmax(final_probs)
        agent_moves[agent_move] += 1

        move_scores = np.array(datum["move score"])
        best_score = np.max(move_scores)

        logging.debug("Move scores: " + np.array_str(move_scores))

        best_moves = [i for i in range(7) if move_scores[i]==best_score]

        if agent_move in best_moves:
            perfectMoves += 1
            
        if win_loss_draw(move_scores[agent_move]) == win_loss_draw(best_score):
            goodMoves += 1
    
        if datum["score"] >= 1:
            score = 1
        elif datum["score"] <= -1:
            score = -1
        else:
            score = 0

        mse = (state_val - score) ** 2
        state_error += (mse - state_error) / (moves + 1)
        
        moves += 1
    
    print(f"Agent Moves: {agent_moves}")
    return (goodMoves / moves, perfectMoves / moves, state_error)


