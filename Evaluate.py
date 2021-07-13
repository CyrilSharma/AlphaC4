import json
from C4 import C4
from MCTS import MCTS
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
from helpers import render
from tensorflow import keras
import logging

def win_loss_draw(score):
    if score>0: 
        return 'win'
    if score<0: 
        return 'loss'
    return 'draw'

def getData(filename):
    data = []
    with open(filename) as f:
        for line in f:
            full_line = line
            dict = json.loads(full_line)
            data.append(dict)
    return data

def Evaluate(samples, agent: MCTS, sample_spacing=8):
    data = getData('c4-eval.txt')

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
        final_probs, state_val = agent.final_probs(game, 1)
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

def Replay(actions):
    game = C4()
    for action in actions:
        render(game.state)
        game.move(action)
    render(game.state)

if __name__ == "__main__":
    with open('parameters.json') as file:
        params = json.load(file)
    params["timeout"] = 2.0
    model = keras.models.load_model("Models/first_model", compile=False)
    agent = MCTS(model, params)
    good, perfect, state_error = Evaluate(800, agent, sample_spacing=1)
    print(f"Good Moves: {good}")
    print(f"Perfect Moves: {perfect}")
    print(f"State Error: {state_error}")