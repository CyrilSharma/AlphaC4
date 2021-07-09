from Trainer import Trainer
from ActorCritic import ActorCritic
from tensorflow import keras
from MCTS import MCTS
from Evaluate import Evaluate
import json

def main():
    with open('parameters.json') as file:
        params = json.load(file)
    with open('config.json') as file:
        config = json.load(file)
    
    model = ActorCritic()
    model_name = "second_model"

    trainer = Trainer(model, params, config)
    trainer.training_loop(model_name)


    # tree = MCTS(model, 0.01, 3)
    # good_moves, perfect_moves, mse = Evaluate(1000, tree, sample_spacing=1)
    # print(f"good_moves: {good_moves * 100}% \nperfect_moves: {perfect_moves * 100}% \nstate mse: {mse}")
main()