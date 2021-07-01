from Trainer import Trainer
from ActorCritic import ActorCritic
import json

def main():
    with open('parameters.json') as file:
        params = json.load(file)
    with open('config.json') as file:
        config = json.load(file)
    
    model = ActorCritic()

    trainer = Trainer(model, params, config)
    trainer.training_loop(100)

main()