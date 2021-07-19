# AlphaC4
An implementation of the Alphazero algorithmn for Connect 4.

# Usage
To train the model execute python3 train.py <br />

The training procedure cycles between game generation through self-play, training the neural network, and finally evaluating the new network through 
two means: battling the new network against the previous iteration, and evaluating the model on a Kaggle dataset. <br /><br />
A detailed summary of each game the network plays can be found in the trainer.log file within the logs folder. Currently, the trainer.log file tracks child node visits, the final and inital action probabilities, the q values, and what actions are terminal, if any.

