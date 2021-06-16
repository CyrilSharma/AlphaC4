from utils.bindings import MCTS, C4
import numpy as np

class C4Agent():
    def __init__(self, model_path, game: C4, params):
        self.game = game
        self.tree = MCTS(model_path, num_threads=params.num_threads, batch_size=params.batch_size, board_dims=params.board_dims, c_puct=params.c_puct, c_virtual_loss=params.c_vl, timeout=params.timeout)
    
    def move(self):
        action, probs, terminal, win = self.tree.MCTS()

        if not terminal:
            self.game.move(action)
            self.tree.shift_root(action)