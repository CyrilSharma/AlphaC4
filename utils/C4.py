import numpy as np


class C4():
    def __init__(self, r=6, c=7, num=4):
        self.state = np.zeros((r, c), dtype=np.float32)
        self.rows = r
        self.columns = c
        self.inarow = num
        self.player = 1.

    def count(self, column: int, row: int, offset_row: int, offset_column: int):
        # Counts number of pieces in a certain direction, excluding the starting piece

        mark = self.state[row][column]
        for i in range(self.inarow):
            r = row + offset_row * i
            c = column + offset_column * i

            # if current mark doesn't fit into the sequence, stop counting
            if (r < 0 or  c < 0 or c >= self.columns or r >= self.rows or self.state[r][c] != mark):
                return i - 1

        return self.inarow

    def is_win(self, action: int):
        # determine row winning piece was placed in
        row = 0
        for i in range(self.rows):
            if (self.state[i][action] == 0):
                row += 1

        # check for connect 4s in all directions
        return (
                ((self.count(action, row, 1, 0)) >= self.inarow - 1) or
                ((self.count(action, row, 0, 1) + self.count(action, row, 0, -1)) >= (self.inarow - 1)) or
                ((self.count(action, row, -1, -1) + self.count(action, row, 1, 1)) >= (self.inarow - 1)) or
                ((self.count(action, row, -1, 1) + self.count(action, row, 1, -1)) >= (self.inarow - 1))
        )

    def is_draw(self):
        non_zeros = np.count_nonzero(self.state[0])

        if (non_zeros == self.columns):
            return True
        else:
            return False


    def is_terminal(self, action: int):
        if (self.is_win(action)):
            return (1.,1.)
        elif (self.is_draw()):
            return (0.,1.)
        else:
            return (0.,0.)

    def move(self, action: int):
        # TODO: Use a numpy function instead for optimization puurposes
        row_num = self.rows - 1
        for row in range(self.rows - 1, 0, -1):
            if (self.state[row][action] == 0):
                break

            row_num -= 1

        # Make board flipped for ease of use
        self.state[row_num][action] = self.player
        self.player *= -1.

    def unmove(self, action: int):
        row_num = rows

        for row in range(self.rows - 1, 0, -1):
            if (self.state[row][action] == 0):
                break

            row_num -= 1

        self.state[row_num][action] = 0.
        self.player *= -1.

    def legal(self):
        legal_actions = np.zeros(self.columns)
        i = 0

        # TODO: Use a numpy functioon instead
        for mark in self.state[0]:
            if (mark == 0):
                legal_actions[i] = 1
            i += 1

        return legal_actions