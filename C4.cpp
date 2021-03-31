//
// Created by Cyrullian Saharmac on 3/28/21.
//

#include <algorithm>
#include "C4.h"

C4::C4(unsigned int r, unsigned int c, unsigned int num):
    state(std::vector<std::vector<int>>(r, std::vector<int>(c, 0))),
    rows(r),
    columns(c),
    inarow(num){}

unsigned int C4::count(int mark, unsigned int column, unsigned int row, int offset_row, int offset_column) {
    // Counts number of pieces in a certain direction, excluding the starting piece
    for (int i = 0; i < this->inarow; i++) {
        unsigned int r = row + offset_row * i;
        unsigned int c = column + offset_column * i;
        // if current mark doesn't fit into the sequence, stop counting
        if (r < 0 || c < 0 || c >= this->columns || r >= this->rows || state[r][c] != mark) {
            return i - 1;
        }
    }
    return this->inarow;
}

bool C4::is_win(unsigned int action) {
    // determine row winning piece was placed in
    unsigned int row = 0;
    for (int i = 0; i < this->rows; i++) {
        if (this->state[i][action] == 0) {
            row += 1;
        }
    }
    int mark = state[row][action];

    // check for connect 4s in all directions
    return (
            ((count(action, row, mark, 1, 0)) >= this->inarow - 1) ||
            ((count(action, row, mark, 0, 1) + count(action, row, mark, 0, -1)) >= (this->inarow - 1)) ||
            ((count(action, row, mark, -1, -1) + count(action, row, mark, 1, 1)) >= (this->inarow - 1)) ||
            ((count(action, row, mark, -1, 1) + count(action, row, mark, 1, -1)) >= (this->inarow - 1))
    );
}

bool C4::is_draw() {
    unsigned int zeros = std::count(this->state[0].begin(), this->state[0].end(), 0);
    if (zeros == 0) {
        return true;
    }
    else {
        return false;
    }
}

std::vector<int> C4::is_terminal(unsigned int action) {
    if (is_win(action)) {
        return {1, 1};
    }
    else if (is_draw()) {
        return {1, 0};
    }
    else {
        return {0, 0};
    }
}

void C4::move(unsigned int action) {
    unsigned int last_row = std::count(this->state[0].begin(), this->state[0].end(), 0);
    this->state[last_row + 1][action] = 1;
    flip();
    player *= -1;
}

void C4::unmove(unsigned int action) {
    unsigned int last_row = std::count(this->state[0].begin(), this->state[0].end(), 0);
    this->state[last_row][action] = 0;
    flip();
    player *= -1;
}

void C4::flip() {
    for (auto v: this->state) {
        transform(v.begin(), v.end(), v.begin(), [](int &c){ return -c; });
    }
}

std::vector<unsigned int> C4::legal() {
    std::vector<unsigned int> legal_actions = std::vector<unsigned int>(this->columns, 0);
    unsigned int i = 0;
    for (auto mark: this->state[0]) {
        if (mark == 0) {
            legal_actions[i] = 1;
        }
        i += 1;
    }
}


