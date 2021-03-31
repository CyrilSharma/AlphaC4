//
// Created by Cyrullian Saharmac on 3/28/21.
//

#ifndef CONNECT_4_C4_H
#define CONNECT_4_C4_H

#endif //CONNECT_4_C4_H

#include <vector>

class C4 {
public:
    C4(unsigned int rows, unsigned int columns, unsigned int num);

    unsigned int count(int mark, unsigned int column, unsigned int row, int offset_row, int offset_column);

    bool is_win(unsigned int action);

    bool is_draw();

    std::vector<int> is_terminal(unsigned int action);

    void move(unsigned int action);

    void unmove(unsigned int action);

    void flip();

    std::vector<unsigned int> legal();

    unsigned int inline num_actions() {
        return columns;
    }

    std::vector<std::vector<int>> inline get_state() {
        return state;
    }

    int inline get_player() {
        return player;
    }

private:
    std::vector<std::vector<int>> state;
    unsigned int rows;
    unsigned int columns;
    unsigned int inarow;
    int player = 1;
};
