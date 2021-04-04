//
// Created by Cyrullian Saharmac on 3/28/21.
//

#ifndef CONNECT_4_C4_H
#define CONNECT_4_C4_H

#include <vector>
#include <Eigen/Dense>

class C4 {
public:
    const int rows;
    const int columns;
    const int inarow;

    C4(unsigned int rows, unsigned int columns, unsigned int inarow);

    unsigned int count(int column, int row, int offset_row, int offset_column);

    bool is_win(unsigned int action);

    bool is_draw();

    std::vector<int> is_terminal(unsigned int action);

    void move(unsigned int action);

    void unmove(unsigned int action);

    void flip();

    std::vector<unsigned int> legal();

    std::vector<std::vector<int>> inline get_state() const{
        return state;
    }

    void set_state(std::vector<std::vector<int>> board) {
        state = board;
    }

    void display() const;

    Eigen::MatrixXd get_eigen_state();

    void set_eigen_state(Eigen::MatrixXd board);

    int inline get_player() const{
        return player;
    }

private:
    std::vector<std::vector<int>> state;
    int player = 1;
};

#endif //CONNECT_4_C4_H