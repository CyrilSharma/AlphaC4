//
// Created by Cyrullian Saharmac on 4/4/21.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <mcts.h>

TEST_CASE("Single-threaded test") {
    std::vector<int> board_dims = {6, 7};
    auto tree = MCTS("../my_model");
    auto board = new C4(board_dims[0], board_dims[1], 4);
    std::vector<double> probs = tree.final_probs(board, 1);

    auto root = tree.get_root();

    auto children = root.get_children();

    for(int i = 0; i < children.size(); i++)
        std::cout << children[i]->get_visits() << " ";
    std::cout << "" << std::endl;

    for(int j = 0; j < probs.size(); j++)
        std::cout << probs[j] << ' ';

    CHECK(1 == 1);
}

TEST_CASE("Multi-threaded test") {
    std::vector<int> board_dims = {6, 7};
    int num_threads = 4;
    auto tree = MCTS("../my_model", num_threads);
    auto board = new C4(board_dims[0], board_dims[1], 4);
    std::vector<double> probs = tree.final_probs(board, 1);

    auto root = tree.get_root();

    auto children = root.get_children();

    for(int i = 0; i < children.size(); i++)
        std::cout << children[i]->get_visits() << " ";
    std::cout << "" << std::endl;

    for(int j = 0; j < probs.size(); j++)
        std::cout << probs[j] << ' ';

    CHECK(1 == 1);
}