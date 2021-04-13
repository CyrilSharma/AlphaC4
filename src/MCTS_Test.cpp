//
// Created by Cyrullian Saharmac on 4/4/21.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "utils/MCTS.h"

TEST_CASE("Testing the entire tree search") {
    std::vector<int> board_dims = {6, 7};
    std::string model_path = "../my_model";
    float c_virtual_loss = 0.01;
    unsigned int num_threads = 1;
    float c_puct = 4;
    auto tree = MCTS("../my_model", 1, board_dims, num_threads, c_puct, c_virtual_loss, board_dims[1]);
    auto board = new C4(board_dims[0], board_dims[1], 4);
    std::vector<float> probs = tree.final_probs(board, 1);

    for(int j = 0; j < probs.size(); j++)
        std::cout << probs[j] << ' ';

    CHECK(1 == 1);
}

TEST_CASE("Testing the entire tree search") {
    std::vector<int> board_dims = {6, 7};
    std::string model_path = "../my_model";
    float c_virtual_loss = 0.01;
    unsigned int num_threads = 1;
    float c_puct = 4;
    auto tree = MCTS("../my_model", 2, board_dims, num_threads, c_puct, c_virtual_loss, board_dims[1]);
    auto board = new C4(board_dims[0], board_dims[1], 4);
    std::vector<float> probs = tree.final_probs(board, 1);

    auto root = tree.get_root();

    auto children = root.get_children();

    auto child1 = children[0];

    for(int i = 0; i < children.size(); i++)
        std::cout << children[i]->get_visits() << " ";
    std::cout << "" << std::endl;

    for(int j = 0; j < probs.size(); j++)
        std::cout << probs[j] << ' ';

    CHECK(1 == 1);
}