//
// Created by Cyrullian Saharmac on 4/3/21.
//

#define CATCH_CONFIG_MAIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "utils/C4.h"

bool compare_boards(std::vector<std::vector<int>> vec1, std::vector<std::vector<int>> vec2, bool display) {
    bool equal = true;

    for (int i = 0; i < vec2.size(); i++) {

        if (display) {
            for(int j = 0; j < vec1[i].size(); j++)
                std::cout << vec1[i].at(j) << ' ';
            std::cout << "" << std::endl;

            for(int j = 0; j < vec2[i].size(); j++)
                std::cout << vec2[i].at(j) << ' ';

            std::cout << "\n" << std::endl;
        }

        equal = (vec1[i] == vec2[i] & equal);
    }
    return equal;
}

TEST_CASE("Testing the display function") {
    auto c4 = new C4(6,7,4);
    c4->display();
    CHECK(1 == 1);
}

TEST_CASE("Testing the move function") {

        auto c4 = new C4(6, 7, 4);

        c4->move(4);

        c4->display();
        auto state = c4->get_state();
        std::vector<std::vector<int>> answer = {{0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 1, 0, 0}};

        bool equal = compare_boards(state, answer, false);

        CHECK(equal);
        c4->move(4);
        c4->display();
        state = c4->get_state();
        answer = {{0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, -1, 0, 0},
                  {0, 0, 0, 0, 1, 0, 0}};

        equal = compare_boards(state, answer, false);

        CHECK(equal);
}

TEST_CASE("Testing the unmove function") {

    auto c4 = new C4(6, 7, 4);

    c4->move(4);
    c4->move(3);
    c4->move(4);
    c4->move(2);

    c4->display();
    c4->unmove(4);
    c4->display();
    auto state = c4->get_state();
    std::vector<std::vector<int>> answer = {
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, -1, -1, 1, 0, 0}};

    bool equal = compare_boards(state, answer, false);

    CHECK(equal);
}

TEST_CASE("Testing the win function") {

    SUBCASE("Test Vertical") {
        auto c4 = new C4(6, 7, 4);

        c4->move(4);
        c4->move(2);
        c4->move(4);
        c4->move(2);
        c4->move(4);
        c4->move(2);
        c4->move(4);

        c4->display();
        bool win = c4->is_win(4);

        CHECK(win == true);
    }

    SUBCASE("Test Horizontal") {
        auto c4 = new C4(6, 7, 4);

        c4->move(0);
        c4->move(0);
        c4->move(1);
        c4->move(0);
        c4->move(2);
        c4->move(0);
        c4->move(3);

        c4->display();
        bool win = c4->is_win(3);

        CHECK(win == true);
    }

    SUBCASE("Test Diagonal") {
        auto c4 = new C4(6, 7, 4);

        c4->set_state({
                        {0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 1, -1, 0},
                        {0, 0, 0, 1, -1, -1, 0},
                        {1, 1, 1, -1, -1, -1, 0}
        });

        c4->display();

        bool win = c4->is_win(5);

        CHECK(win == true);
    }
}

TEST_CASE("Testing the draw function") {

    auto c4 = new C4(6, 7, 4);

    c4->set_state({
        {1, 1, -1, 1, -1, 1, 1},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0}
    });

    c4->display();

    bool draw = c4->is_draw();

    CHECK(draw == true);
}

TEST_CASE("Testing the legal function") {

    auto c4 = new C4(6, 7, 4);

    c4->set_state({
                          {1, 1, 0, 1, -1, 1, 0},
                          {0, 0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0, 0}
                  });

    c4->display();

    std::vector<int> legal_actions = c4->legal();

    std::vector<int> answer = {0, 0, 1, 0, 0, 0, 1};

    CHECK(legal_actions == answer);
}