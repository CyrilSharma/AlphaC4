//
// Created by Cyrullian Saharmac on 4/5/21.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "utils/C4.h"
#include "utils/NeuralNetwork.h"

TEST_CASE("Testing an untrained Network") {

    SUBCASE("Testing with a batch size of 1") {
        auto nn = NeuralNetwork("../my_model", 1, {6, 7});
        auto c4 = new C4(6,7,4);

        auto future = nn.commit(c4);
        future.wait();

        auto result = future.get();

        auto probs = result[0];
        auto value = result[1][0];

        for (auto vec: result) {
            for (auto num: vec) {
                std::cout << num << " " << std::flush;
            }
            std::cout << "" << std::endl;
        }

        CHECK(1 == 1);
    }

    SUBCASE("Testing with a larger batch size") {
        auto nn = NeuralNetwork("../my_model", 2, {6, 7});
        auto c4 = new C4(6,7,4);

        std::vector<std::future<std::vector<std::vector<float>>>> futures;

        for (int i = 0; i < 2; i++) {
            futures.push_back(nn.commit(c4));
        }

        for (int i = 0; i < 2; i++) {
            futures[i].wait();
        }

        auto result1 = futures[0].get();
        auto result2 = futures[1].get();

        for (auto vec: result1) {
            for (auto num: vec) {
                std::cout << num << " " << std::flush;
            }
            std::cout << "" << std::endl;
        }

        for (auto vec: result2) {
            for (auto num: vec) {
                std::cout << num << " " << std::flush;
            }
            std::cout << "" << std::endl;
        }

        CHECK(1 == 1);
    }
}