//
// Created by Cyrullian Saharmac on 3/28/21.
//

#include "NeuralNetwork.h"
#include <iostream>
#include <numeric>

using namespace std::chrono_literals;

static void DeallocateTensor(void* data, std::size_t, void*) {
    std::free(data);
    std::cout << "Deallocate tensor" << std::endl;
}

NeuralNetwork::NeuralNetwork(std::string model_path, unsigned int batch_size, std::vector<unsigned int> board_dims):
  model(cppflow::model(model_path)),
  rows(board_dims[0]),
  columns(board_dims[1]),
  batch_size(batch_size),
  running(true),
  loop(nullptr) {


    // run infer thread
    this->loop = std::make_unique<std::thread>([this] {
        while (this->running) {
            this->predict();
        }
    });
}

NeuralNetwork::~NeuralNetwork() {
    this->running = false;
    this->loop->join();
}

std::future<NeuralNetwork::return_type> NeuralNetwork::commit(C4 *c4) {
    // convert data format
    auto board = c4->get_state();

    std::vector<float> boardvec;

    for (unsigned int i = 0; i < board.size(); i++) {
        boardvec.insert(boardvec.end(), board[i].begin(), board[i].end());
    }

    std::vector<float> floatvec;

    for (unsigned int i = 0; i < boardvec.size(); i++) {
        floatvec.push_back(static_cast<float>(boardvec[i]));
    }

    // Replace swapping in move method with a conditional swapping here for efficiency

    // int cur_player = c4->get_player();

    // emplace task
    std::promise<return_type> promise;
    auto ret = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(this->lock);
        tasks.emplace(std::make_pair(floatvec, std::move(promise)));
    }

    this->cv.notify_all();

    return ret;
}

void NeuralNetwork::predict() {
    std::vector<std::promise<return_type>> promises;
    std::vector<std::vector<float>> states;

    bool timeout = false;

    // fill up queue, but proceed if it takes too long
    while (states.size() < this->batch_size && !timeout) {
        {
            std::unique_lock<std::mutex> lock(this->lock);

            // waits for when the conditional variable indicates its time to check if the thread can proceed
            // waits a maximum of 1 ms
            // proceeds if tasks is greater then zero
            if (this->cv.wait_for(lock, 1ms,
                                  [this] { return this->tasks.size() > 0; })) {
                // pop task
                auto task = std::move(this->tasks.front());
                states.emplace_back(std::move(task.first));
                promises.emplace_back(std::move(task.second));

                this->tasks.pop();

            // if no tasks have been queueueuued in 1 ms, send the batch anyways
            } else {
                // timeout
                timeout = true;
            }
        }
    }

    // if inputs empty return
    if (states.size() == 0) {
        return;
    }

    const std::vector<std::int64_t> dims = {static_cast<long long>(states.size()), rows, columns, 1};

    // Data size of type size_t
    const auto data_size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<std::int64_t>{});

    auto data = static_cast<float*>(std::malloc(data_size));

    std::vector<float> statevec;

    for (unsigned int i = 0; i < states.size(); i++) {
        statevec.insert(statevec.end(), states[i].begin(), states[i].end());
    }

    std::copy(statevec.begin(), statevec.end(), data);


    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims.data(), static_cast<int>(dims.size()),
                                            data, data_size, DeallocateTensor, nullptr);

    // infer
    auto output_tensor = model({{"serving_default_input_1:0", input_tensor}}, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});

    std::vector<float> all_probs = output_tensor[0].get_data<float>();
    std::vector<float> all_values = output_tensor[1].get_data<float>();

    std::vector<float> probs = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    std::vector<float> values = {0.f};
    // set promise value
    for (unsigned int i = 0; i < promises.size(); i++) {

        for (unsigned int j = 0; j < 7; j++) {
            probs[i] = all_probs[i * 7 + j];
        }
        for (unsigned int k = 0; k < 2; k++) {
            values[i] = all_values[i * 2 + k];
        }

        // TODO: get data from output and seperate it into the probabilities and state values

        return_type temp{std::move(probs), std::move(values)};

        promises[i].set_value(std::move(temp));
    }
}