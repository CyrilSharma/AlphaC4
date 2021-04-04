//
// Created by Cyrullian Saharmac on 3/28/21.
//

#ifndef CONNECT_4_NEURALNETWORK_H
#define CONNECT_4_NEURALNETWORK_H


#include <string>
#include <vector>
#include <future>
#include <queue>
#include "C4.h"
#include <cppflow/cppflow.h>
#include <tensorflow/c/c_api.h>

class NeuralNetwork {
public:
    using return_type = std::vector<std::vector<float>>;

    NeuralNetwork(std::string model_path, unsigned int batch_size, std::vector<unsigned int> board_dims);
    ~NeuralNetwork();

    //std::future is used for asynchronous tasks
    std::future<return_type> commit(C4* c4);  // commit task to queue

    void set_batch_size(unsigned int batch_size) {    // set batch_size
        this->batch_size = batch_size;
    };

private:
    // not sure about this, there should be two returns no?
    using task_type = std::pair<std::vector<float>, std::promise<return_type>>;

    void predict();

    cppflow::model model;

    // thread for feeding inputs to NN
    std::unique_ptr<std::thread> loop;
    bool running;

    int rows;
    int columns;

    std::queue<task_type> tasks;  // tasks queue
    std::mutex lock;              // lock for tasks queue
    std::condition_variable cv;   // condition variable for tasks queue

    unsigned int batch_size;                             // batch size
};


#endif //CONNECT_4_NEURALNETWORK_H
