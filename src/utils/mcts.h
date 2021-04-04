//
// Created by Cyril Sharma on 3/27/21.
//

#ifndef CONNECT_4_MCTS_H
#define CONNECT_4_MCTS_H

#endif //CONNECT_4_MCTS_H

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "C4.h"
#include "thread_pool.h"
#include <mutex>

class Node {
public:
    // friend class can access private variables
    friend class MCTS;

    Node();

    // custom copy function because atomic type cannot be copied
    Node(const Node &node);

    Node(Node *parent, unsigned int action, double prob, unsigned int num_actions);

    unsigned int best_child(double c_puct, double c_virtual_loss);

    double PUCT(double c_puct, double c_virtual_loss, unsigned int total_child_visits) const;

    void expand(const std::vector<double> &child_probs);

    void backup(double leaf_value);
    // Defines the assignment operator for this class
    Node &operator=(const Node &p);


private:
    Node *parent;
    std::vector<Node *> children;
    std::mutex lock;
    bool expanded;
    int action;

    // atomic type is well behaved even with multiple threads accessing it
    std::atomic<unsigned int> visits;
    double prob;
    double q;
    std::atomic<int> virtual_loss;
};

class MCTS {
public:
    MCTS(NeuralNetwork *neural_network, unsigned int num_threads, double c_puct, double c_virtual_loss, unsigned int num_actions);

    // method that will be run by multiple threads
    void update(std::shared_ptr<C4> game);

    // Gomoku to be replaced with our connect 4 board state
    std::vector<double> final_probs(C4 *game, double temp = 1e-3);

    void shift_root(int last_move);

private:
    static void tree_deleter(Node *t);

    // unique_ptr is a pointer then can only be moved, not copied
    // the second argument is a custom delete function that will be called when the pointer is reassigned

    // gets type of tree_deleter
    std::unique_ptr<Node, decltype(MCTS::tree_deleter) *> root;

    // multithreading stuffs
    //TODO: Make your own thread pool where tasks can be committed and retrieved at a later time
    std::unique_ptr<ThreadPool> thread_pool;
    NeuralNetwork *neural_network;

    // this value is to be determined empirically
    unsigned int num_sims;

    unsigned int num_actions;

    double c_puct;
    double c_virtual_loss;
};