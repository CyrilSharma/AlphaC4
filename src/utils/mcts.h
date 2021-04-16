//
// Created by Cyril Sharma on 3/27/21.
//

#ifndef CONNECT_4_MCTS_H
#define CONNECT_4_MCTS_H

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include "C4.h"
#include "NeuralNetwork.h"
#include "thread_pool.h"
#include <mutex>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

class Node {
public:
    // friend class can access private variables
    friend class MCTS;

    Node(int num_actions);

    // custom copy function because atomic type cannot be copied
    Node(const Node &node);

    Node(Node *parent, int action, double prob, int num_actions);

    int best_child(double c_puct, double c_virtual_loss);

    double PUCT(double c_puct, double c_virtual_loss, int total_child_visits) const;

    void expand(const std::vector<double> &child_probs);

    void backup(double leaf_value);

    Node* get_parent() const {
        return parent;
    }

    std::vector<Node*> get_children() const {
        std::vector<Node*> v(children);
        return v;
    }

    int get_visits() const {
        return visits.load();
    }

    int get_action() const {
        return action;
    }

    bool get_expanded() const {
        return expanded;
    }

    double get_prob() const {
        return prob;
    }

    double get_q() const {
        return q;
    }

    int get_virtual_loss() const {
        return virtual_loss.load();
    }

    // Defines the assignment operator for this class
    Node &operator=(const Node &p);


private:
    Node *parent;
    std::vector<Node*> children;
    std::mutex lock;
    bool expanded;
    int action;

    // atomic type is well behaved even with multiple threads accessing it
    std::atomic<int> visits;
    double prob;
    double q;
    std::atomic<int> virtual_loss;
};

class MCTS {
public:
    MCTS(std::string model_path, int num_threads = 1, int batch_size = 10,
         std::vector<int> board_dims = {6, 7}, double c_puct = 4, double c_virtual_loss = 0.01,
         int num_sims = 25, double timeout = 2);

    // method that will be run by multiple threads
    void update(std::shared_ptr<C4> game);

    std::vector<double> final_probs(C4 *game, double temp = 1e-3);

    void shift_root(int last_move);

    Node get_root() const {
        return Node(*root);
    }

    double getCPuct() const;

    void setCPuct(double cPuct);

    double getCVirtualLoss() const;

    void setCVirtualLoss(double cVirtualLoss);

private:
    static void tree_deleter(Node *t);

    // adds custom deleter function to node
    std::unique_ptr<Node, decltype(MCTS::tree_deleter) *> root;

    // multithreading stuffs
    std::unique_ptr<ThreadPool> thread_pool;
    NeuralNetwork neural_network;

    // this value is to be determined empirically
    int num_sims;

    int num_actions;

    double c_puct;

    double c_virtual_loss;

    duration<double, std::milli> timeout;
};

#endif //CONNECT_4_MCTS_H