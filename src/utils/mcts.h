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

#include "C4.h"
#include "NeuralNetwork.h"
#include "thread_pool.h"
#include <mutex>

class Node {
public:
    // friend class can access private variables
    friend class MCTS;

    Node(int num_actions);

    // custom copy function because atomic type cannot be copied
    Node(const Node &node);

    Node(Node *parent, int action, float prob, int num_actions);

    int best_child(float c_puct, float c_virtual_loss);

    float PUCT(float c_puct, float c_virtual_loss, int total_child_visits) const;

    void expand(const std::vector<float> &child_probs);

    void backup(float leaf_value);

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

    float get_prob() const {
        return prob;
    }

    float get_q() const {
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
    float prob;
    float q;
    std::atomic<int> virtual_loss;
};

class MCTS {
public:
    MCTS(const std::string model_path, const int batch_size, const std::vector<int> board_dims,
         const int num_threads, float c_puct, float c_virtual_loss, const int num_actions);

    // method that will be run by multiple threads
    void update(std::shared_ptr<C4> game);

    std::vector<float> final_probs(C4 *game, float temp = 1e-3);

    void shift_root(int last_move);

    Node get_root() const {
        return Node(*root);
    }

    float getCPuct() const;

    void setCPuct(float cPuct);

    float getCVirtualLoss() const;

    void setCVirtualLoss(float cVirtualLoss);

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

    float c_puct;

    float c_virtual_loss;
};

#endif //CONNECT_4_MCTS_H