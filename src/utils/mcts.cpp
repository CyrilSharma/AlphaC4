//
// Created by Cyril Sharma on 3/27/21.
//

#include "mcts.h"
#include <cmath>
#include <cfloat>

// basically the numpy of c++
// allows for doing linear algebra
#include <numeric>

// lets you print stuff
#include <iostream>

#include <algorithm>
#include <future>
#include <functional>
#include <chrono>

// Node
Node::Node()
        : parent(nullptr),
          virtual_loss(0),
          visits(0),
          prob(0),
          action(-1),
          q(0) {}

Node::Node(Node *parent, unsigned int action, double prob, unsigned int num_actions)
        : parent(parent),
          children(num_actions, nullptr),
          virtual_loss(0),
          visits(0),
          prob(prob),
          q(0) {}

Node::Node(const Node &node) {
    // Atomic types cannot be copied, so a custom copy function is needed
    this->parent = node.parent;
    this->children = node.children;

    this->visits.store(node.visits.load());
    this->prob = node.prob;
    this->q = node.q;

    this->virtual_loss.store(node.virtual_loss.load());
}

// This is what's called whenever you use the assignment operator with this class
Node &Node::operator=(const Node &node) {
    if (this == &node) {
        return *this;
    }

    // struct
    this->parent = node.parent;
    this->children = node.children;

    this->visits.store(node.visits.load());
    this->action = node.action;
    this->prob = node.prob;
    this->q = node.q;
    this->virtual_loss.store(node.virtual_loss.load());

    return *this;
}

unsigned int Node::best_child(double c_puct, double c_virtual_loss) {
    double best_value = -DBL_MAX;
    unsigned int best_move = 0;
    Node *best_node;

    for (unsigned int i = 0; i < children.size(); i++) {
        // empty node
        if (children[i] == nullptr) {
            continue;
        }

        unsigned int total_child_visits = this->visits.load() + 1;
        double cur_value =
                children[i]->PUCT(c_puct, c_virtual_loss, total_child_visits);
        if (cur_value > best_value) {
            best_value = cur_value;
            best_move = i;
            best_node = children[i];
        }
    }

    // add vitural loss
    best_node->virtual_loss++;

    return best_move;
}

void Node::expand(const std::vector<double> &child_probs) {
    {
        // get lock
        std::lock_guard<std::mutex> lock(this->lock);

        if (!this->expanded) {

            for (unsigned int i = 0; i < children.size(); i++) {
                // illegal action
                if (abs(child_probs[i] - 0) < FLT_EPSILON) {
                    continue;
                }
                this->children[i] = new Node(this, i, child_probs[i], children.size());
            }

            this->expanded = true;
        }
    }
}

void Node::backup(double value) {
    // If it is not root, this node's parent should be updated first
    if (this->parent != nullptr) {
        this->parent->backup(-value);
    }

    // remove vitural loss
    this->virtual_loss--;

    // update visits
    unsigned int visits = this->visits.load();
    this->visits++;

    // update q
    {
        std::lock_guard<std::mutex> lock(this->lock);
        this->q += (value - this->q) / (visits + 1);
    }
}

double Node::PUCT(double c_puct, double c_virtual_loss,
                           unsigned int total_child_visits) const {
    // u
    auto visits = this->visits.load();
    double u = (c_puct * this->prob * sqrt(total_child_visits) / (1 + visits));

    // virtual loss
    double virtual_loss = c_virtual_loss * this->virtual_loss.load();
    // int visits_with_loss = visits - virtual_loss;

    if (visits <= 0) {
        return u;
    } else {
        // average reward with virtual loss
        return u + (this->q * visits - virtual_loss) / visits;
    }
}

// MCTS_Package
MCTS::MCTS(NeuralNetwork *neural_network, unsigned int thread_num, double c_puct,
           unsigned int num_mcts_sims, double c_virtual_loss,
           unsigned int num_actions)
        : neural_network(neural_network),
          thread_pool(new ThreadPool(thread_num)),
          c_puct(c_puct),
          c_virtual_loss(c_virtual_loss),
          num_actions(num_actions),
          root(new Node(nullptr, 1., 0, num_actions), MCTS::tree_deleter){}

void MCTS::shift_root(int last_action) {
    auto old_root = this->root.get();

    // reuse the child tree
    if (last_action >= 0 && old_root->children[last_action] != nullptr) {
        // unlink
        Node *new_node = old_root->children[last_action];
        // so that the new tree isn't deleted
        old_root->children[last_action] = nullptr;
        new_node->parent = nullptr;

        // assigns root to new node and deletes the rest of the tree
        this->root.reset(new_node);
    }
    else {
        // if game is over create empty root node
        this->root.reset(new Node(nullptr, -1, 0, num_actions));
    }
}

void MCTS::tree_deleter(Node *t) {
    if (t == nullptr) {
        return;
    }

    // remove children
    for (unsigned int i = 0; i < t->children.size(); i++) {
        if (t->children[i]) {
            tree_deleter(t->children[i]);
        }
    }

    // remove self
    delete t;
}

std::vector<double> MCTS::final_probs(C4 *c4, double temp) {
    // submit update tasks to thread_pool
    // std::future is the standard means of accessing the results of asynchronous operations
    std::vector<std::future<void>> futures;

    // replace with a time dependent loop
    for (unsigned int i = 0; i < this->num_sims; i++) {
        // copy board
        auto game = std::make_shared<C4>(*c4);
        auto future = this->thread_pool->commit(std::bind(&MCTS::update, this, game));

        // you can't copy future so instead you can copy a reference to future with move
        futures.emplace_back(std::move(future));
    }

    // asynchronous tasks have already started
    // results are now waited for
    for (unsigned int i = 0; i < futures.size(); i++) {
        futures[i].wait();
    }

    // calculate probs
    std::vector<double> action_probs(num_actions, 0);
    const auto &children = this->root->children;

    // according to the alphazero paper, the temperature was set to an infinitesimal to approximate a greedy policy
    // this chooses whatever action had the most visits
    if (temp - 1e-3 < FLT_EPSILON) {
        unsigned int max_count = 0;
        unsigned int best_action = 0;

        for (unsigned int i = 0; i < children.size(); i++) {
            // if child exists and number of child visits is greater then the max
            if (children[i] && children[i]->visits.load() > max_count) {
                max_count = children[i]->visits.load();
                best_action = i;
            }
        }

        action_probs[best_action] = 1.;
        return action_probs;

    //
    } else {
        // explore
        double sum = 0;
        for (unsigned int i = 0; i < children.size(); i++) {
            if (children[i] && children[i]->visits.load() > 0) {
                action_probs[i] = pow(children[i]->visits.load(), 1 / temp);
                sum += action_probs[i];
            }
        }

        // renormalization
        // iterates over each number in action_probs
        std::for_each(action_probs.begin(), action_probs.end(),
                      [sum](double &x) { x /= sum; });

        return action_probs;
    }
}

void MCTS::update(std::shared_ptr<C4> game) {

    auto node = this->root.get();

    while (true) {
        if (!node->expanded) {
            break;
        }

        // select
        auto action = node->best_child(this->c_puct, this->c_virtual_loss);
        game->move(action,1);
        node = node->children[action];
    }

    // get game status
    auto status = game->is_terminal();
    double value = 0;

    // if not terminal
    if (status[0] == 0) {
        // predict action_probs and value by neural network
        std::vector<double> probs(num_actions, 0);

        //TODO: replace this code with a queue of some sort that queries the network in batches

        // Feeds the raw pointer into class neural network
        auto future = this->neural_network->commit(game.get());
        auto result = future.get();

        probs = std::move(result[0]);
        value = result[1][0];

        // mask invalid actions
        auto legal_moves = game->legal();

        // sum legal action values
        double sum = 0;
        for (unsigned int i = 0; i < num_actions; i++) {
            if (legal_moves[i] == 1) {
                sum += probs[i];
            } else {
                probs[i] = 0;
            }
        }

        // renormalization
        if (sum > FLT_EPSILON) {
            std::for_each(probs.begin(), probs.end(),
                          [sum](double &x) { x /= sum; });
        } else {
            // all masked

            // NB! All valid moves may be masked if either your NNet architecture is
            // insufficient or you've get overfitting or something else. If you have
            // got dozens or hundreds of these messages you should pay attention to
            // your NNet and/or training process.
            std::cout << "All valid moves were masked, do workaround." << std::endl;

            // number of legal moves
            sum = std::accumulate(legal_moves.begin(), legal_moves.end(), 0);

            // initialize probs of legal moves to random distribution
            for (unsigned int i = 0; i < probs.size(); i++) {
                probs[i] = legal_moves[i] / sum;
            }
        }

        // expand
        node->expand(probs);

    } else {
        //1 for win, 0 for draw, there is no -1 for loss as the game is never a loss from the final players perspective
        value = status[1];
    }

    // value(parent -> node) = -value
    node->backup(value);
    return;
}