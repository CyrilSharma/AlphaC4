//
// Created by Cyrullian Saharmac on 4/1/21.
//

#include "mcts.h"
#include "C4.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<int> BOARD_DIMS = {6, 7};

PYBIND11_MODULE(bindings, m) {
    py::class_<C4>(m, "C4")
    .def(py::init<int, int, int>(), py::arg("rows") = 6, py::arg("columns") = 7,
         py::arg("inarow") = 4)
    .def_readonly("rows", &C4::rows)
    .def_readonly("columns", &C4::columns)
    .def_readonly("inarow", &C4::inarow)
    .def("count", &C4::count)
    .def("is_win", &C4::is_win, py::arg("action"))
    .def("is_draw", &C4::is_draw)
    .def("is_terminal", &C4::is_terminal, py::arg("action"))
    .def("move", &C4::move, py::arg("action"))
    .def("unmove", &C4::unmove, py::arg("action"))
    .def("flip", &C4::flip)
    .def("legal", &C4::legal)
    .def_property("state", &C4::get_eigen_state, &C4::set_eigen_state)
    ;

    py::class_<Node>(m, "Node")
    .def(py::init<int>(), py::arg("num_action") = 7)
    .def(py::init<const Node>())
    .def(py::init<Node*, int, double, int>(), py::arg("Node"), py::arg("action"),
         py::arg("prob"), py::arg("num_actions"))
    .def("best_child", &Node::best_child)
    .def("PUCT", &Node::PUCT)
    .def("expand", &Node::expand)
    .def("backup", &Node::backup)
    .def_property_readonly("visits", &Node::get_visits)
    .def_property_readonly("prob", &Node::get_prob)
    .def_property_readonly("q", &Node::get_q)
    .def_property_readonly("virtual_loss", &Node::get_virtual_loss)
    .def_property_readonly("action", &Node::get_action)
    .def_property_readonly("children", &Node::get_children)
    .def_property_readonly("parent", &Node::get_parent)
    .def("__repr__", [](const Node &n){
        return "<Node with action " + std::to_string(n.get_action()) + ">";
    });

    py::class_<MCTS>(m, "MCTS")
    .def(py::init<std::string, int, int, std::vector<int>, double, double, int, double>(),
            py::arg("model_path") = "my_model", py::arg("num_threads") = 1, py::arg("batch_size") = 10,
            py::arg("board_dims") = BOARD_DIMS, py::arg("c_puct") = 4, py::arg("c_virtual_loss") = 0.01,
            py::arg("num_sims") = 25, py::arg("timeout") = 2)
    .def("shift_root", &MCTS::shift_root)
    .def_property_readonly("root", &MCTS::get_root)
    .def("final_probs", [](MCTS* mcts, C4 *c4, double temp) -> std::vector<double> {
        py::gil_scoped_release release;
        auto probs = mcts->final_probs(c4, temp);
        py::gil_scoped_acquire acquire;
        return probs;
    }, py::arg("board"), py::arg("temperature"))
    .def_property("c_puct", &MCTS::getCPuct, &MCTS::setCPuct)
    .def_property("c_virtual_loss", &MCTS::getCVirtualLoss, &MCTS::setCVirtualLoss)
    ;
}
