//
// Created by Cyrullian Saharmac on 4/1/21.
//

#include "mcts.h"
#include "C4.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(bindings, m) {
    py::class_<C4>(m, "C4")
    .def(py::init<unsigned int, unsigned int, unsigned int>())
    .def_readonly("rows", &C4::rows)
    .def_readonly("columns", &C4::columns)
    .def_readonly("inarow", &C4::inarow)
    .def("count", &C4::count)
    .def("is_win", &C4::is_win)
    .def("is_draw", &C4::is_draw)
    .def("is_terminal", &C4::is_terminal)
    .def("move", &C4::move)
    .def("unmove", &C4::unmove)
    .def("flip", &C4::flip)
    .def("legal", &C4::legal)
    .def_property("state", &C4::get_eigen_state, &C4::set_eigen_state)
    ;
}

//py::class_<C4>(m, "C4")
//.def(py::init<unsigned int, unsigned int, unsigned int>(), py::arg("rows"), py::arg("columns"), py::arg("inarow"))
//;

//.def_readonly("rows", &C4::rows)
//.def_readonly("columns", &C4::columns)
//.def_readonly("inarow", &C4::inarow)
//.def("count", &C4::count)
//.def("is_win", &C4::is_win)
//.def("is_draw", &C4::is_draw)
//.def("is_terminal", &C4::is_terminal)
//.def("move", &C4::move)
//.def("unmove", &C4::unmove)
//.def("flip", &C4::flip)
//.def("legal", &C4::legal)
