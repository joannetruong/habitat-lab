/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020, 2021 Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#include <pybind11/pybind11.h>

#include "magnum/bootstrap.h"

namespace magnum { namespace platform {

template<class T, class Holder> void windowlessapplication(py::class_<T, Holder>& c) {
    py::class_<typename T::Configuration> configuration{c, "Configuration", "Configuration"};
    configuration
        .def(py::init());
        /** @todo others */

    c
        /* Constructor */
        .def(py::init<const typename T::Configuration&>(), py::arg("configuration") = typename T::Configuration{},
            "Constructor")
        /** @todo the nocreate ones */

        .def("exec", &T::exec, "Execute application")
        /** @todo more */
        ;
}

template<class T> void context(py::class_<T>& c) {
    c
        .def(py::init())
        /** @todo delayed creation */
        ;
}

}}
