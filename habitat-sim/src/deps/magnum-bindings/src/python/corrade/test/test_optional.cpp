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

#include "../bootstrap.h" /* for module / _module alias */

#include "Corrade/Containers/OptionalPythonBindings.h"

using namespace Corrade;
namespace py = pybind11;

namespace {

struct Foo {
    Foo(int a): a{a} {}
    int a;
};

Containers::Optional<int> simpleType(bool set) {
    return set ? Containers::optional(5) : Containers::NullOpt;
}

Containers::Optional<Foo> nestedType(bool set) {
    return set ? Containers::optional(Foo{15}) : Containers::NullOpt;
}

int acquireSimpleType(Containers::Optional<int> value) {
    return value ? *value : -1;
}

int acquireNestedType(Containers::Optional<Foo> value) {
    return value ? value->a : -1;
}

}

/* TODO: remove declaration when https://github.com/pybind/pybind11/pull/1863
   is released */
extern "C" PYBIND11_EXPORT PyObject* PyInit_test_optional();
PYBIND11_MODULE(test_optional, m) {
    py::module_::import("corrade.containers");

    py::class_<Foo>{m, "Foo"}
        .def(py::init<int>())
        .def_readwrite("a", &Foo::a);

    m.def("simple_type", simpleType);
    m.def("nested_type", nestedType);

    m.def("acquire_simple_type", acquireSimpleType);
    m.def("acquire_nested_type", acquireNestedType);
}
