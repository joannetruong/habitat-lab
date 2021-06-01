#ifndef magnum_bootstrap_h
#define magnum_bootstrap_h
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

#include <Python.h>
#include <pybind11/detail/common.h> /* for PYBIND11_VERSION_* */
#include <Magnum/Magnum.h>

namespace pybind11 {
    /* pybind11 2.6 changes py::module to py::module_ to be compatible with C++
       modules. In order to be forward-compatible, we use module_ everywhere
       and define it as an alias to module on < 2.6 */
    #if PYBIND11_VERSION_MAJOR*100 + PYBIND11_VERSION_MINOR >= 206
    class module_;
    #else
    class module;
    typedef module module_;
    #endif
}

namespace Magnum {}

namespace magnum {

using namespace Magnum;
namespace py = pybind11;

template<UnsignedInt dimensions, class T> struct PyDimensionTraits;
template<class T> struct PyDimensionTraits<1, T> {
    typedef T VectorType;
    static VectorType from(const Math::Vector<1, T>& vec) { return vec[0]; }
};
template<class T> struct PyDimensionTraits<2, T> {
    typedef Math::Vector2<T> VectorType;
    static VectorType from(const Math::Vector<2, T>& vec) { return vec; }
};
template<class T> struct PyDimensionTraits<3, T> {
    typedef Math::Vector3<T> VectorType;
    static VectorType from(const Math::Vector<3, T>& vec) { return vec; }
};

void math(py::module_& root, py::module_& m);
void mathVectorFloat(py::module_& root, py::module_& m);
void mathVectorIntegral(py::module_& root, py::module_& m);
void mathMatrixFloat(py::module_& root, PyTypeObject* metaclass);
void mathMatrixDouble(py::module_& root, PyTypeObject* metaclass);
void mathRange(py::module_& root, py::module_& m);

void gl(py::module_& m);
void meshtools(py::module_& m);
void primitives(py::module_& m);
void scenegraph(py::module_& m);
void shaders(py::module_& m);
void trade(py::module_& m);

namespace platform {
    void glfw(py::module_& m);
    void sdl2(py::module_& m);

    void egl(py::module_& m);
    void glx(py::module_& m);
}

}

#endif
