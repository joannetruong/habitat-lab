#ifndef Magnum_GL_PythonBindings_h
#define Magnum_GL_PythonBindings_h
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

#include <memory> /* :( */
#include <vector>
#include <pybind11/pybind11.h>
#include <Magnum/GL/GL.h>

#include "Magnum/PythonBindings.h"

namespace Magnum { namespace GL {

/* Stores additional stuff needed for proper refcounting of buffers owned by
   a mesh. For some reason it *has to be* templated, otherwise
   PYBIND11_DECLARE_HOLDER_TYPE doesn't work. Ugh. */
template<class T> struct PyMeshHolder: std::unique_ptr<T> {
    static_assert(std::is_same<T, GL::Mesh>::value, "mesh holder has to hold a mesh");

    explicit PyMeshHolder(T* object): std::unique_ptr<T>{object} {}

    std::vector<pybind11::object> buffers;
};

template<class T> struct PyFramebufferHolder: std::unique_ptr<T, PyNonDestructibleBaseDeleter<T, std::is_destructible<T>::value>> {
    static_assert(std::is_same<T, GL::Framebuffer>::value, "framebuffer holder has to hold a framebuffer");

    explicit PyFramebufferHolder(T* object): std::unique_ptr<T, PyNonDestructibleBaseDeleter<T, std::is_destructible<T>::value>>{object} {}

    std::vector<pybind11::object> attachments;
};

}}

PYBIND11_DECLARE_HOLDER_TYPE(T, Magnum::GL::PyMeshHolder<T>)
PYBIND11_DECLARE_HOLDER_TYPE(T, Magnum::GL::PyFramebufferHolder<T>)

#endif
