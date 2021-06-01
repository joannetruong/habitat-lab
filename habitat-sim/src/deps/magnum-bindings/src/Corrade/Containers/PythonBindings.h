#ifndef Corrade_Containers_PythonBindings_h
#define Corrade_Containers_PythonBindings_h
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
#include <pybind11/pybind11.h>

namespace Corrade { namespace Containers {

/* Stores additional stuff needed for proper refcounting of array views. Better
   than subclassing ArrayView because then we would need to wrap it every time
   it's exposed to Python, making 3rd party bindings unnecessarily complex. */
template<class T> struct PyArrayViewHolder: std::unique_ptr<T> {
    explicit PyArrayViewHolder(T* object): PyArrayViewHolder{object, pybind11::none{}} {
        /* Array view without an owner can only be empty */
        CORRADE_INTERNAL_ASSERT(!object->data());
    }

    explicit PyArrayViewHolder(T* object, pybind11::object owner): std::unique_ptr<T>{object}, owner{std::move(owner)} {}

    pybind11::object owner;
};

template<class T> PyArrayViewHolder<T> pyArrayViewHolder(const T& view, pybind11::object owner) {
    return PyArrayViewHolder<T>{new T{view}, owner};
}

}}

PYBIND11_DECLARE_HOLDER_TYPE(T, Corrade::Containers::PyArrayViewHolder<T>)

#endif
