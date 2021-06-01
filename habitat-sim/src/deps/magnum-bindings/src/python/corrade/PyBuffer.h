#ifndef corrade_PyBuffer_h
#define corrade_PyBuffer_h
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
#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/StridedArrayView.h>

#include "Corrade/PythonBindings.h"

#include "bootstrap.h"

namespace corrade {

/* pybind's py::buffer_info is EXTREMELY USELESS IT HURTS (and also allocates
   like hell), doing my own thing here instead. IMAGINE, I can pass flags to
   say what features I'm able to USE! WOW! */

template<class Class, bool(*getter)(Class&, Py_buffer&, int)> void enableBetterBufferProtocol(py::object& object) {
    auto& typeObject = reinterpret_cast<PyHeapTypeObject&>(*object.ptr());
    /* Sanity check -- we expect pybind set up its own buffer functions before
       us */
    CORRADE_INTERNAL_ASSERT(typeObject.as_buffer.bf_getbuffer == pybind11::detail::pybind11_getbuffer);
    CORRADE_INTERNAL_ASSERT(typeObject.as_buffer.bf_releasebuffer == pybind11::detail::pybind11_releasebuffer);

    typeObject.as_buffer.bf_getbuffer = [](PyObject *obj, Py_buffer *buffer, int flags) {
        CORRADE_INTERNAL_ASSERT(!PyErr_Occurred() && buffer);

        /* Zero-initialize the output and ask the class to fill it. If that
           fails for some reason, give up. Need to list all members otherwise
           GCC 4.8 loudly complains about missing initializers. */
        *buffer = Py_buffer{nullptr, nullptr, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
        if(!getter(pyInstanceFromHandle<Class>(obj), *buffer, flags)) {
            CORRADE_INTERNAL_ASSERT(!buffer->obj);
            CORRADE_INTERNAL_ASSERT(PyErr_Occurred());
            return -1;
        }

        /* Set the memory owner to the object and increase its reference count.
           We need to keep the object around because buffer->shapes /
           buffer->strides might be refering to it, moreover setting it to
           something else (like ArrayView's memory owner object) would mean
           Python calls the releasebuffer on that object instead of on us,
           leading to reference count getting negative in many cases. */
        CORRADE_INTERNAL_ASSERT(!buffer->obj);
        buffer->obj = obj;
        Py_INCREF(buffer->obj);
        return 0;
    };
    /* No need to release anything, we haven't made any garbage in the first
       place */
    typeObject.as_buffer.bf_releasebuffer = nullptr;
}

}

#endif
