#ifndef magnum_math_vector_h
#define magnum_math_vector_h
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

#include <pybind11/operators.h>
#include <Corrade/Containers/ScopeGuard.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Vector4.h>

#include "corrade/PyBuffer.h"

#include "magnum/math.h"

namespace magnum {

template<class> constexpr bool isTypeCompatible(char);
template<> constexpr bool isTypeCompatible<Float>(char format) {
    return format == 'f' || format == 'd';
}
template<> constexpr bool isTypeCompatible<Double>(char format) {
    return format == 'f' || format == 'd';
}
template<> constexpr bool isTypeCompatible<Int>(char format) {
    return format == 'i' || format == 'l';
}
template<> constexpr bool isTypeCompatible<UnsignedInt>(char format) {
    return format == 'I' || format == 'L';
}

template<class U, class T> void initFromBuffer(T& out, const Py_buffer& buffer) {
    for(std::size_t i = 0; i != T::Size; ++i)
        out[i] = static_cast<typename T::Type>(*reinterpret_cast<const U*>(static_cast<const char*>(buffer.buf) + i*buffer.strides[0]));
}

/* Floating-point init */
template<class T> void initFromBuffer(typename std::enable_if<std::is_floating_point<typename T::Type>::value, T>::type& out, const Py_buffer& buffer) {
    if(buffer.format[0] == 'f') initFromBuffer<Float>(out, buffer);
    else if(buffer.format[0] == 'd') initFromBuffer<Double>(out, buffer);
    else CORRADE_INTERNAL_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */
}

/* Signed integral init */
template<class T> void initFromBuffer(typename std::enable_if<std::is_integral<typename T::Type>::value && std::is_signed<typename T::Type>::value, T>::type& out, const Py_buffer& buffer) {
    if(buffer.format[0] == 'i') initFromBuffer<Int>(out, buffer);
    else if(buffer.format[0] == 'l') initFromBuffer<Long>(out, buffer);
    else CORRADE_INTERNAL_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */
}

/* Unsigned integral init */
template<class T> void initFromBuffer(typename std::enable_if<std::is_integral<typename T::Type>::value && std::is_unsigned<typename T::Type>::value, T>::type& out, const Py_buffer& buffer) {
    if(buffer.format[0] == 'I') initFromBuffer<UnsignedInt>(out, buffer);
    else if(buffer.format[0] == 'L') initFromBuffer<UnsignedLong>(out, buffer);
    else CORRADE_INTERNAL_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */
}

/* Things that have to be defined for both VectorN and Color so they construct
   / return a proper type */
template<class T, class ...Args> void everyVector(py::class_<T, Args...>& c) {
    /* Implicitly convertible from a buffer (which is a numpy array as well).
       Without this implicit conversion from numpy arrays sometimes doesn't
       work. */
    py::implicitly_convertible<py::buffer, T>();

    c
        /* Constructors */
        .def_static("zero_init", []() {
            return T{Math::ZeroInit};
        }, "Construct a zero vector")
        .def(py::init(), "Default constructor")

        /* Operators */
        .def(py::self += py::self, "Add and assign a vector")
        .def(py::self + py::self, "Add a vector")
        #ifdef __clang__
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
        #endif
        .def(py::self -= py::self, "Subtract and assign a vector")
        #ifdef __clang__
        #pragma GCC diagnostic pop
        #endif
        .def(py::self - py::self, "Subtract a vector")
        .def(py::self *= typename T::Type{}, "Multiply with a scalar and assign")
        .def(py::self * typename T::Type{}, "Multiply with a scalar")
        .def(py::self /= typename T::Type{}, "Divide with a scalar and assign")
        .def(py::self / typename T::Type{}, "Divide with a scalar")
        .def(py::self *= py::self, "Multiply a vector component-wise and assign")
        .def(py::self * py::self, "Multiply a vector component-wise")
        #ifdef __clang__
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
        #endif
        .def(py::self /= py::self, "Divide a vector component-wise and assign")
        #ifdef __clang__
        #pragma GCC diagnostic pop
        #endif
        .def(py::self / py::self, "Divide a vector component-wise")
        .def(typename T::Type{} * py::self, "Multiply a scalar with a vector")
        .def(typename T::Type{} / py::self, "Divide a vector with a scalar and invert");
}

template<class T, class ...Args> void everyVectorSigned(py::class_<T, Args...>& c) {
    c.def(-py::self, "Negated vector");
}

/* Separate because it needs to be registered after the type conversion
   constructors. Needs to be called also for subclasses. */
template<class T, class ...Args> void everyVectorBuffer(py::class_<T, Args...>& c) {
    c
        /* Buffer protocol. If not present, implicit conversion from numpy
           arrays of non-default types somehow doesn't work. There's also the
           other part in vectorBuffer(). */
        .def(py::init([](py::buffer other) {
            /* GCC 4.8 otherwise loudly complains about missing initializers */
            Py_buffer buffer{nullptr, nullptr, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
            if(PyObject_GetBuffer(other.ptr(), &buffer, PyBUF_FORMAT|PyBUF_STRIDES) != 0)
                throw py::error_already_set{};

            Containers::ScopeGuard e{&buffer, PyBuffer_Release};

            if(buffer.ndim != 1) {
                PyErr_Format(PyExc_BufferError, "expected 1 dimension but got %i", buffer.ndim);
                throw py::error_already_set{};
            }

            if(buffer.shape[0] != T::Size) {
                PyErr_Format(PyExc_BufferError, "expected %zu elements but got %zi", T::Size, buffer.shape[0]);
                throw py::error_already_set{};
            }

            /* Expecting just an one-letter format */
            if(!buffer.format[0] || buffer.format[1] || !isTypeCompatible<typename T::Type>(buffer.format[0])) {
                PyErr_Format(PyExc_BufferError, "unexpected format %s for a %s vector", buffer.format, FormatStrings[formatIndex<typename T::Type>()]);
                throw py::error_already_set{};
            }

            T out{NoInit};
            initFromBuffer<T>(out, buffer);
            return out;
        }), "Construct from a buffer");
}

template<class T> bool vectorBufferProtocol(T& self, Py_buffer& buffer, int flags) {
    /* I hate the const_casts but I assume this is to make editing easier, NOT
       to make it possible for users to stomp on these values. */
    buffer.ndim = 1;
    buffer.itemsize = sizeof(typename T::Type);
    buffer.len = sizeof(T);
    buffer.buf = self.data();
    buffer.readonly = false;
    if((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
        buffer.format = const_cast<char*>(FormatStrings[formatIndex<typename T::Type>()]);
    if(flags != PyBUF_SIMPLE) {
        /* Reusing shape definitions from matrices because I don't want to
           create another useless array for that and reinterpret_cast on the
           buffer.internal is UGLY. It's flipped from column-major to
           row-major, so adjusting the row instead. */
        buffer.shape = const_cast<Py_ssize_t*>(MatrixShapes[matrixShapeStrideIndex<2, T::Size>()]);
        CORRADE_INTERNAL_ASSERT(buffer.shape[0] == T::Size);
        if((flags & PyBUF_STRIDES) == PyBUF_STRIDES)
            buffer.strides = &buffer.itemsize;
    }

    return true;
}

/* Things common for vectors of all sizes and types */
template<class T> void vector(py::module_& m, py::class_<T>& c) {
    /*
        Missing APIs:

        from(T*)
        Type
        VectorNi * VectorN and variants (5)
    */

    m
        .def("dot", [](const T& a, const T& b) { return Math::dot(a, b); },
            "Dot product of two vectors");

    c
        /* Constructors */
        .def(py::init<typename T::Type>(), "Construct a vector with one value for all components")

        /* Comparison */
        .def(py::self == py::self, "Equality comparison")
        .def(py::self != py::self, "Non-equality comparison")
        .def(py::self < py::self, "Component-wise less than comparison")
        .def(py::self > py::self, "Component-wise greater than comparison")
        .def(py::self <= py::self, "Component-wise less than or equal comparison")
        .def(py::self >= py::self, "Component-wise greater than or equal comparison")

        /* Set / get. Need to raise IndexError in order to allow iteration:
           https://docs.python.org/3/reference/datamodel.html#object.__getitem__
           Using error_already_set is slightly faster than throwing index_error
           directly, but still much slower than not throwing at all. Waiting
           for https://github.com/pybind/pybind11/pull/1853 to get merged. */
        .def("__setitem__", [](T& self, std::size_t i, typename T::Type value) {
            if(i >= T::Size) {
                PyErr_SetString(PyExc_IndexError, "");
                throw py::error_already_set{};
            }
            self[i] = value;
        }, "Set a value at given position")
        .def("__getitem__", [](const T& self, std::size_t i) {
            if(i >= T::Size) {
                PyErr_SetString(PyExc_IndexError, "");
                throw py::error_already_set{};
            }
            return self[i];
        }, "Value at given position")

        /* Swizzle */
        /* TODO: both of these could be *way* more efficiently implemented
           directly on PyObject (no need to throw, no need to do string
           conversions...) but then these wouldn't be visible to docs I fear */
        .def("__getattr__", [](T& self, const std::string& name) -> py::object {
            if(name.size() > 4) {
                PyErr_SetString(PyExc_AttributeError, "only four-component swizzles are supported at most");
                throw py::error_already_set{};
            }

            Math::Vector4<typename T::Type> out;
            for(std::size_t i = 0; i != name.size(); ++i) {
                if(name[i] == 'x' || name[i] == 'r') out[i] = self[0];
                else if(name[i] == 'y' || name[i] == 'g') out[i] = self[1];
                else if(T::Size > 2 && (name[i] == 'z' || name[i] == 'b')) out[i] = self[2];
                else if(T::Size > 3 && (name[i] == 'w' || name[i] == 'a')) out[i] = self[3];
                else {
                    PyErr_SetString(PyExc_AttributeError, "invalid swizzle");
                    throw py::error_already_set{};
                }
            }

            if(name.size() == 4) return py::cast(out);
            else if(name.size() == 3) return py::cast(out.xyz());
            else if(name.size() == 2) return py::cast(out.xy());
            /* this should be handled by the x/y/z/w/r/g/b/a properties instead */
            else CORRADE_INTERNAL_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */
        }, "Vector swizzle")
        .def("__setattr__", [](T& self, py::str nameO, py::object valueO) {
            std::string name = py::cast<std::string>(nameO);
            /* If the name is just one character, this is better handled by
               dedicated properties (and if not, it'll provide a better
               diagnostic than we can). Same for xy / xyz / ... when
               applicable, and when the name contains non-swizzle characters */
            if(name.size() == 1 ||
              (name.compare("xy") == 0 && T::Size > 2) ||
              (name.compare("xyz") == 0 && T::Size > 3) ||
              (name.compare("rgb") == 0 && T::Size > 3) ||
               name.find_first_not_of("xyzwrgba") != std::string::npos) {
                if(PySuper_Type.tp_setattro(py::cast(self).ptr(), nameO.ptr(), valueO.ptr()) != 0)
                    throw py::error_already_set{};
                return;
            }

            /* Here we can be certain it's a swizzle attempt, so raise clear
               error messages */
            const typename T::Type* data;
            std::size_t size;
            if(py::isinstance<Math::Vector2<typename T::Type>>(valueO)) {
                data = py::cast<const Math::Vector2<typename T::Type>&>(valueO).data();
                size = 2;
            } else if(py::isinstance<Math::Vector3<typename T::Type>>(valueO)) {
                data = py::cast<const Math::Vector3<typename T::Type>&>(valueO).data();
                size = 3;
            } else if(py::isinstance<Math::Vector4<typename T::Type>>(valueO)) {
                data = py::cast<const Math::Vector4<typename T::Type>&>(valueO).data();
                size = 4;
            } else {
                PyErr_SetString(PyExc_TypeError, "unrecognized swizzle type");
                throw py::error_already_set{};
            }

            if(name.size() != size) {
                PyErr_SetString(PyExc_TypeError, "swizzle doesn't match passed vector component count");
                throw py::error_already_set{};
            }
            for(std::size_t i = 0; i != name.size(); ++i) {
                if(name[i] == 'x' || name[i] == 'r') self[0] = data[i];
                else if(name[i] == 'y' || name[i] == 'g') self[1] = data[i];
                else if(T::Size > 2 && (name[i] == 'z' || name[i] == 'b')) self[2] = data[i];
                else if(T::Size > 3 && (name[i] == 'w' || name[i] == 'a')) self[3] = data[i];
                else {
                    PyErr_SetString(PyExc_AttributeError, "invalid swizzle");
                    throw py::error_already_set{};
                }
            }
        }, "Vector swizzle")

        /* Member functions common for floating-point and integer types */
        .def("is_zero", &T::isZero, "Whether the vector is zero")
        .def("dot", static_cast<typename T::Type(T::*)() const>(&T::dot), "Dot product of the vector")
        .def("flipped", &T::flipped, "Flipped vector")
        .def("sum", &T::sum, "Sum of values in the vector")
        .def("product", &T::product, "Product of values in the vector")
        .def("min", &T::min, "Minimal value in the vector")
        .def("max", &T::max, "Maximal value in the vector")
        .def("minmax", &T::minmax, "Minimal and maximal value in the vector")

        .def("__repr__", repr<T>, "Object representation");

    /* Ideally, only the constructor (in vectorBuffer()) would be needed
       (and thus also no py::buffer_protocol() specified for the class),
       but conversion of vectors to lists is extremely slow due to pybind
       exceptions being somehow extra heavy compared to native python ones,
       so in order to have acceptable performance we need the buffer
       protocol on the other side as well. See test/benchmark_math.py for more
       information. */
    corrade::enableBetterBufferProtocol<T, vectorBufferProtocol>(c);

    /* Vector length */
    char lenDocstring[] = "Vector size. Returns _.";
    lenDocstring[sizeof(lenDocstring) - 3] = '0' + T::Size;
    c.def_static("__len__", []() { return int(T::Size); }, lenDocstring);
}

template<class T> void vector2(py::class_<Math::Vector2<T>>& c) {
    py::implicitly_convertible<const std::tuple<T, T>&, Math::Vector2<T>>();

    c
        /* Constructors */
        .def(py::init<T, T>(), "Constructor")
        .def(py::init([](const std::tuple<T, T>& value) {
            return Math::Vector2<T>{std::get<0>(value), std::get<1>(value)};
        }), "Construct from a tuple")

        /* Static constructors */
        .def_static("x_axis", &Math::Vector2<T>::xAxis,
            "Vector in a direction of X axis (right)", py::arg("length") = T(1))
        .def_static("y_axis", &Math::Vector2<T>::yAxis,
            "Vector in a direction of Y axis (up)", py::arg("length") = T(1))
        .def_static("x_scale", &Math::Vector2<T>::xScale,
            "Scaling vector in a direction of X axis (width)", py::arg("scale"))
        .def_static("y_scale", &Math::Vector2<T>::yScale,
            "Scaling vector in a direction of Y axis (height)", py::arg("scale"))

        /* Properties */
        .def_property("x",
            static_cast<T(Math::Vector2<T>::*)() const>(&Math::Vector2<T>::x),
            [](Math::Vector2<T>& self, T value) { self.x() = value; },
            "X component")
        .def_property("y",
            static_cast<T(Math::Vector2<T>::*)() const>(&Math::Vector2<T>::y),
            [](Math::Vector2<T>& self, T value) { self.y() = value; },
            "Y component");
}

template<class T> void vector2Signed(py::class_<Math::Vector2<T>>& c) {
    /* Cast needed because these are enabled only for signed types */
    c.def("perpendicular", static_cast<Math::Vector2<T>(Math::Vector2<T>::*)() const>(&Math::Vector2<T>::perpendicular), "Perpendicular vector");
}

template<class T> void vector3(py::class_<Math::Vector3<T>>& c) {
    py::implicitly_convertible<const std::tuple<T, T, T>&, Math::Vector3<T>>();

    c
        /* Constructors */
        .def(py::init<T, T, T>(), "Constructor")
        .def(py::init([](const std::tuple<T, T, T>& value) {
            return Math::Vector3<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        }), "Construct from a tuple")

        /* Static constructors */
        .def_static("x_axis", &Math::Vector3<T>::xAxis,
            "Vector in a direction of X axis (right)", py::arg("length") = T(1))
        .def_static("y_axis", &Math::Vector3<T>::yAxis,
            "Vector in a direction of Y axis (up)", py::arg("length") = T(1))
        .def_static("z_axis", &Math::Vector3<T>::zAxis,
            "Vector in a direction of Z axis (backward)", py::arg("length") = T(1))
        .def_static("x_scale", &Math::Vector3<T>::xScale,
            "Scaling vector in a direction of X axis (width)", py::arg("scale"))
        .def_static("y_scale", &Math::Vector3<T>::yScale,
            "Scaling vector in a direction of Y axis (height)", py::arg("scale"))
        .def_static("z_scale", &Math::Vector3<T>::zScale,
            "Scaling vector in a direction of Z axis (depth)", py::arg("scale"))

        /* Properties */
        .def_property("x",
            static_cast<T(Math::Vector3<T>::*)() const>(&Math::Vector3<T>::x),
            [](Math::Vector3<T>& self, T value) { self.x() = value; },
            "X component")
        .def_property("y",
            static_cast<T(Math::Vector3<T>::*)() const>(&Math::Vector3<T>::y),
            [](Math::Vector3<T>& self, T value) { self.y() = value; },
            "Y component")
        .def_property("z",
            static_cast<T(Math::Vector3<T>::*)() const>(&Math::Vector3<T>::z),
            [](Math::Vector3<T>& self, T value) { self.z() = value; },
            "Z component")

        .def_property("r",
            static_cast<T(Math::Vector3<T>::*)() const>(&Math::Vector3<T>::r),
            [](Math::Vector3<T>& self, T value) { self.r() = value; },
            "R component")
        .def_property("g",
            static_cast<T(Math::Vector3<T>::*)() const>(&Math::Vector3<T>::g),
            [](Math::Vector3<T>& self, T value) { self.g() = value; },
            "G component")
        .def_property("b",
            static_cast<T(Math::Vector3<T>::*)() const>(&Math::Vector3<T>::b),
            [](Math::Vector3<T>& self, T value) { self.b() = value; },
            "B component")

        .def_property("xy",
            static_cast<const Math::Vector2<T>(Math::Vector3<T>::*)() const>(&Math::Vector3<T>::xy),
            [](Math::Vector3<T>& self, const Math::Vector2<T>& value) { self.xy() = value; },
            "XY part of the vector");
}

template<class T> void vector4(py::class_<Math::Vector4<T>>& c) {
    py::implicitly_convertible<const std::tuple<T, T, T, T>&, Math::Vector4<T>>();

    c
        /* Constructors */
        .def(py::init<T, T, T, T>(), "Constructor")
        .def(py::init([](const std::tuple<T, T, T, T>& value) {
            return Math::Vector4<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value)};
        }), "Construct from a tuple")

        /* Properties */
        .def_property("x",
            static_cast<T(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::x),
            [](Math::Vector4<T>& self, T value) { self.x() = value; },
            "X component")
        .def_property("y",
            static_cast<T(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::y),
            [](Math::Vector4<T>& self, T value) { self.y() = value; },
            "Y component")
        .def_property("z",
            static_cast<T(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::z),
            [](Math::Vector4<T>& self, T value) { self.z() = value; },
            "Z component")
        .def_property("w",
            static_cast<T(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::w),
            [](Math::Vector4<T>& self, T value) { self.w() = value; },
            "W component")

        .def_property("r",
            static_cast<T(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::r),
            [](Math::Vector4<T>& self, T value) { self.r() = value; },
            "R component")
        .def_property("g",
            static_cast<T(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::g),
            [](Math::Vector4<T>& self, T value) { self.g() = value; },
            "G component")
        .def_property("b",
            static_cast<T(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::b),
            [](Math::Vector4<T>& self, T value) { self.b() = value; },
            "B component")
        .def_property("a",
            static_cast<T(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::a),
            [](Math::Vector4<T>& self, T value) { self.a() = value; },
            "A component")

        .def_property("xyz",
            static_cast<const Math::Vector3<T>(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::xyz),
            [](Math::Vector4<T>& self, const Math::Vector3<T>& value) { self.xyz() = value; },
            "XYZ part of the vector")
        .def_property("rgb",
            static_cast<const Math::Vector3<T>(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::rgb),
            [](Math::Vector4<T>& self, const Math::Vector3<T>& value) { self.rgb() = value; },
            "RGB part of the vector")
        .def_property("xy",
            static_cast<const Math::Vector2<T>(Math::Vector4<T>::*)() const>(&Math::Vector4<T>::xy),
            [](Math::Vector4<T>& self, const Math::Vector2<T>& value) { self.xy() = value; },
            "XY part of the vector");
}

template<class U, template<class> class Type, class T, class ...Args> void convertibleImplementation(py::class_<Type<T>, Args...>& c, std::false_type) {
    c.def(py::init<Type<U>>(), "Construct from different underlying type");
}

template<class U, template<class> class Type, class T, class ...Args> void convertibleImplementation(py::class_<Type<T>, Args...>&, std::true_type) {}

template<template<class> class Type, class T, class ...Args> void convertible(py::class_<Type<T>, Args...>& c) {
    convertibleImplementation<UnsignedInt>(c, std::is_same<T, UnsignedInt>{});
    convertibleImplementation<Int>(c, std::is_same<T, Int>{});
    convertibleImplementation<Float>(c, std::is_same<T, Float>{});
    convertibleImplementation<Double>(c, std::is_same<T, Double>{});
}

template<class T, class Base> void color(py::class_<T, Base>& c) {
    c
        .def_static("zero_init", []() {
            return T{Math::ZeroInit};
        }, "Construct a zero color")
        .def(py::init(), "Default constructor");
}

template<class T> void color3(py::class_<Math::Color3<T>, Math::Vector3<T>>& c) {
    py::implicitly_convertible<const std::tuple<T, T, T>&, Math::Color3<T>>();

    c
        /* Constructors */
        .def(py::init<T, T, T>(), "Constructor")
        .def(py::init<T>(), "Construct with one value for all components")
        .def(py::init<Math::Vector3<T>>(), "Construct from a vector")
        .def(py::init([](const std::tuple<T, T, T>& value) {
            return Math::Color3<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        }), "Construct from a tuple")

        .def_static("from_hsv", [](Degd hue, typename Math::Color3<T>::FloatingPointType saturation, typename Math::Color3<T>::FloatingPointType value) {
            return Math::Color3<T>::fromHsv({Math::Deg<T>(hue), saturation, value});
        }, "Create RGB color from HSV representation", py::arg("hue"), py::arg("saturation"), py::arg("value"))

        .def_static("from_srgb", [](UnsignedInt srgb) {
            return Math::Color3<T>::fromSrgb(srgb);
        }, "Create linear RGB color from 24-bit sRGB representation", py::arg("srgb"))

        /* Accessors */
        .def("to_srgb_int", &Math::Color3<T>::toSrgbInt,
             "Convert to 32-bit integral sRGB representation")
        .def("to_hsv", [](Math::Color3<T>& self) {
            auto hsv = self.toHsv();
            return std::make_tuple(Degd(hsv.hue), hsv.saturation, hsv.value);
        }, "Convert to HSV representation")
        .def("hue", [](Math::Color3<T>& self) {
            return Degd(self.hue());
        }, "Hue")
        .def("saturation", &Math::Color3<T>::saturation, "Saturation")
        .def("value", &Math::Color3<T>::value, "Value");
}

/* Needs to be separate to make it a priority over buffer protocol */
template<class T> void color4from3(py::class_<Math::Color4<T>, Math::Vector4<T>>& c) {
    py::implicitly_convertible<const Math::Vector3<T>&, Math::Color4<T>>();

    c
        .def(py::init<Math::Vector3<T>, T>(), "Construct from a three-component color", py::arg("rgb"), py::arg("alpha") = Math::Implementation::fullChannel<T>())
        .def(py::init<Math::Vector4<T>>(), "Construct from a vector");
}

template<class T> void color4(py::class_<Math::Color4<T>, Math::Vector4<T>>& c) {
    py::implicitly_convertible<const std::tuple<T, T, T>&, Math::Color4<T>>();
    py::implicitly_convertible<const std::tuple<T, T, T, T>&, Math::Color4<T>>();

    c
        /* Constructors */
        .def(py::init<T, T, T, T>(), "Constructor", py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a") = Math::Implementation::fullChannel<T>())
        .def(py::init<T, T>(), "Construct with one value for all components", py::arg("rgb"), py::arg("alpha") = Math::Implementation::fullChannel<T>())
        .def(py::init([](const std::tuple<T, T, T>& value) {
            return Math::Color4<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        }), "Construct from a RGB tuple")
        .def(py::init([](const std::tuple<T, T, T, T>& value) {
            return Math::Color4<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value)};
        }), "Construct from a RGBA tuple")

        .def_static("from_hsv", [](Degd hue, typename Math::Color4<T>::FloatingPointType saturation, typename Math::Color4<T>::FloatingPointType value, T alpha) {
            return Math::Color4<T>::fromHsv({Math::Deg<T>(hue), saturation, value}, alpha);
        }, "Create RGB color from HSV representation", py::arg("hue"), py::arg("saturation"), py::arg("value"), py::arg("alpha") = Math::Implementation::fullChannel<T>())

        .def_static("from_srgb_alpha", [](UnsignedInt srgbAlpha) {
            return Math::Color4<T>::fromSrgbAlpha(srgbAlpha);
        }, "Create linear RGBA color from 32-bit sRGB a alpha representation", py::arg("srgb_alpha"))

        .def_static("from_srgb", [](UnsignedInt srgb, T a) {
            return Math::Color4<T>::fromSrgb(srgb, a);
        }, "Create linear RGBA color from 32-bit sRGB a alpha representation", py::arg("srgb"), py::arg("a") = Math::Implementation::fullChannel<T>())

        /* Accessors */
        .def("to_srgb_alpha_int", &Math::Color4<T>::toSrgbAlphaInt,
             "Convert to 32-bit integral sRGB + linear alpha representation")
        .def("to_hsv", [](Math::Color4<T>& self) {
            auto hsv = self.toHsv();
            return std::make_tuple(Degd(hsv.hue), hsv.saturation, hsv.value);
        }, "Convert to HSV representation")
        .def("hue", [](Math::Color4<T>& self) {
            return Degd(self.hue());
        }, "Hue")
        .def("saturation", &Math::Color4<T>::saturation, "Saturation")
        .def("value", &Math::Color4<T>::value, "Value")

        /* Properties */
        .def_property("xyz",
            static_cast<const Math::Color3<T>(Math::Color4<T>::*)() const>(&Math::Color4<T>::xyz),
            [](Math::Color4<T>& self, const Math::Color3<T>& value) { self.xyz() = value; },
            "XYZ part of the vector")
        .def_property("rgb",
            static_cast<const Math::Color3<T>(Math::Color4<T>::*)() const>(&Math::Color4<T>::rgb),
            [](Math::Color4<T>& self, const Math::Color3<T>& value) { self.rgb() = value; },
            "RGB part of the vector");
}

}

#endif
