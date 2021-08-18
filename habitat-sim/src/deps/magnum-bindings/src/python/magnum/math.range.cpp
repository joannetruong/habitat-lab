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
#include <pybind11/operators.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>

#include "magnum/bootstrap.h"
#include "magnum/math.h"

namespace magnum {

namespace {

template<class T> void range(py::module_& m, py::class_<T>& c) {
    /*
        Missing APIs:

        VectorType
    */

    py::implicitly_convertible<std::pair<typename T::VectorType, typename T::VectorType>, T>();

    c
        /* Constructors */
        .def_static("from_size", &T::fromSize, "Create a range from minimal coordinates and size")
        .def_static("from_center", &T::fromCenter, "Create a range from center and half size")
        .def_static("zero_init", []() {
            return T{Math::ZeroInit};
        }, "Construct a zero range")
        .def(py::init(), "Default constructor")
        .def(py::init<typename T::VectorType, typename T::VectorType>(), "Construct a range from minimal and maximal coordinates")
        .def(py::init<std::pair<typename T::VectorType, typename T::VectorType>>(), "Construct a range from minimal and maximal coordinates")

        /* Comparison */
        .def(py::self == py::self, "Equality comparison")
        .def(py::self != py::self, "Non-equality comparison")

        /* Properties */
        .def_property("min",
            static_cast<const typename T::VectorType(T::*)() const>(&T::min),
            [](T& self, const typename T::VectorType& value) { self.min() = value; },
            "Minimal coordinates (inclusive)")
        .def_property("max",
            static_cast<const typename T::VectorType(T::*)() const>(&T::max),
            [](T& self, const typename T::VectorType& value) { self.max() = value; },
            "Maximal coordinates (exclusive)")

        /* Methods */
        .def("size", &T::size, "Range size")
        .def("center", &T::center, "Range center")
        .def("translated", &T::translated, "Translated range")
        .def("padded", &T::padded, "Padded ange")
        .def("scaled", &T::scaled, "Scaled range")
        .def("scaled_from_center", &T::scaledFromCenter, "Range scaled from the center")

        .def("contains", static_cast<bool(T::*)(const typename T::VectorType&) const>(&T::contains),
            "Whether given point is contained inside the range")
        .def("contains", [](const T& self, const T& other) {
            return self.contains(other);
        }, "Whether another range is fully contained inside this range")

        .def("__repr__", repr<T>, "Object representation");

    m
        /* Free functions */
        .def("join", [](const T& a, const T& b) -> T {
            return Math::join(a, b);
        }, "Join two ranges")
        .def("intersect", [](const T& a, const T& b) -> T {
            return Math::intersect(a, b);
        }, "intersect two ranges")
        .def("intersects", [](const T& a, const T& b) {
            return Math::intersects(a, b);
        }, "Whether two ranges intersect");
}

template<class T> void range2D(py::class_<T>& c) {
    py::implicitly_convertible<std::pair<std::tuple<typename T::VectorType::Type, typename T::VectorType::Type>, std::tuple<typename T::VectorType::Type, typename T::VectorType::Type>>, T>();

    c
        /* Constructors */
        .def(py::init([](const std::pair<std::tuple<typename T::VectorType::Type, typename T::VectorType::Type>, std::tuple<typename T::VectorType::Type, typename T::VectorType::Type>>& value) {
            return T{typename T::VectorType{std::get<0>(value.first), std::get<1>(value.first)},
                      typename T::VectorType{std::get<0>(value.second), std::get<1>(value.second)}};
        }), "Construct a range from a pair of minimal and maximal coordinates")

        /* Properties */
        .def_property("bottom_left",
            static_cast<typename T::VectorType(T::*)() const>(&T::bottomLeft),
            [](T& self, const typename T::VectorType& value) {
                self.bottomLeft() = value;
            },
            "Bottom left corner")
        .def_property("bottom_right",
            static_cast<typename T::VectorType(T::*)() const>(&T::bottomRight),
            [](T& self, const typename T::VectorType& value) {
                self.max().x() = value.x();
                self.min().y() = value.y();
            },
            "Bottom right corner")
        .def_property("top_left",
            static_cast<typename T::VectorType(T::*)() const>(&T::topLeft),
            [](T& self, const typename T::VectorType& value) {
                self.min().x() = value.x();
                self.max().y() = value.y();
            },
            "Top left corner")
        .def_property("top_right",
            static_cast<typename T::VectorType(T::*)() const>(&T::topRight),
            [](T& self, const typename T::VectorType& value) {
                self.topRight() = value;
            },
            "Top right corner")

        .def_property("left",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::left),
            [](T& self, const typename T::VectorType::Type& value) {
                self.left() = value;
            },
            "Left edge")
        .def_property("right",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::right),
            [](T& self, const typename T::VectorType::Type& value) {
                self.right() = value;
            },
            "Right edge")
        .def_property("bottom",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::bottom),
            [](T& self, const typename T::VectorType::Type& value) {
                self.bottom() = value;
            },
            "Bottom edge")
        .def_property("top",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::top),
            [](T& self, const typename T::VectorType::Type& value) {
                self.top() = value;
            },
            "Top edge")

        /* Methods */
        .def("x", &T::x, "Range in the X axis")
        .def("y", &T::y, "Range in the Y axis")
        .def("size_x", &T::sizeX, "Range width")
        .def("size_y", &T::sizeY, "Range height")
        .def("center_x", &T::centerX, "Range center on X axis")
        .def("center_y", &T::centerY, "Range center on Y axis");
}

template<class T> void range3D(py::class_<T>& c) {
    py::implicitly_convertible<std::pair<std::tuple<typename T::VectorType::Type, typename T::VectorType::Type, typename T::VectorType::Type>, std::tuple<typename T::VectorType::Type, typename T::VectorType::Type, typename T::VectorType::Type>>, T>();

    c
        /* Constructors */
        .def(py::init([](const std::pair<std::tuple<typename T::VectorType::Type, typename T::VectorType::Type, typename T::VectorType::Type>, std::tuple<typename T::VectorType::Type, typename T::VectorType::Type, typename T::VectorType::Type>>& value) {
            return T{typename T::VectorType{std::get<0>(value.first), std::get<1>(value.first), std::get<2>(value.first)},
                      typename T::VectorType{std::get<0>(value.second), std::get<1>(value.second), std::get<2>(value.second)}};
        }), "Construct a range from a pair of minimal and maximal coordinates")

        /* Properties */
        .def_property("back_bottom_left",
            static_cast<typename T::VectorType(T::*)() const>(&T::backBottomLeft),
            [](T& self, const typename T::VectorType& value) {
                self.backBottomLeft() = value;
            },
            "Back bottom left corner")
        .def_property("back_bottom_right",
            static_cast<typename T::VectorType(T::*)() const>(&T::backBottomRight),
            [](T& self, const typename T::VectorType& value) {
                self.max().x() = value.x();
                self.min().y() = value.y();
                self.min().z() = value.z();
            },
            "Back bottom right corner")
        .def_property("back_top_left",
            static_cast<typename T::VectorType(T::*)() const>(&T::backTopLeft),
            [](T& self, const typename T::VectorType& value) {
                self.min().x() = value.x();
                self.max().y() = value.y();
                self.min().z() = value.z();
            },
            "Back top left corner")
        .def_property("back_top_right",
            static_cast<typename T::VectorType(T::*)() const>(&T::backTopRight),
            [](T& self, const typename T::VectorType& value) {
                self.max().x() = value.x();
                self.max().y() = value.y();
                self.min().z() = value.z();
            },
            "Back top right corner")
        .def_property("front_bottom_left",
            static_cast<typename T::VectorType(T::*)() const>(&T::frontBottomLeft),
            [](T& self, const typename T::VectorType& value) {
                self.min().x() = value.x();
                self.min().y() = value.y();
                self.max().z() = value.z();
            },
            "Front bottom left corner")
        .def_property("front_bottom_right",
            static_cast<typename T::VectorType(T::*)() const>(&T::frontBottomRight),
            [](T& self, const typename T::VectorType& value) {
                self.max().x() = value.x();
                self.min().y() = value.y();
                self.max().z() = value.z();
            },
            "Front bottom right corner")
        .def_property("front_top_right",
            static_cast<typename T::VectorType(T::*)() const>(&T::frontTopRight),
            [](T& self, const typename T::VectorType& value) {
                self.frontTopRight() = value;
            },
            "Front top right corner")
        .def_property("front_top_left",
            static_cast<typename T::VectorType(T::*)() const>(&T::frontTopLeft),
            [](T& self, const typename T::VectorType& value) {
                self.min().x() = value.x();
                self.max().y() = value.y();
                self.max().z() = value.z();
            },
            "Front top left corner")

        .def_property("left",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::left),
            [](T& self, const typename T::VectorType::Type& value) {
                self.left() = value;
            },
            "Left edge")
        .def_property("right",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::right),
            [](T& self, const typename T::VectorType::Type& value) {
                self.right() = value;
            },
            "Right edge")
        .def_property("bottom",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::bottom),
            [](T& self, const typename T::VectorType::Type& value) {
                self.bottom() = value;
            },
            "Bottom edge")
        .def_property("top",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::top),
            [](T& self, const typename T::VectorType::Type& value) {
                self.top() = value;
            },
            "Top edge")
        .def_property("back",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::back),
            [](T& self, const typename T::VectorType::Type& value) {
                self.back() = value;
            },
            "Back edge")
        .def_property("front",
            static_cast<typename T::VectorType::Type(T::*)() const>(&T::front),
            [](T& self, const typename T::VectorType::Type& value) {
                self.front() = value;
            },
            "Front edge")

        /* Methods */
        .def("x", &T::x, "Range in the X axis")
        .def("y", &T::y, "Range in the Y axis")
        .def("z", &T::z, "Range in the Z axis")
        .def("xy", &T::xy, "Range in the XY plane")
        .def("size_x", &T::sizeX, "Range width")
        .def("size_y", &T::sizeY, "Range height")
        .def("size_z", &T::sizeZ, "Range depth")
        .def("center_x", &T::centerX, "Range center on X axis")
        .def("center_y", &T::centerY, "Range center on Y axis")
        .def("center_z", &T::centerZ, "Range center on Z axis");
}

template<class U, template<UnsignedInt, class> class Type, UnsignedInt dimensions, class T, class ...Args> void convertibleImplementation(py::class_<Type<dimensions, T>, Args...>& c, std::false_type) {
    c.def(py::init<Type<dimensions, U>>(), "Construct from different underlying type");
}

template<class U, template<class> class Type, class T, class ...Args> void convertibleImplementation(py::class_<Type<T>, Args...>& c, std::false_type) {
    c.def(py::init<Type<U>>(), "Construct from different underlying type");
}

template<class U, template<UnsignedInt, class> class Type, UnsignedInt dimensions, class T, class ...Args> void convertibleImplementation(py::class_<Type<dimensions, T>, Args...>&, std::true_type) {}

template<class U, template<class> class Type, class T, class ...Args> void convertibleImplementation(py::class_<Type<T>, Args...>&, std::true_type) {}

template<template<UnsignedInt, class> class Type, UnsignedInt dimensions, class T, class ...Args> void convertible(py::class_<Type<dimensions, T>, Args...>& c) {
    convertibleImplementation<Int>(c, std::is_same<T, Int>{});
    convertibleImplementation<Float>(c, std::is_same<T, Float>{});
    convertibleImplementation<Double>(c, std::is_same<T, Double>{});
}

template<template<class> class Type, class T, class ...Args> void convertible(py::class_<Type<T>, Args...>& c) {
    convertibleImplementation<Int>(c, std::is_same<T, Int>{});
    convertibleImplementation<Float>(c, std::is_same<T, Float>{});
    convertibleImplementation<Double>(c, std::is_same<T, Double>{});
}

}

void mathRange(py::module_& root, py::module_& m) {
    py::class_<Range1D> range1D_{root, "Range1D", "One-dimensional float range"};
    py::class_<Range2D> range2D_{root, "Range2D", "Two-dimensional float range"};
    py::class_<Range3D> range3D_{root, "Range3D", "Three-dimensional float range"};

    py::class_<Range1Di> range1Di{root, "Range1Di", "One-dimensional float range"};
    py::class_<Range2Di> range2Di{root, "Range2Di", "Two-dimensional float range"};
    py::class_<Range3Di> range3Di{root, "Range3Di", "Three-dimensional float range"};

    py::class_<Range1Dd> range1Dd{root, "Range1Dd", "One-dimensional double range"};
    py::class_<Range2Dd> range2Dd{root, "Range2Dd", "Two-dimensional double range"};
    py::class_<Range3Dd> range3Dd{root, "Range3Dd", "Three-dimensional double range"};

    convertible(range1D_);
    convertible(range2D_);
    convertible(range3D_);
    convertible(range1Di);
    convertible(range2Di);
    convertible(range3Di);
    convertible(range1Dd);
    convertible(range2Dd);
    convertible(range3Dd);

    range(m, range1D_);
    range(m, range2D_);
    range(m, range3D_);
    range(m, range1Di);
    range(m, range2Di);
    range(m, range3Di);
    range(m, range1Dd);
    range(m, range2Dd);
    range(m, range3Dd);

    range2D(range2D_);
    range2D(range2Di);
    range2D(range2Dd);

    range3D(range3D_);
    range3D(range3Di);
    range3D(range3Dd);
}

}
