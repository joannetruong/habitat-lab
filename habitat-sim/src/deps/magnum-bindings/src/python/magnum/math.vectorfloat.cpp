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

#include "magnum/math.vector.h"

namespace magnum {

namespace {

template<class T> void vectorFloat(py::module_& m, py::class_<T>& c) {
    m
        .def("angle", [](const T& a, const T& b) { return Radd(Math::angle(a, b)); },
            "Angle between normalized vectors", py::arg("normalized_a"), py::arg("normalized_b"));

    c
        .def("is_normalized", &T::isNormalized, "Whether the vector is normalized")
        .def("length", &T::length, "Vector length")

        /* Cast needed because these are enabled only for floats */
        .def("length_inverted", static_cast<typename T::Type(T::*)() const>(&T::lengthInverted), "Inverse vector length")
        .def("normalized", static_cast<T(T::*)() const>(&T::normalized),
             "Normalized vector (of unit length)")
        .def("resized", static_cast<T(T::*)(typename T::Type) const>(&T::resized),
             "Resized vector")
        .def("projected", [](const T& self, const T& line) {
            return self.projected(line);
        }, "Vector projected onto a line")
        .def("projected_onto_normalized", [](const T& self, const T& line) {
            return self.projectedOntoNormalized(line);
        }, "Vector projected onto a normalized line");
}

template<class T> void vectorsFloat(py::module_& m, py::class_<Math::Vector2<T>>& vector2_, py::class_<Math::Vector3<T>>& vector3_, py::class_<Math::Vector4<T>>& vector4_) {
    vector2_.def("aspect_ratio", static_cast<T(Math::Vector2<T>::*)() const>(&Math::Vector2<T>::aspectRatio),
        "Aspect ratio");
    m.def("cross", static_cast<T(*)(const Math::Vector2<T>&, const Math::Vector2<T>&)>(Math::cross),
        "2D cross product");
    everyVector(vector2_);
    everyVectorSigned(vector2_);
    vector<Math::Vector2<T>>(m, vector2_);
    vectorFloat<Math::Vector2<T>>(m, vector2_);
    vector2<T>(vector2_);
    vector2Signed<T>(vector2_);

    m.def("cross", static_cast<Math::Vector3<T>(*)(const Math::Vector3<T>&, const Math::Vector3<T>&)>(Math::cross),
        "Cross product");
    everyVector(vector3_);
    everyVectorSigned(vector3_);
    vector<Math::Vector3<T>>(m, vector3_);
    vectorFloat<Math::Vector3<T>>(m, vector3_);
    vector3<T>(vector3_);

    everyVector(vector4_);
    everyVectorSigned(vector4_);
    vector<Math::Vector4<T>>(m, vector4_);
    vectorFloat<Math::Vector4<T>>(m, vector4_);
    vector4<T>(vector4_);
}

}

void mathVectorFloat(py::module_& root, py::module_& m) {
    py::class_<Vector2> vector2{root, "Vector2", "Two-component float vector", py::buffer_protocol{}};
    py::class_<Vector3> vector3{root, "Vector3", "Three-component float vector", py::buffer_protocol{}};
    py::class_<Vector4> vector4{root, "Vector4", "Four-component float vector", py::buffer_protocol{}};
    py::class_<Vector2d> vector2d{root, "Vector2d", "Two-component double vector", py::buffer_protocol{}};
    py::class_<Vector3d> vector3d{root, "Vector3d", "Three-component double vector", py::buffer_protocol{}};
    py::class_<Vector4d> vector4d{root, "Vector4d", "Four-component double vector", py::buffer_protocol{}};

    /* The subclasses don't have buffer protocol enabled, as that's already
       done by the base classes. Moreover, just adding py::buffer_protocol{}
       would cause it to not find the buffer functions as we don't add them
       anywhere, thus failing with `pybind11_getbuffer(): Internal error`. */
    py::class_<Color3, Vector3> color3_{root, "Color3", "Color in linear RGB color space"};
    py::class_<Color4, Vector4> color4_{root, "Color4", "Color in linear RGBA color space"};

    /* Register the integer types first, only after that register type
       conversions because they need all the types */
    mathVectorIntegral(root, m);

    /* Register type conversions as soon as possible as those should have a
       priority over buffer and list constructors. These need all the types to
       be present, so can't be interwinted with the class definitions above. */
    convertible(vector2);
    convertible(vector3);
    convertible(vector4);
    convertible(vector2d);
    convertible(vector3d);
    convertible(vector4d);
    /* Colors are float-only at the moment, thus no conversions */

    /* This needs to be before buffer constructors otherwise a buffer
       constructor gets picked and it will fail because there are just 3
       elements */
    color4from3(color4_);

    /* This needs to be *after* conversion constructors so the type conversion
       gets picked before the general buffer constructor (which would then
       fail). On the other hand, this needs to be before generic from-list
       constructors because buffer protocol is generally faster than
       iteration. */
    everyVectorBuffer(vector2);
    everyVectorBuffer(vector3);
    everyVectorBuffer(vector4);
    everyVectorBuffer(vector2d);
    everyVectorBuffer(vector3d);
    everyVectorBuffer(vector4d);
    everyVectorBuffer(color3_);
    everyVectorBuffer(color4_);

    /* Now register the generic from-list constructors and everything else */
    vectorsFloat<Float>(m, vector2, vector3, vector4);
    vectorsFloat<Double>(m, vector2d, vector3d, vector4d);
    everyVector(color3_);
    everyVectorSigned(color3_);
    color(color3_);
    color3(color3_);
    everyVector(color4_);
    everyVectorSigned(color4_);
    color(color4_);
    color4(color4_);
}

}
