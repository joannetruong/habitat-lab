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

#include "magnum/math.matrix.h"

namespace magnum {

void mathMatrixFloat(py::module_& root, PyTypeObject* const metaclass) {
    py::class_<Matrix2x2> matrix2x2{root, "Matrix2x2", "2x2 float matrix", py::buffer_protocol{}};
    py::class_<Matrix2x3> matrix2x3{root, "Matrix2x3", "2x3 float matrix", py::buffer_protocol{}};
    py::class_<Matrix2x4> matrix2x4{root, "Matrix2x4", "2x4 float matrix", py::buffer_protocol{}};

    py::class_<Matrix3x2> matrix3x2{root, "Matrix3x2", "3x2 float matrix", py::buffer_protocol{}};
    py::class_<Matrix3x3> matrix3x3{root, "Matrix3x3", "3x3 float matrix", py::buffer_protocol{}};
    py::class_<Matrix3x4> matrix3x4{root, "Matrix3x4", "3x4 float matrix", py::buffer_protocol{}};

    py::class_<Matrix4x2> matrix4x2{root, "Matrix4x2", "4x2 float matrix", py::buffer_protocol{}};
    py::class_<Matrix4x3> matrix4x3{root, "Matrix4x3", "4x3 float matrix", py::buffer_protocol{}};
    py::class_<Matrix4x4> matrix4x4{root, "Matrix4x4", "4x4 float matrix", py::buffer_protocol{}};

    /* The subclasses don't have buffer protocol enabled, as that's already
       done by the base classes. Moreover, just adding py::buffer_protocol{}
       would cause it to not find the buffer functions as we don't add them
       anywhere, thus failing with `pybind11_getbuffer(): Internal error`. The
       metaclasses are needed for supporting the magic translation attribute,
       see transformationMatrixMetaclass() in math.cpp for more information. */
    py::class_<Matrix3, Matrix3x3> matrix3{root, "Matrix3", "2D float transformation matrix", py::metaclass(reinterpret_cast<PyObject*>(metaclass))};
    py::class_<Matrix4, Matrix4x4> matrix4{root, "Matrix4", "3D float transformation matrix", py::metaclass(reinterpret_cast<PyObject*>(metaclass))};

    /* Register the double types as well, only after that register type
       conversions because they need all the types */
    mathMatrixDouble(root, metaclass);

    /* Register type conversions as soon as possible as those should have a
       priority over buffer and list constructors. These need all the types to
       be present, so can't be interwinted with the class definitions above. */
    convertible<Matrix2x2d>(matrix2x2);
    convertible<Matrix2x3d>(matrix2x3);
    convertible<Matrix2x4d>(matrix2x4);
    convertible<Matrix3x2d>(matrix3x2);
    convertible<Matrix3x3d>(matrix3x3);
    convertible<Matrix3x4d>(matrix3x4);
    convertible<Matrix4x2d>(matrix4x2);
    convertible<Matrix4x3d>(matrix4x3);
    convertible<Matrix4x4d>(matrix4x4);
    convertible<Matrix3d>(matrix3);
    convertible<Matrix4d>(matrix4);

    /* This needs to be *after* conversion constructors so the type conversion
       gets picked before the general buffer constructor (which would then
       fail). On the other hand, this needs to be before generic from-list
       constructors because buffer protocol is generally faster than
       iteration. */
    everyRectangularMatrixBuffer(matrix2x2);
    everyRectangularMatrixBuffer(matrix2x3);
    everyRectangularMatrixBuffer(matrix2x4);
    everyRectangularMatrixBuffer(matrix3x2);
    everyRectangularMatrixBuffer(matrix3x3);
    everyRectangularMatrixBuffer(matrix3x4);
    everyRectangularMatrixBuffer(matrix4x2);
    everyRectangularMatrixBuffer(matrix4x3);
    everyRectangularMatrixBuffer(matrix4x4);
    everyRectangularMatrixBuffer(matrix3);
    everyRectangularMatrixBuffer(matrix4);

    /* Now register the generic from-list constructors and everything else */
    matrices<Float>(
        matrix2x2, matrix2x3, matrix2x4,
        matrix3x2, matrix3x3, matrix3x4,
        matrix4x2, matrix4x3, matrix4x4,
        matrix3, matrix4);
}

}
