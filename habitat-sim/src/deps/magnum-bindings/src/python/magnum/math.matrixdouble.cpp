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

void mathMatrixDouble(py::module_& root, PyTypeObject* const metaclass) {
    py::class_<Matrix2x2d> matrix2x2d{root, "Matrix2x2d", "2x2 double matrix", py::buffer_protocol{}};
    py::class_<Matrix2x3d> matrix2x3d{root, "Matrix2x3d", "2x3 double matrix", py::buffer_protocol{}};
    py::class_<Matrix2x4d> matrix2x4d{root, "Matrix2x4d", "2x4 double matrix", py::buffer_protocol{}};

    py::class_<Matrix3x2d> matrix3x2d{root, "Matrix3x2d", "3x2 double matrix", py::buffer_protocol{}};
    py::class_<Matrix3x3d> matrix3x3d{root, "Matrix3x3d", "3x3 double matrix", py::buffer_protocol{}};
    py::class_<Matrix3x4d> matrix3x4d{root, "Matrix3x4d", "3x4 double matrix", py::buffer_protocol{}};

    py::class_<Matrix4x2d> matrix4x2d{root, "Matrix4x2d", "4x2 double matrix", py::buffer_protocol{}};
    py::class_<Matrix4x3d> matrix4x3d{root, "Matrix4x3d", "4x3 double matrix", py::buffer_protocol{}};
    py::class_<Matrix4x4d> matrix4x4d{root, "Matrix4x4d", "4x4 double matrix", py::buffer_protocol{}};

    /* The subclasses don't have buffer protocol enabled, as that's already
       done by the base classes. Moreover, just adding py::buffer_protocol{}
       would cause it to not find the buffer functions as we don't add them
       anywhere, thus failing with `pybind11_getbuffer(): Internal error`. The
       metaclasses are needed for supporting the magic translation attribute,
       see transformationMatrixMetaclass() in math.cpp for more information. */
    py::class_<Matrix3d, Matrix3x3d> matrix3d{root, "Matrix3d", "2D double transformation matrix", py::metaclass(reinterpret_cast<PyObject*>(metaclass))};
    py::class_<Matrix4d, Matrix4x4d> matrix4d{root, "Matrix4d", "3D double transformation matrix", py::metaclass(reinterpret_cast<PyObject*>(metaclass))};

    /* Register type conversions as soon as possible as those should have a
       priority over buffer and list constructors. These need all the types to
       be present, so can't be interwinted with the class definitions above. */
    convertible<Matrix2x2>(matrix2x2d);
    convertible<Matrix2x3>(matrix2x3d);
    convertible<Matrix2x4>(matrix2x4d);
    convertible<Matrix3x2>(matrix3x2d);
    convertible<Matrix3x3>(matrix3x3d);
    convertible<Matrix3x4>(matrix3x4d);
    convertible<Matrix4x2>(matrix4x2d);
    convertible<Matrix4x3>(matrix4x3d);
    convertible<Matrix4x4>(matrix4x4d);
    convertible<Matrix3>(matrix3d);
    convertible<Matrix4>(matrix4d);

    /* This needs to be *after* conversion constructors so the type conversion
       gets picked before the general buffer constructor (which would then
       fail). On the other hand, this needs to be before generic from-list
       constructors because buffer protocol is generally faster than
       iteration. */
    everyRectangularMatrixBuffer(matrix2x2d);
    everyRectangularMatrixBuffer(matrix2x3d);
    everyRectangularMatrixBuffer(matrix2x4d);
    everyRectangularMatrixBuffer(matrix3x2d);
    everyRectangularMatrixBuffer(matrix3x3d);
    everyRectangularMatrixBuffer(matrix3x4d);
    everyRectangularMatrixBuffer(matrix4x2d);
    everyRectangularMatrixBuffer(matrix4x3d);
    everyRectangularMatrixBuffer(matrix4x4d);
    everyRectangularMatrixBuffer(matrix3d);
    everyRectangularMatrixBuffer(matrix4d);

    /* Now register the generic from-list constructors and everything else */
    matrices<Double>(
        matrix2x2d, matrix2x3d, matrix2x4d,
        matrix3x2d, matrix3x3d, matrix3x4d,
        matrix4x2d, matrix4x3d, matrix4x4d,
        matrix3d, matrix4d);

}

}
