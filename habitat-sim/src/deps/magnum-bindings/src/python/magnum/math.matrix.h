#ifndef magnum_math_matrix_h
#define magnum_math_matrix_h
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
#include <Corrade/Containers/ScopeGuard.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Matrix4.h>

#include "corrade/PyBuffer.h"

#include "magnum/math.h"

namespace magnum {

/* A variant of Magnum's own DimensionTraits, but working for 2/3/4 dimensions
   instead of 1/2/3 dimensions */
template<UnsignedInt, class> struct VectorTraits;
template<class T> struct VectorTraits<2, T> { typedef Math::Vector2<T> Type; };
template<class T> struct VectorTraits<3, T> { typedef Math::Vector3<T> Type; };
template<class T> struct VectorTraits<4, T> { typedef Math::Vector4<T> Type; };

template<class U, class T> void initFromBuffer(T& out, const Py_buffer& buffer) {
    for(std::size_t i = 0; i != T::Cols; ++i)
        for(std::size_t j = 0; j != T::Rows; ++j)
            out[i][j] = static_cast<typename T::Type>(*reinterpret_cast<const U*>(static_cast<const char*>(buffer.buf) + i*buffer.strides[1] + j*buffer.strides[0]));
}

/* Called for both Matrix3x3 and Matrix3 in order to return a proper type /
   construct correctly from a numpy array, so has to be separate */
template<class T, class ...Args> void everyRectangularMatrix(py::class_<T, Args...>& c) {
    /* Matrix is implicitly convertible from a buffer, but not from tuples
       because there it isn't clear if it's column-major or row-major. */
    py::implicitly_convertible<py::buffer, T>();

    c
        .def_static("from_diagonal", [](const typename VectorTraits<T::DiagonalSize, typename T::Type>::Type& vector) {
            return T::fromDiagonal(vector);
        }, "Construct a diagonal matrix")
        .def_static("zero_init", []() {
            return T{Math::ZeroInit};
        }, "Construct a zero-filled matrix")
        .def(py::init(), "Default constructor")
        .def(py::init<typename T::Type>(), "Construct a matrix with one value for all components")

        /* Operators */
        .def(-py::self, "Negated matrix")
        .def(py::self += py::self, "Add and assign a matrix")
        .def(py::self + py::self, "Add a matrix")
        #ifdef __clang__
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
        #endif
        .def(py::self -= py::self, "Subtract and assign a matrix")
        #ifdef __clang__
        #pragma GCC diagnostic pop
        #endif
        .def(py::self - py::self, "Subtract a matrix")
        .def(py::self *= typename T::Type{}, "Multiply with a scalar and assign")
        .def(py::self * typename T::Type{}, "Multiply with a scalar")
        .def(py::self /= typename T::Type{}, "Divide with a scalar and assign")
        .def(py::self / typename T::Type{}, "Divide with a scalar")
        .def("__mul__", [](const T& self, const typename VectorTraits<T::Cols, typename T::Type>::Type& vector) -> typename VectorTraits<T::Rows, typename T::Type>::Type {
            return self*vector;
        }, "Multiply a vector")
        .def(typename T::Type{} * py::self, "Multiply a scalar with a matrix")
        .def(typename T::Type{} / py::self, "Divide a matrix with a scalar and invert")

        /* Member functions that don't return a size-dependent type */
        .def("flipped_cols", &T::flippedCols, "Matrix with flipped cols")
        .def("flipped_rows", &T::flippedRows, "Matrix with flipped rows")
        .def("diagonal", [](const T& self) -> typename VectorTraits<T::DiagonalSize, typename T::Type>::Type {
            return self.diagonal();
        }, "Values on diagonal");
}

/* Separate because it needs to be registered after the type conversion
   constructors. Needs to be called also for subclasses. */
template<class T, class ...Args> void everyRectangularMatrixBuffer(py::class_<T, Args...>& c) {
    c
        /* Buffer protocol, needed in order to properly detect row-major
           layouts. Has to be defined *before* the from-tuple constructor so it
           gets precedence for types that implement the buffer protocol. */
        .def(py::init([](py::buffer other) {
            /* GCC 4.8 otherwise loudly complains about missing initializers */
            Py_buffer buffer{nullptr, nullptr, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
            if(PyObject_GetBuffer(other.ptr(), &buffer, PyBUF_FORMAT|PyBUF_STRIDES) != 0)
                throw py::error_already_set{};

            Containers::ScopeGuard e{&buffer, PyBuffer_Release};

            if(buffer.ndim != 2) {
                PyErr_Format(PyExc_BufferError, "expected 2 dimensions but got %i", buffer.ndim);
                throw py::error_already_set{};
            }

            if(buffer.shape[0] != T::Rows || buffer.shape[1] != T::Cols) {
                PyErr_Format(PyExc_BufferError, "expected %zux%zu elements but got %zix%zi", T::Cols, T::Rows, buffer.shape[1], buffer.shape[0]);
                throw py::error_already_set{};
            }

            T out{NoInit};

            /* Expecting just an one-letter format */
            if(buffer.format[0] == 'f' && !buffer.format[1])
                initFromBuffer<Float>(out, buffer);
            else if(buffer.format[0] == 'd' && !buffer.format[1])
                initFromBuffer<Double>(out, buffer);
            else {
                PyErr_Format(PyExc_BufferError, "expected format f or d but got %s", buffer.format);
                throw py::error_already_set{};
            }

            return out;
        }), "Construct from a buffer");
}

template<class T> bool rectangularMatrixBufferProtocol(T& self, Py_buffer& buffer, int flags) {
    /* I hate the const_casts but I assume this is to make editing easier, NOT
       to make it possible for users to stomp on these values. */
    buffer.ndim = 2;
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
        buffer.shape = const_cast<Py_ssize_t*>(MatrixShapes[matrixShapeStrideIndex<T::Cols, T::Rows>()]);
        CORRADE_INTERNAL_ASSERT(buffer.shape[0] == T::Rows);
        CORRADE_INTERNAL_ASSERT(buffer.shape[1] == T::Cols);
        if((flags & PyBUF_STRIDES) == PyBUF_STRIDES)
            buffer.strides = const_cast<Py_ssize_t*>(matrixStridesFor<typename T::Type>(matrixShapeStrideIndex<T::Cols, T::Rows>()));
    }

    return true;
}

template<class T> void rectangularMatrix(py::class_<T>& c) {
    /*
        Missing APIs:

        from(T*)
        fromVector() (would need Vector6,...Vector16 for that)
        Type
        construction by slicing or expanding differently sized matrices
        row() / setRow() (function? that's ugly. property? not sure how)
        component-wise operations (would need BoolVector6 ... BoolVector16)
        ij() (doesn't make sense in generic code as we don't have Matrix1)
    */

    c
        /* Comparison */
        .def(py::self == py::self, "Equality comparison")
        .def(py::self != py::self, "Non-equality comparison")

        /* Set / get. Need to raise IndexError in order to allow iteration:
           https://docs.python.org/3/reference/datamodel.html#object.__getitem__
           Using error_already_set is slightly faster than throwing index_error
           directly, but still much slower than not throwing at all. Waiting
           for https://github.com/pybind/pybind11/pull/1853 to get merged. */
        .def("__setitem__", [](T& self, std::size_t i, const  typename VectorTraits<T::Rows, typename T::Type>::Type& value) {
            if(i >= T::Cols) {
                PyErr_SetString(PyExc_IndexError, "");
                throw py::error_already_set{};
            }
            self[i] = value;
        }, "Set a column at given position")
        .def("__getitem__", [](const T& self, std::size_t i) ->  typename VectorTraits<T::Rows, typename T::Type>::Type {
            if(i >= T::Cols) {
                PyErr_SetString(PyExc_IndexError, "");
                throw py::error_already_set{};
            }
            return self[i];
        }, "Column at given position")
        /* Set / get for direct elements, because [a][b] = 2.5 won't work
           without involving shared pointers */
        .def("__setitem__", [](T& self, const std::pair<std::size_t, std::size_t>& i, typename T::Type value) {
            if(i.first >= T::Cols || i.second >= T::Rows) {
                PyErr_SetString(PyExc_IndexError, "");
                throw py::error_already_set{};
            }
            self[i.first][i.second] = value;
        }, "Set a value at given col/row")
        .def("__getitem__", [](const T& self, const std::pair<std::size_t, std::size_t>& i) {
            if(i.first >= T::Cols || i.second >= T::Rows) {
                PyErr_SetString(PyExc_IndexError, "");
                throw py::error_already_set{};
            }
            return self[i.first][i.second];
        }, "Value at given col/row")

        .def("__repr__", repr<T>, "Object representation");

    /* Buffer protocol, needed in order to make numpy treat the matrix
       correctly as column-major. The constructor is defined in
       everyRectangularMatrix(). */
    corrade::enableBetterBufferProtocol<T, rectangularMatrixBufferProtocol>(c);

    /* Matrix column count */
    char lenDocstring[] = "Matrix column count. Returns _.";
    lenDocstring[sizeof(lenDocstring) - 3] = '0' + T::Cols;
    c.def_static("__len__", []() { return int(T::Cols); }, lenDocstring);
}

/* Called for both Matrix3x3 and Matrix3 in order to return a proper type, so
   has to be separate */
template<class T, class ...Args> void everyMatrix(py::class_<T, Args...>& c) {
    c
        /* Constructors */
        .def_static("identity_init", [](typename T::Type value) {
            return T{Math::IdentityInit, value};
        }, "Construct an identity matrix", py::arg("value") = typename T::Type(1))

        /* Methods */
        .def("comatrix", [](const T& self) -> T {
            return self.comatrix();
        }, "Matrix of cofactors")
        .def("adjugate", [](const T& self) -> T {
            return self.adjugate();
        }, "Adjugate matrix")
        .def("inverted", &T::inverted, "Inverted matrix")
        .def("inverted_orthogonal", &T::invertedOrthogonal, "Inverted orthogonal matrix")
        .def("__matmul__", [](const T& self, const T& other) -> T {
            return self*other;
        }, "Multiply a matrix")
        .def("transposed", [](const T& self) -> T {
            return self.transposed();
        }, "Transposed matrix");
}

template<class T> void matrix(py::class_<T>& c) {
    c
        /* Member functions for square matrices only */
        .def("is_orthogonal", &T::isOrthogonal, "Whether the matrix is orthogonal")
        .def("trace", &T::trace, "Trace of the matrix")
        .def("cofactor", &T::cofactor, "Cofactor", py::arg("col"), py::arg("row"))
        .def("determinant", &T::determinant, "Determinant");
}

template<class U, class T, class ...Args> void convertible(py::class_<T, Args...>& c) {
    c.def(py::init<U>(), "Construct from different underlying type");
}

template<class T> void matrices(
    py::class_<Math::Matrix2x2<T>>& matrix2x2,
    py::class_<Math::Matrix2x3<T>>& matrix2x3,
    py::class_<Math::Matrix2x4<T>>& matrix2x4,

    py::class_<Math::Matrix3x2<T>>& matrix3x2,
    py::class_<Math::Matrix3x3<T>>& matrix3x3,
    py::class_<Math::Matrix3x4<T>>& matrix3x4,

    py::class_<Math::Matrix4x2<T>>& matrix4x2,
    py::class_<Math::Matrix4x3<T>>& matrix4x3,
    py::class_<Math::Matrix4x4<T>>& matrix4x4,

    py::class_<Math::Matrix3<T>, Math::Matrix3x3<T>>& matrix3,
    py::class_<Math::Matrix4<T>, Math::Matrix4x4<T>>& matrix4
) {
    /* Two-column matrices. Buffer constructors need to be *before* tuple
       constructors so numpy buffer protocol gets extracted correctly. */
    everyRectangularMatrix(matrix2x2);
    everyRectangularMatrix(matrix2x3);
    everyRectangularMatrix(matrix2x4);
    rectangularMatrix(matrix2x2);
    rectangularMatrix(matrix2x3);
    rectangularMatrix(matrix2x4);
    everyMatrix(matrix2x2);
    matrix(matrix2x2);
    matrix2x2
        .def(py::init<const Math::Vector2<T>&, const Math::Vector2<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector2<T>, Math::Vector2<T>>& value) {
            return Math::Matrix2x2<T>{std::get<0>(value), std::get<1>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T>,
                                          std::tuple<T, T>>& value) {
            return Math::Matrix2x2<T>{
                Math::Vector2<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value))},
                Math::Vector2<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value))}
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix2x2<T>& self, const Math::Matrix3x2<T>& other) -> Math::Matrix3x2<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix2x2<T>& self, const Math::Matrix4x2<T>& other) -> Math::Matrix4x2<T> {
            return self*other;
        }, "Multiply a matrix");
    matrix2x3
        .def(py::init<const Math::Vector3<T>&, const Math::Vector3<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector3<T>, Math::Vector3<T>>& value) {
            return Math::Matrix2x3<T>{std::get<0>(value), std::get<1>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T, T>,
                                          std::tuple<T, T, T>>& value) {
            return Math::Matrix2x3<T>{
                Math::Vector3<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value)), std::get<2>(std::get<0>(value))},
                Math::Vector3<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value)), std::get<2>(std::get<1>(value))}
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix2x3<T>& self, const Math::Matrix2x2<T>& other) -> Math::Matrix2x3<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix2x3<T>& self, const Math::Matrix3x2<T>& other) -> Math::Matrix3x3<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix2x3<T>& self, const Math::Matrix4x2<T>& other) -> Math::Matrix4x3<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("transposed", [](const Math::Matrix2x3<T>& self) -> Math::Matrix3x2<T> {
            return self.transposed();
        }, "Transposed matrix");
    matrix2x4
        .def(py::init<const Math::Vector4<T>&, const Math::Vector4<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector4<T>, Math::Vector4<T>>& value) {
            return Math::Matrix2x4<T>{std::get<0>(value), std::get<1>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>>& value) {
            return Math::Matrix2x4<T>{
                Math::Vector4<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value)), std::get<2>(std::get<0>(value)), std::get<3>(std::get<0>(value))},
                Math::Vector4<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value)), std::get<2>(std::get<1>(value)), std::get<3>(std::get<1>(value))}
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix2x4<T>& self, const Math::Matrix2x2<T>& other) -> Math::Matrix2x4<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix2x4<T>& self, const Math::Matrix3x2<T>& other) -> Math::Matrix3x4<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix2x4<T>& self, const Math::Matrix4x2<T>& other) -> Math::Matrix4x4<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("transposed", [](const Math::Matrix2x4<T>& self) -> Math::Matrix4x2<T> {
            return self.transposed();
        }, "Transposed matrix");

    /* Three-column matrices. Buffer constructors need to be *before* tuple
       constructors so numpy buffer protocol gets extracted correctly. */
    everyRectangularMatrix(matrix3x2);
    everyRectangularMatrix(matrix3x3);
    everyRectangularMatrix(matrix3x4);
    rectangularMatrix(matrix3x2);
    rectangularMatrix(matrix3x3);
    rectangularMatrix(matrix3x4);
    everyMatrix(matrix3x3);
    matrix(matrix3x3);
    matrix3x2
        .def(py::init<const Math::Vector2<T>&, const Math::Vector2<T>&, const Math::Vector2<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector2<T>, Math::Vector2<T>, Math::Vector2<T>>& value) {
            return Math::Matrix3x2<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T>,
                                          std::tuple<T, T>,
                                          std::tuple<T, T>>& value) {
            return Math::Matrix3x2<T>{
                Math::Vector2<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value))},
                Math::Vector2<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value))},
                Math::Vector2<T>{std::get<0>(std::get<2>(value)), std::get<1>(std::get<2>(value))}
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix3x2<T>& self, const Math::Matrix2x3<T>& other) -> Math::Matrix2x2<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix3x2<T>& self, const Math::Matrix3x3<T>& other) -> Math::Matrix3x2<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix3x2<T>& self, const Math::Matrix4x3<T>& other) -> Math::Matrix4x2<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("transposed", [](const Math::Matrix3x2<T>& self) -> Math::Matrix2x3<T> {
            return self.transposed();
        }, "Transposed matrix");
    matrix3x3
        .def(py::init<const Math::Vector3<T>&, const Math::Vector3<T>&, const Math::Vector3<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector3<T>, Math::Vector3<T>, Math::Vector3<T>>& value) {
            return Math::Matrix3x3<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T, T>,
                                          std::tuple<T, T, T>,
                                          std::tuple<T, T, T>>& value) {
            return Math::Matrix3x3<T>{
                Math::Vector3<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value)), std::get<2>(std::get<0>(value))},
                Math::Vector3<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value)), std::get<2>(std::get<1>(value))},
                Math::Vector3<T>{std::get<0>(std::get<2>(value)), std::get<1>(std::get<2>(value)), std::get<2>(std::get<2>(value))}
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix3x3<T>& self, const Math::Matrix2x3<T>& other) -> Math::Matrix2x3<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix3x3<T>& self, const Math::Matrix4x3<T>& other) -> Math::Matrix4x3<T> {
            return self*other;
        }, "Multiply a matrix");
    matrix3x4
        .def(py::init<const Math::Vector4<T>&, const Math::Vector4<T>&, const Math::Vector4<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector4<T>, Math::Vector4<T>, Math::Vector4<T>>& value) {
            return Math::Matrix3x4<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>>& value) {
            return Math::Matrix3x4<T>{
                Math::Vector4<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value)), std::get<2>(std::get<0>(value)), std::get<3>(std::get<0>(value))},
                Math::Vector4<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value)), std::get<2>(std::get<1>(value)), std::get<3>(std::get<1>(value))},
                Math::Vector4<T>{std::get<0>(std::get<2>(value)), std::get<1>(std::get<2>(value)), std::get<2>(std::get<2>(value)), std::get<3>(std::get<2>(value))}
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix3x4<T>& self, const Math::Matrix2x3<T>& other) -> Math::Matrix2x4<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix3x4<T>& self, const Math::Matrix3x3<T>& other) -> Math::Matrix3x4<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix3x4<T>& self, const Math::Matrix4x3<T>& other) -> Math::Matrix4x4<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("transposed", [](const Math::Matrix3x4<T>& self) -> Math::Matrix4x3<T> {
            return self.transposed();
        }, "Transposed matrix");

    /* Four-column matrices. Buffer constructors need to be *before* tuple
       constructors so numpy buffer protocol gets extracted correctly. */
    everyRectangularMatrix(matrix4x2);
    everyRectangularMatrix(matrix4x3);
    everyRectangularMatrix(matrix4x4);
    rectangularMatrix(matrix4x2);
    rectangularMatrix(matrix4x3);
    rectangularMatrix(matrix4x4);
    everyMatrix(matrix4x4);
    matrix(matrix4x4);
    matrix4x2
        .def(py::init<const Math::Vector2<T>&, const Math::Vector2<T>&, const Math::Vector2<T>&, const Math::Vector2<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector2<T>, Math::Vector2<T>, Math::Vector2<T>, Math::Vector2<T>>& value) {
            return Math::Matrix4x2<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T>,
                                          std::tuple<T, T>,
                                          std::tuple<T, T>,
                                          std::tuple<T, T>>& value) {
            return Math::Matrix4x2<T>{
                Math::Vector2<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value))},
                Math::Vector2<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value))},
                Math::Vector2<T>{std::get<0>(std::get<2>(value)), std::get<1>(std::get<2>(value))},
                Math::Vector2<T>{std::get<0>(std::get<3>(value)), std::get<1>(std::get<3>(value))}
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix4x2<T>& self, const Math::Matrix2x4<T>& other) -> Math::Matrix2x2<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix4x2<T>& self, const Math::Matrix3x4<T>& other) -> Math::Matrix3x2<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix4x2<T>& self, const Math::Matrix4x4<T>& other) -> Math::Matrix4x2<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("transposed", [](const Math::Matrix4x2<T>& self) -> Math::Matrix2x4<T> {
            return self.transposed();
        }, "Transposed matrix");
    matrix4x3
        .def(py::init<const Math::Vector3<T>&, const Math::Vector3<T>&, const Math::Vector3<T>&, const Math::Vector3<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector3<T>, Math::Vector3<T>, Math::Vector3<T>, Math::Vector3<T>>& value) {
            return Math::Matrix4x3<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T, T>,
                                          std::tuple<T, T, T>,
                                          std::tuple<T, T, T>,
                                          std::tuple<T, T, T>>& value) {
            return Math::Matrix4x3<T>{
                Math::Vector3<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value)), std::get<2>(std::get<0>(value))},
                Math::Vector3<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value)), std::get<2>(std::get<1>(value))},
                Math::Vector3<T>{std::get<0>(std::get<2>(value)), std::get<1>(std::get<2>(value)), std::get<2>(std::get<2>(value))},
                Math::Vector3<T>{std::get<0>(std::get<3>(value)), std::get<1>(std::get<3>(value)), std::get<2>(std::get<3>(value))}
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix4x3<T>& self, const Math::Matrix2x4<T>& other) -> Math::Matrix2x3<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix4x3<T>& self, const Math::Matrix3x4<T>& other) -> Math::Matrix3x3<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix4x3<T>& self, const Math::Matrix4x4<T>& other) -> Math::Matrix4x3<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("transposed", [](const Math::Matrix4x3<T>& self) -> Math::Matrix3x4<T> {
            return self.transposed();
        }, "Transposed matrix");
    matrix4x4
        .def(py::init<const Math::Vector4<T>&, const Math::Vector4<T>&, const Math::Vector4<T>&, const Math::Vector4<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector4<T>, Math::Vector4<T>, Math::Vector4<T>, Math::Vector4<T>>& value) {
            return Math::Matrix4x4<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>>& value) {
            return Math::Matrix4x4<T>{
                Math::Vector4<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value)), std::get<2>(std::get<0>(value)), std::get<3>(std::get<0>(value))},
                Math::Vector4<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value)), std::get<2>(std::get<1>(value)), std::get<3>(std::get<1>(value))},
                Math::Vector4<T>{std::get<0>(std::get<2>(value)), std::get<1>(std::get<2>(value)), std::get<2>(std::get<2>(value)), std::get<3>(std::get<2>(value))},
                Math::Vector4<T>{std::get<0>(std::get<3>(value)), std::get<1>(std::get<3>(value)), std::get<2>(std::get<3>(value)), std::get<3>(std::get<3>(value))},
            };
        }), "Construct from a column tuple")
        .def("__matmul__", [](const Math::Matrix4x4<T>& self, const Math::Matrix2x4<T>& other) -> Math::Matrix2x4<T> {
            return self*other;
        }, "Multiply a matrix")
        .def("__matmul__", [](const Math::Matrix4x4<T>& self, const Math::Matrix3x4<T>& other) -> Math::Matrix3x4<T> {
            return self*other;
        }, "Multiply a matrix");

    /* 3x3 transformation matrix. Buffer constructors need to be *before* tuple
       constructors so numpy buffer protocol gets extracted correctly. */
    py::implicitly_convertible<Math::Matrix3x3<T>, Math::Matrix3<T>>();
    everyRectangularMatrix(matrix3);
    everyMatrix(matrix3);
    matrix3
        /* Constructors. The translation() / scaling() / rotation() are handled
           below as they conflict with member functions. */
        .def_static("reflection", &Math::Matrix3<T>::reflection,
            "2D reflection matrix")
        .def_static("shearing_x", &Math::Matrix3<T>::shearingX,
            "2D shearing matrix along the X axis", py::arg("amount"))
        .def_static("shearing_y", &Math::Matrix3<T>::shearingY,
            "2D shearning matrix along the Y axis", py::arg("amount"))
        .def_static("projection", &Math::Matrix3<T>::projection,
            "2D projection matrix", py::arg("size"))
        .def_static("from_", static_cast<Math::Matrix3<T>(*)(const Math::Matrix2x2<T>&, const Math::Vector2<T>&)>(&Math::Matrix3<T>::from),
            "Create a matrix from a rotation/scaling part and a translation part",
            py::arg("rotation_scaling"), py::arg("translation"))
        .def(py::init<const Math::Vector3<T>&, const Math::Vector3<T>&, const Math::Vector3<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector3<T>, Math::Vector3<T>, Math::Vector3<T>>& value) {
            return Math::Matrix3<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T, T>,
                                          std::tuple<T, T, T>,
                                          std::tuple<T, T, T>>& value) {
            return Math::Matrix3<T>{
                Math::Vector3<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value)), std::get<2>(std::get<0>(value))},
                Math::Vector3<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value)), std::get<2>(std::get<1>(value))},
                Math::Vector3<T>{std::get<0>(std::get<2>(value)), std::get<1>(std::get<2>(value)), std::get<2>(std::get<2>(value))}
            };
        }), "Construct from a column tuple")

        /* Member functions */
        .def("is_rigid_transformation", &Math::Matrix3<T>::isRigidTransformation,
            "Check whether the matrix represents a rigid transformation")
        .def("rotation_scaling", &Math::Matrix3<T>::rotationScaling,
            "2D rotation and scaling part of the matrix")
        .def("rotation_shear", &Math::Matrix3<T>::rotationShear,
            "2D rotation and shear part of the matrix")
        .def("rotation_normalized", &Math::Matrix3<T>::rotationNormalized,
            "2D rotation part of the matrix assuming there is no scaling")
        .def("scaling_squared", &Math::Matrix3<T>::scalingSquared,
            "Non-uniform scaling part of the matrix, squared")
        .def("uniform_scaling_squared", &Math::Matrix3<T>::uniformScalingSquared,
            "Uniform scaling part of the matrix, squared")
        .def("uniform_scaling", &Math::Matrix3<T>::uniformScaling,
            "Uniform scaling part of the matrix")
        .def("inverted_rigid", &Math::Matrix3<T>::invertedRigid,
             "Inverted rigid transformation matrix")
        .def("transform_vector", &Math::Matrix3<T>::transformVector,
            "Transform a 2D vector with the matrix")
        .def("transform_point", &Math::Matrix3<T>::transformPoint,
            "Transform a 2D point with the matrix")

        /* Properties. The translation is handled below together with a static
           translation(). */
        .def_property("right",
            static_cast<Math::Vector2<T>(Math::Matrix3<T>::*)() const>(&Math::Matrix3<T>::right),
            [](Math::Matrix3<T>& self, const Math::Vector2<T>& value) { self.right() = value; },
            "Right-pointing 2D vector")
        .def_property("up",
            static_cast<Math::Vector2<T>(Math::Matrix3<T>::*)() const>(&Math::Matrix3<T>::up),
            [](Math::Matrix3<T>& self, const Math::Vector2<T>& value) { self.up() = value; },
            "Up-pointing 2D vector");

    /* "Magic" static/member functions and properties. In order to have
       reasonable docs, we need to disable pybind's function signatures and
       supply ours faked instead. */
    {
        py::options options;
        options.disable_function_signatures();

        constexpr const char* ScalingDocstring[] {
            R"(scaling(*args, **kwargs)
Overloaded function.

1. scaling(arg0: _magnum.Vector2) -> _magnum.Matrix3

2D scaling matrix

2. scaling(self: _magnum.Matrix3) -> _magnum.Vector2

Non-uniform scaling part of the matrix
)",
            R"(scaling(*args, **kwargs)
Overloaded function.

1. scaling(arg0: _magnum.Vector2d) -> _magnum.Matrix3d

2D scaling matrix

2. scaling(self: _magnum.Matrix3d) -> _magnum.Vector2d

Non-uniform scaling part of the matrix
)"};
        constexpr const char* RotationDocstring[] {
            R"(rotation(*args, **kwargs)
Overloaded function.

1. rotation(arg0: _magnum.Rad) -> _magnum.Matrix3

2D rotation matrix

2. rotation(self: _magnum.Matrix3) -> _magnum.Matrix2x2

2D rotation part of the matrix
)",
            R"(rotation(*args, **kwargs)
Overloaded function.

1. rotation(arg0: _magnum.Rad) -> _magnum.Matrix3d

2D rotation matrix

2. rotation(self: _magnum.Matrix3d) -> _magnum.Matrix2x2d

2D rotation part of the matrix
)"};
        /* This one is special, as it renames the function */
        constexpr const char* TranslationDocstring[] {
            R"(_stranslation(*args, **kwargs)
Overloaded function.

1. translation(arg0: _magnum.Vector2) -> _magnum.Matrix3

2D translation matrix
)",
            R"(_stranslation(*args, **kwargs)
Overloaded function.

1. translation(arg0: _magnum.Vector2d) -> _magnum.Matrix3d

2D translation matrix
)"};

        matrix3
            /* Static/member scaling(). Pybind doesn't support that natively,
               so we create a scaling(*args, **kwargs) and dispatch ourselves. */
            .def_static("_sscaling", static_cast<Math::Matrix3<T>(*)(const Math::Vector2<T>&)>(&Math::Matrix3<T>::scaling))
            .def("_iscaling", static_cast<Math::Vector2<T>(Math::Matrix3<T>::*)() const>(&Math::Matrix3<T>::scaling))
            .def("scaling", [matrix3](const py::args& args, const py::kwargs& kwargs) {
                if(py::len(args) && py::isinstance<Math::Matrix3<T>>(args[0])) {
                    return matrix3.attr("_iscaling")(*args, **kwargs);
                } else {
                    return matrix3.attr("_sscaling")(*args, **kwargs);
                }
            }, ScalingDocstring[sizeof(T)/4 - 1])

            /* Static/member rotation(). Pybind doesn't support that natively,
               so we create a rotation(*args, **kwargs) and dispatch ourselves. */
            .def_static("_srotation", [](Radd angle) {
                return Math::Matrix3<T>::rotation(Math::Rad<T>(angle));
            })
            .def("_irotation", static_cast<Math::Matrix2x2<T>(Math::Matrix3<T>::*)() const>(&Math::Matrix3<T>::rotation))
            .def("rotation", [matrix3](const py::args& args, const py::kwargs& kwargs) {
                if(py::len(args) && py::isinstance<Math::Matrix3<T>>(args[0])) {
                    return matrix3.attr("_irotation")(*args, **kwargs);
                } else {
                    return matrix3.attr("_srotation")(*args, **kwargs);
                }
            }, RotationDocstring[sizeof(T)/4 - 1])

            /* Static translation function, member translation property. This
               one is tricky and can't be done without supplying a special
               metaclass that replaces static access to `translation` with
               `_stranslation`. */
            .def_static("_stranslation", static_cast<Math::Matrix3<T>(*)(const Math::Vector2<T>&)>(&Math::Matrix3<T>::translation), std::getenv("MCSS_GENERATING_OUTPUT") ? TranslationDocstring[sizeof(T)/4 - 1] : "");
    }

    /* The translation property again needs a pybind signature so we can
       extract its type */
    matrix3.def_property("translation",
        static_cast<Math::Vector2<T>(Math::Matrix3<T>::*)() const>(&Math::Matrix3<T>::translation),
        [](Math::Matrix3<T>& self, const Math::Vector2<T>& value) { self.translation() = value; },
        "2D translation part of the matrix");

    /* 4x4 transformation matrix. Buffer constructors need to be *before* tuple
       constructors so numpy buffer protocol gets extracted correctly. */
    py::implicitly_convertible<Math::Matrix4x4<T>, Math::Matrix4<T>>();
    everyRectangularMatrix(matrix4);
    everyMatrix(matrix4);
    matrix4
        /* Constructors. The translation() / scaling() / rotation() are handled
           below as they conflict with member functions. */
        .def_static("rotation_x", [](Radd angle) {
            return Math::Matrix4<T>::rotationX(Math::Rad<T>(angle));
        }, "3D rotation matrix around the X axis")
        .def_static("rotation_y", [](Radd angle) {
            return Math::Matrix4<T>::rotationY(Math::Rad<T>(angle));
        }, "3D rotation matrix around the Y axis")
        .def_static("rotation_z", [](Radd angle) {
            return Math::Matrix4<T>::rotationZ(Math::Rad<T>(angle));
        }, "3D rotation matrix around the Z axis")
        .def_static("reflection", &Math::Matrix4<T>::reflection,
            "3D reflection matrix")
        .def_static("shearing_xy", &Math::Matrix4<T>::shearingXY,
            "3D shearing matrix along the XY plane", py::arg("amount_x"), py::arg("amount_y"))
        .def_static("shearing_xz", &Math::Matrix4<T>::shearingXZ,
            "3D shearning matrix along the XZ plane", py::arg("amount_x"), py::arg("amount_z"))
        .def_static("shearing_yz", &Math::Matrix4<T>::shearingYZ,
            "3D shearing matrix along the YZ plane", py::arg("amount_y"), py::arg("amount_z"))
        .def_static("orthographic_projection", &Math::Matrix4<T>::orthographicProjection,
            "3D orthographic projection matrix", py::arg("size"), py::arg("near"), py::arg("far"))
        .def_static("perspective_projection",
            static_cast<Math::Matrix4<T>(*)(const Math::Vector2<T>&, T, T)>(&Math::Matrix4<T>::perspectiveProjection),
            "3D perspective projection matrix", py::arg("size"), py::arg("near"), py::arg("far"))
        .def_static("perspective_projection", [](Radd fov, T aspectRatio, T near, T far) {
            return Math::Matrix4<T>::perspectiveProjection(Math::Rad<T>(fov), aspectRatio, near, far);
        }, "3D perspective projection matrix", py::arg("fov"), py::arg("aspect_ratio"), py::arg("near"), py::arg("far"))
        .def_static("perspective_projection",
            static_cast<Math::Matrix4<T>(*)(const Math::Vector2<T>&, const Math::Vector2<T>&, T, T)>(&Math::Matrix4<T>::perspectiveProjection),
            "3D off-center perspective projection matrix", py::arg("bottom_left"), py::arg("top_right"), py::arg("near"), py::arg("far"))
        .def_static("look_at", &Math::Matrix4<T>::lookAt,
            "Matrix oriented towards a specific point", py::arg("eye"), py::arg("target"), py::arg("up"))
        .def_static("from_", static_cast<Math::Matrix4<T>(*)(const Math::Matrix3x3<T>&, const Math::Vector3<T>&)>(&Math::Matrix4<T>::from),
            "Create a matrix from a rotation/scaling part and a translation part",
            py::arg("rotation_scaling"), py::arg("translation"))
        .def(py::init<const Math::Vector4<T>&, const Math::Vector4<T>&, const Math::Vector4<T>&, const Math::Vector4<T>&>(),
            "Construct from column vectors")
        .def(py::init([](const std::tuple<Math::Vector4<T>, Math::Vector4<T>, Math::Vector4<T>, Math::Vector4<T>>& value) {
            return Math::Matrix4<T>{std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value)};
        }), "Construct from a column vector tuple")
        .def(py::init([](const std::tuple<std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>,
                                          std::tuple<T, T, T, T>>& value) {
            return Math::Matrix4<T>{
                Math::Vector4<T>{std::get<0>(std::get<0>(value)), std::get<1>(std::get<0>(value)), std::get<2>(std::get<0>(value)), std::get<3>(std::get<0>(value))},
                Math::Vector4<T>{std::get<0>(std::get<1>(value)), std::get<1>(std::get<1>(value)), std::get<2>(std::get<1>(value)), std::get<3>(std::get<1>(value))},
                Math::Vector4<T>{std::get<0>(std::get<2>(value)), std::get<1>(std::get<2>(value)), std::get<2>(std::get<2>(value)), std::get<3>(std::get<2>(value))},
                Math::Vector4<T>{std::get<0>(std::get<3>(value)), std::get<1>(std::get<3>(value)), std::get<2>(std::get<3>(value)), std::get<3>(std::get<3>(value))},
            };
        }), "Construct from a column tuple")

        /* Member functions */
        .def("is_rigid_transformation", &Math::Matrix4<T>::isRigidTransformation,
            "Check whether the matrix represents a rigid transformation")
        .def("rotation_scaling", &Math::Matrix4<T>::rotationScaling,
            "3D rotation and scaling part of the matrix")
        .def("rotation_shear", &Math::Matrix4<T>::rotationShear,
            "3D rotation and shear part of the matrix")
        .def("rotation_normalized", &Math::Matrix4<T>::rotationNormalized,
            "3D rotation part of the matrix assuming there is no scaling")
        .def("scaling_squared", &Math::Matrix4<T>::scalingSquared,
            "Non-uniform scaling part of the matrix, squared")
        .def("uniform_scaling_squared", &Math::Matrix4<T>::uniformScalingSquared,
            "Uniform scaling part of the matrix, squared")
        .def("uniform_scaling", &Math::Matrix4<T>::uniformScaling,
            "Uniform scaling part of the matrix")
        .def("normal_matrix", &Math::Matrix4<T>::normalMatrix,
             "Normal matrix")
        .def("inverted_rigid", &Math::Matrix4<T>::invertedRigid,
             "Inverted rigid transformation matrix")
        .def("transform_vector", &Math::Matrix4<T>::transformVector,
            "Transform a 3D vector with the matrix")
        .def("transform_point", &Math::Matrix4<T>::transformPoint,
            "Transform a 3D point with the matrix")

        /* Properties. The translation is handled below together with a static
           translation(). */
        .def_property("right",
            static_cast<Math::Vector3<T>(Math::Matrix4<T>::*)() const>(&Math::Matrix4<T>::right),
            [](Math::Matrix4<T>& self, const Math::Vector3<T>& value) { self.right() = value; },
            "Right-pointing 3D vector")
        .def_property("up",
            static_cast<Math::Vector3<T>(Math::Matrix4<T>::*)() const>(&Math::Matrix4<T>::up),
            [](Math::Matrix4<T>& self, const Math::Vector3<T>& value) { self.up() = value; },
            "Up-pointing 3D vector")
        .def_property("backward",
            static_cast<Math::Vector3<T>(Math::Matrix4<T>::*)() const>(&Math::Matrix4<T>::backward),
            [](Math::Matrix4<T>& self, const Math::Vector3<T>& value) { self.backward() = value; },
            "Backward-pointing 3D vector");

    /* "Magic" static/member functions and properties. In order to have
       reasonable docs, we need to disable pybind's function signatures and
       supply ours faked instead. */
    {
        py::options options;
        options.disable_function_signatures();

        constexpr const char* ScalingDocstring[] {
            R"(scaling(*args, **kwargs)
Overloaded function.

1. scaling(arg0: _magnum.Vector3) -> _magnum.Matrix4

3D scaling matrix

2. scaling(self: _magnum.Matrix4) -> _magnum.Vector3

Non-uniform scaling part of the matrix
)",
            R"(scaling(*args, **kwargs)
Overloaded function.

1. scaling(arg0: _magnum.Vector3d) -> _magnum.Matrix4d

2D scaling matrix

2. scaling(self: _magnum.Matrix3d) -> _magnum.Vector3d

Non-uniform scaling part of the matrix
)"
        };
        constexpr const char* RotationDocstring[] {
            R"(rotation(*args, **kwargs)
Overloaded function.

1. rotation(arg0: _magnum.Rad, arg1: _magnum.Vector3) -> _magnum.Matrix4

3D rotation matrix

2. rotation(self: _magnum.Matrix3) -> _magnum.Matrix3x3

3D rotation part of the matrix
)",
            R"(rotation(*args, **kwargs)
Overloaded function.

1. rotation(arg0: _magnum.Rad, arg1: _magnum.Vector3d) -> _magnum.Matrix4d

3D rotation matrix

2. rotation(self: _magnum.Matrix4d) -> _magnum.Matrix3x3d

3D rotation part of the matrix
)",
        };
        /* This one is special, as it renames the function */
        constexpr const char* TranslationDocstring[] {
            R"(_stranslation(*args, **kwargs)
Overloaded function.

1. translation(arg0: _magnum.Vector3) -> _magnum.Matrix4

3D translation matrix
)",
            R"(_stranslation(*args, **kwargs)
Overloaded function.

1. translation(arg0: _magnum.Vector3d) -> _magnum.Matrix4d

3D translation matrix
)"};

        matrix4
            /* Static/member scaling(). Pybind doesn't support that natively,
               so we create a scaling(*args, **kwargs) and dispatch ourselves. */
            .def_static("_sscaling", static_cast<Math::Matrix4<T>(*)(const Math::Vector3<T>&)>(&Math::Matrix4<T>::scaling))
            .def("_iscaling", static_cast<Math::Vector3<T>(Math::Matrix4<T>::*)() const>(&Math::Matrix4<T>::scaling))
            .def("scaling", [matrix4](const py::args& args, const py::kwargs& kwargs) {
                if(py::len(args) && py::isinstance<Math::Matrix4<T>>(args[0])) {
                    return matrix4.attr("_iscaling")(*args, **kwargs);
                } else {
                    return matrix4.attr("_sscaling")(*args, **kwargs);
                }
            }, ScalingDocstring[sizeof(T)/4 - 1])

            /* Static/member rotation(). Pybind doesn't support that natively,
               so we create a rotation(*args, **kwargs) and dispatch ourselves. */
            .def_static("_srotation", [](Radd angle, const Math::Vector3<T>& axis) {
                return Math::Matrix4<T>::rotation(Math::Rad<T>(angle), axis);
            })
            .def("_irotation", static_cast<Math::Matrix3x3<T>(Math::Matrix4<T>::*)() const>(&Math::Matrix4<T>::rotation))
            .def("rotation", [matrix4](const py::args& args, const py::kwargs& kwargs) {
                if(py::len(args) && py::isinstance<Math::Matrix4<T>>(args[0])) {
                    return matrix4.attr("_irotation")(*args, **kwargs);
                } else {
                    return matrix4.attr("_srotation")(*args, **kwargs);
                }
            }, RotationDocstring[sizeof(T)/4 - 1])

            /* Static translation function, member translation property. This
               one is tricky and can't be done without supplying a special
               metaclass that replaces static access to `translation` with
               `_stranslation`. */
            .def_static("_stranslation", static_cast<Math::Matrix4<T>(*)(const Math::Vector3<T>&)>(&Math::Matrix4<T>::translation), std::getenv("MCSS_GENERATING_OUTPUT") ? TranslationDocstring[sizeof(T)/4 - 1] : "");
    }

    /* The translation property again needs a pybind signature so we can
       extract its type */
    matrix4.def_property("translation",
        static_cast<Math::Vector3<T>(Math::Matrix4<T>::*)() const>(&Math::Matrix4<T>::translation),
        [](Math::Matrix4<T>& self, const Math::Vector3<T>& value) { self.translation() = value; },
        "3D translation part of the matrix");
}

}

#endif
