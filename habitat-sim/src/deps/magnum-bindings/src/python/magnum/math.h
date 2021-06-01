#ifndef magnum_math_h
#define magnum_math_h
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

#include <sstream>
#include <Python.h>
#include <Corrade/Utility/Debug.h>
#include <Magnum/Magnum.h>

#include "magnum/bootstrap.h"

namespace magnum {

/* Keep in sync with math.cpp */

extern const char* const FormatStrings[];
template<class> constexpr std::size_t formatIndex();
template<> constexpr std::size_t formatIndex<char>() { return 0; }
template<> constexpr std::size_t formatIndex<Byte>() { return 1; }
template<> constexpr std::size_t formatIndex<UnsignedByte>() { return 2; }
template<> constexpr std::size_t formatIndex<Int>() { return 3; }
template<> constexpr std::size_t formatIndex<UnsignedInt>() { return 4; }
template<> constexpr std::size_t formatIndex<Float>() { return 5; }
template<> constexpr std::size_t formatIndex<Double>() { return 6; }

extern const Py_ssize_t MatrixShapes[][2];
template<UnsignedInt cols, UnsignedInt rows> constexpr std::size_t matrixShapeStrideIndex();
template<> constexpr std::size_t matrixShapeStrideIndex<2, 2>() { return 0; }
template<> constexpr std::size_t matrixShapeStrideIndex<2, 3>() { return 1; }
template<> constexpr std::size_t matrixShapeStrideIndex<2, 4>() { return 2; }
template<> constexpr std::size_t matrixShapeStrideIndex<3, 2>() { return 3; }
template<> constexpr std::size_t matrixShapeStrideIndex<3, 3>() { return 4; }
template<> constexpr std::size_t matrixShapeStrideIndex<3, 4>() { return 5; }
template<> constexpr std::size_t matrixShapeStrideIndex<4, 2>() { return 6; }
template<> constexpr std::size_t matrixShapeStrideIndex<4, 3>() { return 7; }
template<> constexpr std::size_t matrixShapeStrideIndex<4, 4>() { return 8; }

extern const Py_ssize_t MatrixStridesFloat[][2];
extern const Py_ssize_t MatrixStridesDouble[][2];
template<class> constexpr const Py_ssize_t* matrixStridesFor(std::size_t i);
template<> constexpr const Py_ssize_t* matrixStridesFor<Float>(std::size_t i) {
    return MatrixStridesFloat[i];
}
template<> constexpr const Py_ssize_t* matrixStridesFor<Double>(std::size_t i) {
    return MatrixStridesDouble[i];
}

template<class T> std::string repr(const T& value) {
    std::ostringstream out;
    Debug{&out, Debug::Flag::NoNewlineAtTheEnd} << value;
    return out.str();
}

}

#endif
