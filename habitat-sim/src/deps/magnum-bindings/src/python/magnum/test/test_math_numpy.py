#
#   This file is part of Magnum.
#
#   Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
#               2020, 2021 Vladimír Vondruš <mosra@centrum.cz>
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#

import unittest

from magnum import *
from magnum import math

try:
    import numpy as np
except ModuleNotFoundError:
    raise unittest.SkipTest("numpy not installed")

class Vector(unittest.TestCase):
    def test_from_numpy(self):
        a = Vector3(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(a, Vector3(1.0, 2.0, 3.0))

    def test_to_numpy(self):
        a = np.array(Vector3(1.0, 2.0, 3.0))
        np.testing.assert_array_equal(a, np.array([1.0, 2.0, 3.0]))

    def test_from_numpy_implicit(self):
        # This works even w/o buffer protocol
        a = Vector4()
        a.xyz = np.array([1.0, 2.0, 3.0])

        b = Matrix4.translation(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(b.translation, Vector3(1.0, 2.0, 3.0))

    def test_from_numpy_implicit_typed(self):
        # But this doesn't, works only if buffer protocol is defined
        a = Vector4()
        a.xyz = np.array([1.0, 2.0, 3.0], dtype='float32')

        a = Matrix4.translation(np.array([1.0, 2.0, 3.0], dtype='float32'))
        self.assertEqual(a.translation, Vector3(1.0, 2.0, 3.0))

    def test_from_numpy_invalid_dimensions(self):
        a = np.array([[1, 2], [3, 4]])
        self.assertEqual(a.ndim, 2)

        with self.assertRaisesRegex(BufferError, "expected 1 dimension but got 2"):
            b = Vector3i(a)

    def test_from_numpy_invalid_size(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertEqual(a.shape[0], 3)

        with self.assertRaisesRegex(BufferError, "expected 2 elements but got 3"):
            b = Vector2(a)

    def test_type_from_numpy(self):
        a = Vector3i(np.array([1, 2, -3], dtype='int32'))
        self.assertEqual(a, Vector3i(1, 2, -3))

        a = Vector2ui(np.array([1, 2], dtype='uint32'))
        self.assertEqual(a, Vector2ui(1, 2))

        a = Vector4i(np.array([1, 2, -3, 0], dtype='int64'))
        self.assertEqual(a, Vector4i(1, 2, -3, 0))

        a = Vector3ui(np.array([1, 2, 3333], dtype='uint64'))
        self.assertEqual(a, Vector3i(1, 2, 3333))

        a = Vector2d(np.array([1.0, 2.0], dtype='float32'))
        self.assertEqual(a, Vector2d(1.0, 2.0))

        a = Vector2(np.array([1.0, 2.0], dtype='float64'))
        self.assertEqual(a, Vector2(1.0, 2.0))

    def test_type_from_numpy_invalid_float(self):
        a = np.array([1, 2, 3])
        self.assertEqual(a.dtype, 'int64')

        with self.assertRaisesRegex(BufferError, "unexpected format l for a f vector"):
            b = Vector3(a)

    def test_type_from_numpy_invalid_signed(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertEqual(a.dtype, 'float64')

        with self.assertRaisesRegex(BufferError, "unexpected format d for a i vector"):
            b = Vector3i(a)

class Matrix(unittest.TestCase):
    def test_from_numpy(self):
        a = Matrix2x3(np.array(
            [[1.0, 2.0],
             [4.0, 5.0],
             [7.0, 8.0]]))
        self.assertEqual(a, Matrix2x3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0)))

        a = Matrix3x3d(np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))
        self.assertEqual(a, Matrix3x3d(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))

        a = Matrix4x2(np.array(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]]))
        self.assertEqual(a, Matrix4x2(
            (1.0, 5.0),
            (2.0, 6.0),
            (3.0, 7.0),
            (4.0, 8.0)))

    def test_to_numpy(self):
        a = np.array(Matrix2x3d(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0)))
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0],
             [4.0, 5.0],
             [7.0, 8.0]]))

        a = np.array(Matrix3x3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))

        a = np.array(Matrix4x2(
            (1.0, 5.0),
            (2.0, 6.0),
            (3.0, 7.0),
            (4.0, 8.0)))
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]]))

    def test_from_numpy_invalid_dimensions(self):
        a = np.array([1, 2, 3, 4])
        self.assertEqual(a.ndim, 1)

        with self.assertRaisesRegex(BufferError, "expected 2 dimensions but got 1"):
            b = Matrix2x2(a)

    def test_from_numpy_invalid_size(self):
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        self.assertEqual(a.shape[0], 2)
        self.assertEqual(a.shape[1], 3)

        with self.assertRaisesRegex(BufferError, "expected 2x3 elements but got 3x2"):
            b = Matrix2x3(a)

    def test_order_from_numpy(self):
        a = np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]])
        self.assertEqual(a.strides[0], 24)
        self.assertEqual(a.strides[1], 8)
        self.assertEqual(Matrix3x3d(a), Matrix3x3d(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))

        a = np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]], order='C')
        self.assertEqual(a.strides[0], 24)
        self.assertEqual(a.strides[1], 8)
        self.assertEqual(Matrix3x3d(a), Matrix3x3d(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))

        a = np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]], order='F')
        self.assertEqual(a.strides[0], 8)
        self.assertEqual(a.strides[1], 24)
        self.assertEqual(Matrix3x3d(a), Matrix3x3d(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))

    def test_order_to_numpy(self):
        a = np.array(Matrix3x3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))
        self.assertEqual(a.strides[0], 4)
        self.assertEqual(a.strides[1], 12)
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))

        a = np.array(Matrix3x3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)), order='C')
        self.assertEqual(a.strides[0], 12)
        self.assertEqual(a.strides[1], 4)
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))

        a = np.array(Matrix3x3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)), order='F')
        self.assertEqual(a.strides[0], 4)
        self.assertEqual(a.strides[1], 12)
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))

    def test_type_from_numpy(self):
        a = np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]], dtype='f')
        self.assertEqual(a.dtype, 'f')
        self.assertEqual(a.itemsize, 4)
        self.assertEqual(Matrix3x3d(a), Matrix3x3d(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))

        a = np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]], dtype='d')
        self.assertEqual(a.dtype, 'd')
        self.assertEqual(a.itemsize, 8)
        self.assertEqual(Matrix3x3(a), Matrix3x3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))

    def test_type_from_numpy_invalid(self):
        a = np.array([[1, 2], [3, 4]])
        self.assertEqual(a.dtype, 'int64')

        with self.assertRaisesRegex(BufferError, "expected format f or d but got l"):
            b = Matrix2x2(a)

    def test_type_to_numpy(self):
        a = np.array(Matrix3x3d(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)), dtype='f')
        self.assertEqual(a.dtype, 'f')
        self.assertEqual(a.itemsize, 4)
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))

        a = np.array(Matrix3x3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)), dtype='d')
        self.assertEqual(a.dtype, 'd')
        self.assertEqual(a.itemsize, 8)
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))

class Matrix3_(unittest.TestCase):
    def test_from_numpy(self):
        a = Matrix3(np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))
        self.assertEqual(a, Matrix3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))

    def test_to_numpy(self):
        a = np.array(Matrix3(
            (1.0, 4.0, 7.0),
            (2.0, 5.0, 8.0),
            (3.0, 6.0, 9.0)))
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]))

class Matrix4_(unittest.TestCase):
    def test_from_numpy(self):
        a = Matrix4(np.array(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]]))
        self.assertEqual(a, Matrix4(
            (1.0, 5.0, 9.0, 13.0),
            (2.0, 6.0, 10.0, 14.0),
            (3.0, 7.0, 11.0, 15.0),
            (4.0, 8.0, 12.0, 16.0)))

    def test_to_numpy(self):
        a = np.array(Matrix4(
            (1.0, 5.0, 9.0, 13.0),
            (2.0, 6.0, 10.0, 14.0),
            (3.0, 7.0, 11.0, 15.0),
            (4.0, 8.0, 12.0, 16.0)))
        np.testing.assert_array_equal(a, np.array(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]]))
