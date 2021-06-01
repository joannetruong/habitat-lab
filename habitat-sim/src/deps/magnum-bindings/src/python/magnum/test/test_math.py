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

import array
import unittest

from magnum import *
from magnum import math

class Angle(unittest.TestCase):
    def test_init(self):
        a = Deg()
        b = Rad.zero_init()
        c = Deg(90.0)
        self.assertEqual(a, Deg(0.0))
        self.assertEqual(b, Rad(0.0))
        self.assertEqual(c, Deg(90.0))

    def test_conversion(self):
        self.assertEqual(Rad(Deg(90.0)), Rad(math.pi_half))

    def test_ops(self):
        self.assertEqual(-Deg(30.0), Deg(-30.0))
        self.assertEqual(Deg(30.0) + Deg(45.0), Deg(75.0))
        self.assertEqual(Deg(75.0) - Deg(45.0), Deg(30.0))
        self.assertEqual(Deg(45.0)*2.0, Deg(90.0))
        self.assertEqual(Deg(90.0)/2.0, Deg(45.0))
        self.assertEqual(Deg(180.0)/Deg(9.0), 20.0)

    def test_inplace_ops(self):
        a = Deg(30.0)
        a += Deg(45.0)
        self.assertEqual(a, Deg(75.0))

        a = Deg(75.0)
        a -= Deg(45.0)
        self.assertEqual(a, Deg(30.0))

        a = Deg(45.0)
        a *= 2.0
        self.assertEqual(a, Deg(90.0))

        a = Deg(90.0)
        a /= 2.0
        self.assertEqual(a, Deg(45.0))

    def test_repr(self):
        self.assertEqual(repr(Deg(45.3)), 'Deg(45.3)')

class BoolVector(unittest.TestCase):
    def test_init(self):
        a = BoolVector2()
        b = BoolVector2.zero_init()
        c = BoolVector2(0b11)
        self.assertFalse(a.any())
        self.assertTrue(c.all())
        self.assertEqual(a, BoolVector2(0b00))
        self.assertEqual(b, BoolVector2(0b00))
        self.assertEqual(c, BoolVector2(0b11))

    def test_length(self):
        self.assertEqual(BoolVector3.__len__(), 3)
        #self.assertEqual(len(BoolVector3), 3) TODO: Y not?
        self.assertEqual(len(BoolVector4()), 4)

    def test_set_get(self):
        a = BoolVector4(0b1010)
        self.assertEqual(a[1], 1)

        a[2] = True
        self.assertEqual(a, BoolVector4(0b1110))

    def test_iterate(self):
        a = [i for i in BoolVector4(0b1010)]
        self.assertEqual(a, [False, True, False, True])

    def test_ops(self):
        a = BoolVector3(0b101) | BoolVector3(0b010)
        self.assertEqual(a, BoolVector3(0b111))

    def test_repr(self):
        self.assertEqual(repr(BoolVector4(0b0101)), 'BoolVector(0b0101)')

class Constants(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(math.sqrt2**2, 2.0, 6)
        self.assertAlmostEqual(math.sqrt3**2, 3.0)

class Functions(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(math.sin(Deg(45.0)), 0.7071067811865475)
        self.assertAlmostEqual(Deg(math.asin(0.7071067811865475)), Deg(45.0))

        sincos = math.sincos(Deg(90.0))
        self.assertAlmostEqual(sincos[0], 1.0)
        self.assertAlmostEqual(sincos[1], 0.0)

class Vector(unittest.TestCase):
    def test_init(self):
        a = Vector4i()
        b = Vector3d.zero_init()
        c = Vector2i(44, -3)
        d = Vector3((1.0, 0.3, 1.1))
        e = Vector4d((1.0, 0.3, 1.1, 0.5))
        self.assertEqual(a, Vector4i(0, 0, 0, 0))
        self.assertEqual(b, Vector3d(0.0, 0.0, 0.0))
        self.assertEqual(c, Vector2i(44, -3))
        self.assertEqual(d, Vector3(1.0, 0.3, 1.1))
        self.assertEqual(e, Vector4d(1.0, 0.3, 1.1, 0.5))

    def test_convert(self):
        a = Vector2i(Vector2(4.3, 3.1))
        self.assertEqual(a, Vector2i(4, 3))

    def test_static_methods(self):
        self.assertEqual(Vector2.y_scale(5), Vector2(1, 5))
        self.assertEqual(Vector3d.z_axis(-3), Vector3d(0, 0, -3))
        self.assertEqual(Vector3i.x_axis(), Vector3i(1, 0, 0))

    def test_length(self):
        self.assertEqual(Vector3.__len__(), 3)
        #self.assertEqual(len(Vector3), 3) TODO: Y not?
        self.assertEqual(len(Vector4i()), 4)

    def test_properties(self):
        a = Vector2i()
        a.x = 1
        a.y = 2
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)
        self.assertEqual(a, Vector2i(1, 2))

        a = Vector3()
        a.x = 1.0
        a.y = 2.0
        a.z = 3.0
        self.assertEqual(a.x, 1.0)
        self.assertEqual(a.y, 2.0)
        self.assertEqual(a.z, 3.0)
        self.assertEqual(a, Vector3(1.0, 2.0, 3.0))

        a.xy = (-1.0, -2.0)
        self.assertEqual(a.xy, Vector2(-1.0, -2.0))
        self.assertEqual(a, Vector3(-1.0, -2.0, 3.0))

        a = Vector3()
        a.r = 1.0
        a.g = 2.0
        a.b = 3.0
        self.assertEqual(a.r, 1.0)
        self.assertEqual(a.g, 2.0)
        self.assertEqual(a.b, 3.0)
        self.assertEqual(a, Vector3(1.0, 2.0, 3.0))

        a = Vector4d()
        a.x = 1.0
        a.y = 2.0
        a.z = 3.0
        a.w = 4.0
        self.assertEqual(a.x, 1.0)
        self.assertEqual(a.y, 2.0)
        self.assertEqual(a.z, 3.0)
        self.assertEqual(a.w, 4.0)
        self.assertEqual(a, Vector4d(1.0, 2.0, 3.0, 4.0))

        a = Vector4d()
        a.r = 1.0
        a.g = 2.0
        a.b = 3.0
        a.a = 4.0
        self.assertEqual(a.r, 1.0)
        self.assertEqual(a.g, 2.0)
        self.assertEqual(a.b, 3.0)
        self.assertEqual(a.a, 4.0)
        self.assertEqual(a, Vector4d(1.0, 2.0, 3.0, 4.0))

        a.xy = (-1.0, -2.0)
        self.assertEqual(a.xy, Vector2d(-1.0, -2.0))
        self.assertEqual(a, Vector4d(-1.0, -2.0, 3.0, 4.0))

        a.xyz = (0.5, 0.25, 0.125)
        self.assertEqual(a.xyz, Vector3d(0.5, 0.25, 0.125))
        self.assertEqual(a, Vector4d(0.5, 0.25, 0.125, 4.0))

    def test_properties_rgb(self):
        a = Vector3()
        a.r = 1.0
        a.g = 2.0
        a.b = 3.0
        self.assertEqual(a.r, 1.0)
        self.assertEqual(a.g, 2.0)
        self.assertEqual(a.b, 3.0)
        self.assertEqual(a, Vector3(1.0, 2.0, 3.0))

        a = Vector4d()
        a.r = 1.0
        a.g = 2.0
        a.b = 3.0
        a.a = 4.0
        self.assertEqual(a.r, 1.0)
        self.assertEqual(a.g, 2.0)
        self.assertEqual(a.b, 3.0)
        self.assertEqual(a.a, 4.0)
        self.assertEqual(a, Vector4d(1.0, 2.0, 3.0, 4.0))

        a.rgb = (0.5, 0.25, 0.125)
        self.assertEqual(a.rgb, Vector3d(0.5, 0.25, 0.125))
        self.assertEqual(a, Vector4d(0.5, 0.25, 0.125, 4.0))

    def test_set_get(self):
        a = Vector3(1.0, 3.14, -13.37)
        self.assertAlmostEqual(a[1], 3.14, 6)

        a[2] = 0.13
        self.assertEqual(a, Vector3(1.0, 3.14, 0.13))

        b = Vector4i(3, 4, 5, 6)
        b.b *= 3
        b.xy += (1, -1)
        self.assertEqual(b, Vector4i(4, 3, 15, 6))

        # OOB access should fire, as that's needed for iteration
        with self.assertRaises(IndexError):
            a[5]
        with self.assertRaises(IndexError):
            a[3] = 1.1

    def test_iterate(self):
        a = [i for i in Vector4(1.0, 3.25, 3.5, -1.125)]
        # assertAlmostEqual doesn't work on lists so i'm using values directly
        # representable as floats to avoid the need for delta comparison
        self.assertEqual(a, [1.0, 3.25, 3.5, -1.125])

    def test_ops(self):
        self.assertEqual(math.dot(Vector2(0.5, 3.0), Vector2(2.0, 0.5)), 2.5)
        self.assertEqual(Deg(math.angle(
            Vector2(0.5, 3.0).normalized(),
            Vector2(2.0, 0.5).normalized())), Deg(66.5014333443446))
        self.assertEqual(Vector3(1.0, 2.0, 0.3).projected(Vector3.y_axis()),
                         Vector3.y_axis(2.0))
        self.assertEqual(Vector3(1.0, 2.0, 0.3).projected_onto_normalized(Vector3.y_axis()),
                         Vector3.y_axis(2.0))

    def test_ops_number_on_the_left(self):
        self.assertEqual(2.0*Vector2(1.0, -3.0), Vector2(2.0, -6.0))
        self.assertEqual(6.0/Vector2(2.0, -3.0), Vector2(3.0, -2.0))

    def test_swizzle(self):
        self.assertEqual(Vector3(3.0, 1.5, 0.4).yzxz, Vector4(1.5, 0.4, 3.0, 0.4))
        self.assertEqual(Vector3(3.0, 1.5, 0.4).gbrb, Vector4(1.5, 0.4, 3.0, 0.4))
        self.assertEqual(Vector4(3.0, 1.5, 0.4, 1.2).wyx, Vector3(1.2, 1.5, 3.0))
        self.assertEqual(Vector4(3.0, 1.5, 0.4, 1.2).agr, Vector3(1.2, 1.5, 3.0))
        self.assertEqual(Vector2(3.0, 1.5).yx, Vector2(1.5, 3.0))
        self.assertEqual(Vector2(3.0, 1.5).gr, Vector2(1.5, 3.0))

        with self.assertRaisesRegex(AttributeError, "only four-component swizzles are supported at most"):
            Vector4().xyzwx
        with self.assertRaisesRegex(AttributeError, "invalid swizzle"):
            Vector3().xyzw
        with self.assertRaisesRegex(AttributeError, "invalid swizzle"):
            Vector2().xyz
        with self.assertRaisesRegex(AttributeError, "invalid swizzle"):
            Vector4().c

    def test_swizzle_set(self):
        a1 = Vector3(3.0, 1.5, 0.4)
        a2 = Vector3(3.0, 1.5, 0.4)
        a1.zy = Vector2(0.5, 1.3)
        a2.bg = Vector2(0.5, 1.3)
        self.assertEqual(a1, Vector3(3.0, 1.3, 0.5))
        self.assertEqual(a1, Vector3(3.0, 1.3, 0.5))

        b1 = Vector4(3.0, 1.5, 0.4, 1.2)
        b2 = Vector4(3.0, 1.5, 0.4, 1.2)
        b1.wxz = Vector3(1.1, 0.0, -1.3)
        b2.arb = Vector3(1.1, 0.0, -1.3)
        self.assertEqual(b1, Vector4(0.0, 1.5, -1.3, 1.1))
        self.assertEqual(b2, Vector4(0.0, 1.5, -1.3, 1.1))

        # Not sure if this should be supported, but also why not
        c = Vector2(1.1, 0.4)
        c.xyyx = Vector4(0.1, 0.2, 0.3, 0.7)
        self.assertEqual(c, Vector2(0.7, 0.3))

        # Passing derived types should work too
        d = Vector4(3.0, 1.5, 0.4, 1.2)
        d.wxz = Color3(1.1, 0.0, -1.3)
        self.assertEqual(d, Vector4(0.0, 1.5, -1.3, 1.1))

        # Handled by pybind / python as a fallback directly
        with self.assertRaises(AttributeError):
            Vector4().xc = Vector2()
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            Vector4().xyz = 3
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            Vector4().xy = 3

        # Handled by the swizzle implementation
        with self.assertRaisesRegex(TypeError, "unrecognized swizzle type"):
            Vector4().xzy = 3
        with self.assertRaisesRegex(TypeError, "swizzle doesn't match passed vector component count"):
            Vector2().yx = Vector3()
        with self.assertRaisesRegex(AttributeError, "invalid swizzle"):
            Vector3().xyzw = Vector4()
        with self.assertRaisesRegex(AttributeError, "invalid swizzle"):
            Vector2().xyz = Vector3()

    def test_repr(self):
        self.assertEqual(repr(Vector3(1.0, 3.14, -13.37)), 'Vector(1, 3.14, -13.37)')

    def test_from_buffer(self):
        a = Vector3i(array.array('i', [2, 3, 5]))
        self.assertEqual(a, Vector3i(2, 3, 5))

    def test_to_buffer(self):
        a = memoryview(Vector4(1.0, 2.0, 3.0, 4.0))
        self.assertEqual(a.tolist(), [1.0, 2.0, 3.0, 4.0])

class Color3_(unittest.TestCase):
    def test_init(self):
        a1 = Color3()
        a2 = Color3.zero_init()
        self.assertEqual(a1, Color3(0.0, 0.0, 0.0))
        self.assertEqual(a2, Color3(0.0, 0.0, 0.0))

        b = Color3(0.5)
        self.assertEqual(b, Color3(0.5, 0.5, 0.5))

        c1 = Color3(0.5, 0.75, 1.0)
        c2 = Color3((0.5, 0.75, 1.0))
        c3 = Color3(Vector3(0.5, 0.75, 1.0))
        self.assertEqual(c1, Color3(0.5, 0.75, 1.0))
        self.assertEqual(c2, Color3(0.5, 0.75, 1.0))
        self.assertEqual(c3, Color3(0.5, 0.75, 1.0))

    def test_srgb(self):
        # Cross-checked with C++ tests
        a = Color3.from_srgb(0xf32a80)
        self.assertEqual(a, Color3(0.896269, 0.0231534, 0.215861))
        self.assertEqual(a.to_srgb_int(), 0xf32a80)

    def test_hsv(self):
        a = Color3.from_hsv(Deg(230.0), 0.749, 0.427)
        self.assertEqual(a, Color3(0.107177, 0.160481, 0.427))

        self.assertEqual(a.hue(), Deg(230.0))
        self.assertAlmostEqual(a.saturation(), 0.749)
        self.assertAlmostEqual(a.value(), 0.427)

        self.assertEqual(a.to_hsv()[0], Deg(230.0))
        self.assertAlmostEqual(a.to_hsv()[1], 0.749)
        self.assertAlmostEqual(a.to_hsv()[2], 0.427)

    def test_methods_return_type(self):
        self.assertIsInstance(Color3()*1.5, Color3)
        self.assertIsInstance(Color3()+Color3(), Color3)
        self.assertIsInstance(Color3.zero_init(), Color3)

class Color4_(unittest.TestCase):
    def test_init(self):
        a1 = Color4()
        a2 = Color4.zero_init()
        self.assertEqual(a1, Color4(0.0, 0.0, 0.0, 0.0))
        self.assertEqual(a2, Color4(0.0, 0.0, 0.0, 0.0))

        b = Color4(0.5)
        self.assertEqual(b, Color4(0.5, 0.5, 0.5, 1.0))

        c = Color4(0.5, 0.75)
        self.assertEqual(c, Color4(0.5, 0.5, 0.5, 0.75))

        d1 = Color4(0.5, 0.75, 0.875)
        d2 = Color4((0.5, 0.75, 0.875))
        d3 = Color4(Vector3(0.5, 0.75, 0.875))
        self.assertEqual(d1, Color4(0.5, 0.75, 0.875, 1.0))
        self.assertEqual(d2, Color4(0.5, 0.75, 0.875, 1.0))
        self.assertEqual(d3, Color4(0.5, 0.75, 0.875, 1.0))

        e1 = Color4(0.5, 0.75, 0.875, 0.9)
        e2 = Color4((0.5, 0.75, 0.875), 0.9)
        e3 = Color4((0.5, 0.75, 0.875, 0.9))
        e4 = Color4(Vector3(0.5, 0.75, 0.875), 0.9)
        e5 = Color4(Vector4(0.5, 0.75, 0.875, 0.9))
        self.assertEqual(e1, Color4(0.5, 0.75, 0.875, 0.9))
        self.assertEqual(e2, Color4(0.5, 0.75, 0.875, 0.9))
        self.assertEqual(e3, Color4(0.5, 0.75, 0.875, 0.9))
        self.assertEqual(e4, Color4(0.5, 0.75, 0.875, 0.9))
        self.assertEqual(e5, Color4(0.5, 0.75, 0.875, 0.9))

    def test_srgb(self):
        # Cross-checked with C++ tests
        a = Color4.from_srgb(0xf32a80)
        self.assertEqual(a, Color4(0.896269, 0.0231534, 0.215861, 1.0))

        self.assertEqual(a.to_srgb_alpha_int(), 0xf32a80ff)

    def test_srgb_alpha(self):
        # Cross-checked with C++ tests
        a = Color4.from_srgb(0xf32a80, a=0.137255)
        b = Color4.from_srgb_alpha(0xf32a8023)
        self.assertEqual(a, Color4(0.896269, 0.0231534, 0.215861, 0.137255))
        self.assertEqual(b, Color4(0.896269, 0.0231534, 0.215861, 0.137255))

        self.assertEqual(a.to_srgb_alpha_int(), 0xf32a8023)

    def test_hsv(self):
        a = Color4.from_hsv(Deg(230.0), 0.749, 0.427, 0.95)
        self.assertEqual(a, Color4(0.107177, 0.160481, 0.427, 0.95))

        self.assertEqual(a.hue(), Deg(230.0))
        self.assertAlmostEqual(a.saturation(), 0.749)
        self.assertAlmostEqual(a.value(), 0.427)

        self.assertEqual(a.to_hsv()[0], Deg(230.0))
        self.assertAlmostEqual(a.to_hsv()[1], 0.749)
        self.assertAlmostEqual(a.to_hsv()[2], 0.427)

        b = Color4.from_hsv(Deg(230.0), 0.749, 0.427)
        self.assertEqual(b, Color4(0.107177, 0.160481, 0.427, 1.0))

    def test_methods_return_type(self):
        self.assertIsInstance(Color4()*1.5, Color4)
        self.assertIsInstance(Color4()+Color4(), Color4)
        self.assertIsInstance(Color4.zero_init(), Color4)
        self.assertIsInstance(Color4().rgb, Color3)
        self.assertIsInstance(Color4().xyz, Color3)

    def test_from_buffer(self):
        a = Color3(array.array('f', [2.0, 3.0, 5.0]))
        self.assertEqual(a, Color3(2.0, 3.0, 5.0))

    def test_to_buffer(self):
        # Color4 doesn't define py::buffer_protocol(), the one from base should
        # "just work"
        a = memoryview(Color4(1.0, 2.0, 3.0, 4.0))
        self.assertEqual(a.tolist(), [1.0, 2.0, 3.0, 4.0])

class Matrix(unittest.TestCase):
    def test_init(self):
        a = Matrix3x2()
        self.assertEqual(a[0], Vector2(0))
        self.assertEqual(a[1], Vector2(0))
        self.assertEqual(a[2], Vector2(0))

        b = Matrix4x4.zero_init()
        self.assertEqual(b[0], Vector4(0))
        self.assertEqual(b[1], Vector4(0))
        self.assertEqual(b[2], Vector4(0))
        self.assertEqual(b[3], Vector4(0))

        c1 = Matrix3x3.identity_init()
        self.assertEqual(c1[0], Vector3.x_axis())
        self.assertEqual(c1[1], Vector3.y_axis())
        self.assertEqual(c1[2], Vector3.z_axis())

        c3 = Matrix3x3.identity_init(3.0)
        self.assertEqual(c3[0], Vector3.x_axis(3.0))
        self.assertEqual(c3[1], Vector3.y_axis(3.0))
        self.assertEqual(c3[2], Vector3.z_axis(3.0))

        d = Matrix2x3(Vector3(1.0, 2.0, 3.0),
                      Vector3(4.0, 5.0, 6.0))
        self.assertEqual(d[0], Vector3(1.0, 2.0, 3.0))
        self.assertEqual(d[1], Vector3(4.0, 5.0, 6.0))

        e = Matrix2x3((1.0, 2.0, 3.0),
                      (4.0, 5.0, 6.0))
        self.assertEqual(e[0], Vector3(1.0, 2.0, 3.0))
        self.assertEqual(e[1], Vector3(4.0, 5.0, 6.0))

    def test_init_tuple_of_vectors(self):
        a = Matrix2x2((Vector2(1.0, 2.0),
                       Vector2(3.0, 4.0)))
        self.assertEqual(a, Matrix2x2(Vector2(1.0, 2.0),
                                      Vector2(3.0, 4.0)))

        a = Matrix2x3((Vector3(1.0, 2.0, 3.0),
                       Vector3(4.0, 5.0, 6.0)))
        self.assertEqual(a, Matrix2x3(Vector3(1.0, 2.0, 3.0),
                                      Vector3(4.0, 5.0, 6.0)))

        a = Matrix2x4((Vector4(1.0, 2.0, 3.0, 4.0),
                       Vector4(5.0, 6.0, 7.0, 8.0)))
        self.assertEqual(a, Matrix2x4(Vector4(1.0, 2.0, 3.0, 4.0),
                                      Vector4(5.0, 6.0, 7.0, 8.0)))

        a = Matrix3x2((Vector2(1.0, 2.0),
                       Vector2(3.0, 4.0),
                       Vector2(5.0, 6.0)))
        self.assertEqual(a, Matrix3x2(Vector2(1.0, 2.0),
                                      Vector2(3.0, 4.0),
                                      Vector2(5.0, 6.0)))

        a = Matrix3x3((Vector3(1.0, 2.0, 3.0),
                       Vector3(4.0, 5.0, 6.0),
                       Vector3(7.0, 8.0, 9.0)))
        self.assertEqual(a, Matrix3x3(Vector3(1.0, 2.0, 3.0),
                                      Vector3(4.0, 5.0, 6.0),
                                      Vector3(7.0, 8.0, 9.0)))

        a = Matrix3x4((Vector4(1.0, 2.0, 3.0, 4.0),
                       Vector4(5.0, 6.0, 7.0, 8.0),
                       Vector4(9.0, 10.0, 11.0, 12.0)))
        self.assertEqual(a, Matrix3x4(Vector4(1.0, 2.0, 3.0, 4.0),
                                      Vector4(5.0, 6.0, 7.0, 8.0),
                                      Vector4(9.0, 10.0, 11.0, 12.0)))

        a = Matrix4x2((Vector2(1.0, 2.0),
                       Vector2(3.0, 4.0),
                       Vector2(5.0, 6.0),
                       Vector2(7.0, 8.0)))
        self.assertEqual(a, Matrix4x2(Vector2(1.0, 2.0),
                                      Vector2(3.0, 4.0),
                                      Vector2(5.0, 6.0),
                                      Vector2(7.0, 8.0)))

        a = Matrix4x3((Vector3(1.0, 2.0, 3.0),
                       Vector3(4.0, 5.0, 6.0),
                       Vector3(7.0, 8.0, 9.0),
                       Vector3(10.0, 11.0, 12.0)))
        self.assertEqual(a, Matrix4x3(Vector3(1.0, 2.0, 3.0),
                                      Vector3(4.0, 5.0, 6.0),
                                      Vector3(7.0, 8.0, 9.0),
                                      Vector3(10.0, 11.0, 12.0)))

        a = Matrix4x4((Vector4(1.0, 2.0, 3.0, 4.0),
                       Vector4(5.0, 6.0, 7.0, 8.0),
                       Vector4(9.0, 10.0, 11.0, 12.0),
                       Vector4(13.0, 14.0, 15.0, 16.0)))
        self.assertEqual(a, Matrix4x4(Vector4(1.0, 2.0, 3.0, 4.0),
                                      Vector4(5.0, 6.0, 7.0, 8.0),
                                      Vector4(9.0, 10.0, 11.0, 12.0),
                                      Vector4(13.0, 14.0, 15.0, 16.0)))

    def test_init_tuple_of_tuples(self):
        a = Matrix2x2(((1.0, 2.0),
                       (3.0, 4.0)))
        self.assertEqual(a, Matrix2x2(Vector2(1.0, 2.0),
                                      Vector2(3.0, 4.0)))

        a = Matrix2x3(((1.0, 2.0, 3.0),
                       (4.0, 5.0, 6.0)))
        self.assertEqual(a, Matrix2x3(Vector3(1.0, 2.0, 3.0),
                                      Vector3(4.0, 5.0, 6.0)))

        a = Matrix2x4(((1.0, 2.0, 3.0, 4.0),
                       (5.0, 6.0, 7.0, 8.0)))
        self.assertEqual(a, Matrix2x4(Vector4(1.0, 2.0, 3.0, 4.0),
                                      Vector4(5.0, 6.0, 7.0, 8.0)))

        a = Matrix3x2(((1.0, 2.0),
                       (3.0, 4.0),
                       (5.0, 6.0)))
        self.assertEqual(a, Matrix3x2(Vector2(1.0, 2.0),
                                      Vector2(3.0, 4.0),
                                      Vector2(5.0, 6.0)))

        a = Matrix3x3(((1.0, 2.0, 3.0),
                       (4.0, 5.0, 6.0),
                       (7.0, 8.0, 9.0)))
        self.assertEqual(a, Matrix3x3(Vector3(1.0, 2.0, 3.0),
                                      Vector3(4.0, 5.0, 6.0),
                                      Vector3(7.0, 8.0, 9.0)))

        a = Matrix3x4(((1.0, 2.0, 3.0, 4.0),
                       (5.0, 6.0, 7.0, 8.0),
                       (9.0, 10.0, 11.0, 12.0)))
        self.assertEqual(a, Matrix3x4(Vector4(1.0, 2.0, 3.0, 4.0),
                                      Vector4(5.0, 6.0, 7.0, 8.0),
                                      Vector4(9.0, 10.0, 11.0, 12.0)))

        a = Matrix4x2(((1.0, 2.0),
                       (3.0, 4.0),
                       (5.0, 6.0),
                       (7.0, 8.0)))
        self.assertEqual(a, Matrix4x2(Vector2(1.0, 2.0),
                                      Vector2(3.0, 4.0),
                                      Vector2(5.0, 6.0),
                                      Vector2(7.0, 8.0)))

        a = Matrix4x3(((1.0, 2.0, 3.0),
                       (4.0, 5.0, 6.0),
                       (7.0, 8.0, 9.0),
                       (10.0, 11.0, 12.0)))
        self.assertEqual(a, Matrix4x3(Vector3(1.0, 2.0, 3.0),
                                      Vector3(4.0, 5.0, 6.0),
                                      Vector3(7.0, 8.0, 9.0),
                                      Vector3(10.0, 11.0, 12.0)))

        a = Matrix4x4(((1.0, 2.0, 3.0, 4.0),
                       (5.0, 6.0, 7.0, 8.0),
                       (9.0, 10.0, 11.0, 12.0),
                       (13.0, 14.0, 15.0, 16.0)))
        self.assertEqual(a, Matrix4x4(Vector4(1.0, 2.0, 3.0, 4.0),
                                      Vector4(5.0, 6.0, 7.0, 8.0),
                                      Vector4(9.0, 10.0, 11.0, 12.0),
                                      Vector4(13.0, 14.0, 15.0, 16.0)))

    def test_static_methods(self):
        a = Matrix3.from_(Matrix2x2((1.0, 2.0),
                                    (4.0, 5.0)),
                          Vector2(7.0, 8.0))
        self.assertEqual(a, Matrix3(Vector3(1.0, 2.0, 0.0),
                                    Vector3(4.0, 5.0, 0.0),
                                    Vector3(7.0, 8.0, 1.0)))

        a = Matrix4.from_(Matrix3x3((1.0, 2.0, 3.0),
                                    (5.0, 6.0, 7.0),
                                    (9.0, 10.0, 11.0)),
                          Vector3(13.0, 14.0, 15.0))
        self.assertEqual(a, Matrix4x4(Vector4(1.0, 2.0, 3.0, 0.0),
                                      Vector4(5.0, 6.0, 7.0, 0.0),
                                      Vector4(9.0, 10.0, 11.0, 0.0),
                                      Vector4(13.0, 14.0, 15.0, 1.0)))

        a = Matrix3x4.from_diagonal((1.0, 2.0, 3.0))
        self.assertEqual(a.diagonal(), (1.0, 2.0, 3.0))
        self.assertEqual(a, Matrix3x4((1.0, 0.0, 0.0, 0.0),
                                      (0.0, 2.0, 0.0, 0.0),
                                      (0.0, 0.0, 3.0, 0.0)))

    def test_convert(self):
        a = Matrix2x3d(Matrix2x3((1.0, 2.0, 3.0),
                                 (4.0, 5.0, 6.0)))
        self.assertEqual(a, Matrix2x3d((1.0, 2.0, 3.0),
                                       (4.0, 5.0, 6.0)))

    def test_length(self):
        self.assertEqual(Matrix3x4.__len__(), 3)
        #self.assertEqual(len(Matrix4x3), 4) TODO: Y not?
        self.assertEqual(len(Matrix4x3()), 4)

    def test_set_get(self):
        a = Matrix2x3((1.0, 2.0, 3.0),
                      (4.0, 5.0, 6.0))
        self.assertEqual(a[1][2], 6.0)
        self.assertEqual(a[1, 2], 6.0)

        a[1] = (4.5, 5.5, 6.5)
        self.assertEqual(a[1], Vector3(4.5, 5.5, 6.5))

        a[0, 1] = 2.5
        self.assertEqual(a[0], Vector3(1.0, 2.5, 3.0)) # yes, 2.5

        # OOB access should fire, as that's needed for iteration
        with self.assertRaises(IndexError):
            a[2]
        with self.assertRaises(IndexError):
            a[1, 3]
        with self.assertRaises(IndexError):
            a[2] = Vector3()
        with self.assertRaises(IndexError):
            a[2, 1] = 0.5

    @unittest.expectedFailure
    def test_set_two_brackets(self):
        a = Matrix2x3((1.0, 2.0, 3.0),
                      (4.0, 5.0, 6.0))
        a[0][1] = 2.5
        self.assertEqual(a[0], Vector3(1.0, 2.5, 3.0))

    def test_iterate(self):
        a = Matrix3x2((1.0, 2.0),
                      (3.0, 4.0),
                      (5.0, 6.0))
        self.assertEqual([i.sum() for i in a], [3.0, 7.0, 11.0])

    def test_ops(self):
        a = Matrix2x3((1.0, 2.0, 3.0),
                      (4.0, 5.0, 6.0))
        b = Matrix3x2((1.0, 2.0),
                      (3.0, 4.0),
                      (5.0, 6.0))
        c = Matrix3x3((9.0, 12.0, 15.0),
                      (19.0, 26.0, 33.0),
                      (29.0, 40.0, 51.0))
        self.assertEqual(a@b, c)
        self.assertEqual(a*Vector2(0.5, 0.25), Vector3(1.5, 2.25, 3.0))

    def test_ops_number_on_the_left(self):
        a = Matrix2x3((1.0, 2.0, 3.0),
                      (4.0, 5.0, 6.0))

        self.assertEqual(2.0*a, Matrix2x3((2.0, 4.0, 6.0),
                                          (8.0, 10.0, 12.0)))
        self.assertEqual(6.0/a, Matrix2x3((6.0, 3.0, 2.0),
                                          (1.5, 1.2, 1.0)))

    def test_repr(self):
        a = Matrix2x3((1.0, 2.0, 3.0),
                      (4.0, 5.0, 6.0))
        self.assertEqual(repr(a), 'Matrix(1, 4,\n'
                                  '       2, 5,\n'
                                  '       3, 6)')

    # conversion from buffer is tested in test_math_numpy, array.array is
    # one-dimensional and I don't want to drag numpy here just for one test

    def test_to_buffer(self):
        a = memoryview(Matrix2x2((1.0, 2.0), (3.0, 4.0)))
        self.assertEqual(a.tolist(), [[1.0, 3.0], [2.0, 4.0]])

class Matrix3_(unittest.TestCase):
    def test_init(self):
        a = Matrix3()
        self.assertEqual(a[0], Vector3.x_axis())
        self.assertEqual(a[1], Vector3.y_axis())
        self.assertEqual(a[2], Vector3.z_axis())

        b = Matrix3.zero_init()
        self.assertEqual(b[0], Vector3(0))
        self.assertEqual(b[1], Vector3(0))
        self.assertEqual(b[2], Vector3(0))

        c1 = Matrix3.identity_init()
        self.assertEqual(c1[0], Vector3.x_axis())
        self.assertEqual(c1[1], Vector3.y_axis())
        self.assertEqual(c1[2], Vector3.z_axis())

        c3 = Matrix3.identity_init(3.0)
        self.assertEqual(c3[0], Vector3.x_axis(3.0))
        self.assertEqual(c3[1], Vector3.y_axis(3.0))
        self.assertEqual(c3[2], Vector3.z_axis(3.0))

        d = Matrix3((1.0, 2.0, 3.0),
                    (4.0, 5.0, 6.0),
                    (7.0, 8.0, 9.0))
        self.assertEqual(d[0], Vector3(1.0, 2.0, 3.0))
        self.assertEqual(d[1], Vector3(4.0, 5.0, 6.0))
        self.assertEqual(d[2], Vector3(7.0, 8.0, 9.0))

        # Tuple of vectors
        e = Matrix3((Vector3(1.0, 2.0, 3.0),
                     Vector3(4.0, 5.0, 6.0),
                     Vector3(7.0, 8.0, 9.0)))
        self.assertEqual(e, Matrix3(Vector3(1.0, 2.0, 3.0),
                                    Vector3(4.0, 5.0, 6.0),
                                    Vector3(7.0, 8.0, 9.0)))

        # Tuple of tuples
        e = Matrix3(((1.0, 2.0, 3.0),
                     (4.0, 5.0, 6.0),
                     (7.0, 8.0, 9.0)))
        self.assertEqual(e, Matrix3(Vector3(1.0, 2.0, 3.0),
                                    Vector3(4.0, 5.0, 6.0),
                                    Vector3(7.0, 8.0, 9.0)))

    def test_convert(self):
        a = Matrix3(Matrix3d((1.0, 2.0, 3.0),
                             (4.0, 5.0, 6.0),
                             (7.0, 8.0, 9.0)))
        self.assertEqual(a, Matrix3((1.0, 2.0, 3.0),
                                    (4.0, 5.0, 6.0),
                                    (7.0, 8.0, 9.0)))

    def test_static_methods(self):
        a = Matrix3.translation((0.0, -1.0))
        self.assertEqual(a[2].xy, Vector2(0.0, -1.0))
        self.assertEqual(a.translation, Vector2(0.0, -1.0))

        b = Matrix3.rotation(Deg(45.0))
        self.assertEqual(b.rotation(), Matrix2x2(
            (0.707107, 0.707107),
            (-0.707107, 0.707107)))

        c = Matrix3.scaling((1.0, 2.0))
        self.assertEqual(c.scaling(), Vector2(1.0, 2.0))

    def test_properties(self):
        a = Matrix3.translation(Vector2.y_axis(-5.0))@Matrix3.rotation(Deg(45.0))
        self.assertEqual(a.right, Vector2(0.707107, 0.707107))
        self.assertEqual(a.up, Vector2(-0.707107, 0.707107))
        self.assertEqual(a.translation, Vector2.y_axis(-5.0))

        a.right = Vector2.x_axis(2.0)
        a.up = -Vector2.y_axis()
        a.translation = Vector2(0.0)
        self.assertEqual(a, Matrix3.from_diagonal((2.0, -1.0, 1.0)))

    def test_methods(self):
        self.assertEqual(Matrix3.rotation(Deg(45.0)).transposed(),
                         Matrix3.rotation(Deg(-45.0)))
        self.assertEqual(Matrix3.scaling(Vector2(3.0)).inverted(),
                         Matrix3.scaling(Vector2(1/3.0)))

    def test_methods_return_type(self):
        self.assertIsInstance(Matrix3.zero_init(), Matrix3)
        self.assertIsInstance(Matrix3.from_diagonal((3.0, 1.0, 1.0)), Matrix3)
        self.assertIsInstance(Matrix3()@Matrix3(), Matrix3)
        self.assertIsInstance(Matrix3()+Matrix3(), Matrix3)
        self.assertIsInstance(Matrix3().transposed(), Matrix3)
        self.assertIsInstance(Matrix3().inverted(), Matrix3)

    # conversion from buffer is tested in test_math_numpy, array.array is
    # one-dimensional and I don't want to drag numpy here just for one test

    def test_to_buffer(self):
        # Matrix3 doesn't define py::buffer_protocol(), the one from base
        # should "just work"
        a = memoryview(Matrix3((1.0, 2.0, 3.0),
                               (4.0, 5.0, 6.0),
                               (7.0, 8.0, 9.0)))
        self.assertEqual(a.tolist(), [
            [1.0, 4.0, 7.0],
            [2.0, 5.0, 8.0],
            [3.0, 6.0, 9.0]])

class Matrix4_(unittest.TestCase):
    def test_init(self):
        a = Matrix4()
        self.assertEqual(a[0], Vector4(1.0, 0.0, 0.0, 0.0))
        self.assertEqual(a[1], Vector4(0.0, 1.0, 0.0, 0.0))
        self.assertEqual(a[2], Vector4(0.0, 0.0, 1.0, 0.0))
        self.assertEqual(a[3], Vector4(0.0, 0.0, 0.0, 1.0))

        b = Matrix4.zero_init()
        self.assertEqual(b[0], Vector4(0))
        self.assertEqual(b[1], Vector4(0))
        self.assertEqual(b[2], Vector4(0))

        c1 = Matrix4.identity_init()
        self.assertEqual(c1[0], Vector4(1.0, 0.0, 0.0, 0.0))
        self.assertEqual(c1[1], Vector4(0.0, 1.0, 0.0, 0.0))
        self.assertEqual(c1[2], Vector4(0.0, 0.0, 1.0, 0.0))
        self.assertEqual(c1[3], Vector4(0.0, 0.0, 0.0, 1.0))

        c3 = Matrix4.identity_init(3.0)
        self.assertEqual(c3[0], Vector4(3.0, 0.0, 0.0, 0.0))
        self.assertEqual(c3[1], Vector4(0.0, 3.0, 0.0, 0.0))
        self.assertEqual(c3[2], Vector4(0.0, 0.0, 3.0, 0.0))
        self.assertEqual(c3[3], Vector4(0.0, 0.0, 0.0, 3.0))

        d = Matrix4((1.0, 2.0, 3.0, 4.0),
                    (5.0, 6.0, 7.0, 8.0),
                    (9.0, 10.0, 11.0, 12.0),
                    (13.0, 14.0, 15.0, 16.0))
        self.assertEqual(d[0], Vector4(1.0, 2.0, 3.0, 4.0))
        self.assertEqual(d[1], Vector4(5.0, 6.0, 7.0, 8.0))
        self.assertEqual(d[2], Vector4(9.0, 10.0, 11.0, 12.0))
        self.assertEqual(d[3], Vector4(13.0, 14.0, 15.0, 16.0))

        # Tuple of vectors
        e = Matrix4((Vector4(1.0, 2.0, 3.0, 4.0),
                     Vector4(5.0, 6.0, 7.0, 8.0),
                     Vector4(9.0, 10.0, 11.0, 12.0),
                     Vector4(13.0, 14.0, 15.0, 16.0)))
        self.assertEqual(e, Matrix4(Vector4(1.0, 2.0, 3.0, 4.0),
                                    Vector4(5.0, 6.0, 7.0, 8.0),
                                    Vector4(9.0, 10.0, 11.0, 12.0),
                                    Vector4(13.0, 14.0, 15.0, 16.0)))

        # Tuple of tuples
        e = Matrix4(((1.0, 2.0, 3.0, 4.0),
                     (5.0, 6.0, 7.0, 8.0),
                     (9.0, 10.0, 11.0, 12.0),
                     (13.0, 14.0, 15.0, 16.0)))
        self.assertEqual(e, Matrix4(Vector4(1.0, 2.0, 3.0, 4.0),
                                    Vector4(5.0, 6.0, 7.0, 8.0),
                                    Vector4(9.0, 10.0, 11.0, 12.0),
                                    Vector4(13.0, 14.0, 15.0, 16.0)))

    def test_convert(self):
        a = Matrix4d(Matrix4((1.0, 2.0, 3.0, 4.0),
                             (5.0, 6.0, 7.0, 8.0),
                             (9.0, 10.0, 11.0, 12.0),
                             (13.0, 14.0, 15.0, 16.0)))
        self.assertEqual(a, Matrix4d((1.0, 2.0, 3.0, 4.0),
                                     (5.0, 6.0, 7.0, 8.0),
                                     (9.0, 10.0, 11.0, 12.0),
                                     (13.0, 14.0, 15.0, 16.0)))

    def test_static_methods(self):
        a = Matrix4.translation((0.0, -1.0, 2.0))
        self.assertEqual(a[3].xyz, Vector3(0.0, -1.0, 2.0))
        self.assertEqual(a.translation, Vector3(0.0, -1.0, 2.0))

        b = Matrix4.rotation(Deg(45.0), Vector3.x_axis())
        self.assertEqual(b.rotation(), Matrix3x3(
            (1.0, 0.0, 0.0),
            (0.0, 0.707107, 0.707107),
            (0.0, -0.707107, 0.707107)))

        c = Matrix4.scaling((1.0, 2.0, 3.5))
        self.assertEqual(c.scaling(), Vector3(1.0, 2.0, 3.5))

    def test_properties(self):
        a = Matrix4.translation(Vector3.y_axis(-5.0))@Matrix4.rotation_z(Deg(45.0))
        self.assertEqual(a.right, Vector3(0.707107, 0.707107, 0.0))
        self.assertEqual(a.up, Vector3(-0.707107, 0.707107, 0.0))
        self.assertEqual(a.backward, Vector3(0.0, 0.0, 1.0))
        self.assertEqual(a.translation, Vector3.y_axis(-5.0))

        a.right = Vector3.x_axis(3.0)
        a.up = -Vector3.y_axis()
        a.backward = Vector3.z_axis(2.0)
        a.translation = Vector3(0.0)
        self.assertEqual(a, Matrix4.from_diagonal((3.0, -1.0, 2.0, 1.0)))

    def test_methods(self):
        self.assertEqual(Matrix4.rotation_y(Deg(45.0)).transposed(),
                         Matrix4.rotation_y(Deg(-45.0)))
        self.assertEqual(Matrix4.scaling(Vector3(3.0)).inverted(),
                         Matrix4.scaling(Vector3(1/3.0)))

    def test_methods_return_type(self):
        self.assertIsInstance(Matrix4.identity_init(), Matrix4)
        self.assertIsInstance(Matrix4.from_diagonal((3.0, 1.5, 1.0, 1.0)), Matrix4)
        self.assertIsInstance(Matrix4()@Matrix4(), Matrix4)
        self.assertIsInstance(Matrix4()+Matrix4(), Matrix4)
        self.assertIsInstance(Matrix4().transposed(), Matrix4)
        self.assertIsInstance(Matrix4().inverted(), Matrix4)

    # conversion from buffer is tested in test_math_numpy, array.array is
    # one-dimensional and I don't want to drag numpy here just for one test

    def test_to_buffer(self):
        # Matrix3 doesn't define py::buffer_protocol(), the one from base
        # should "just work"
        a = memoryview(Matrix4((1.0, 2.0, 3.0, 4.0),
                               (5.0, 6.0, 7.0, 8.0),
                               (9.0, 10.0, 11.0, 12.0),
                               (13.0, 14.0, 15.0, 16.0)))
        self.assertEqual(a.tolist(), [
            [1.0, 5.0, 9.0, 13.0],
            [2.0, 6.0, 10.0, 14.0],
            [3.0, 7.0, 11.0, 15.0],
            [4.0, 8.0, 12.0, 16.0]])

class Quaternion_(unittest.TestCase):
    def test_init(self):
        a = Quaternion()
        self.assertEqual(a.vector, Vector3(0.0, 0.0, 0.0))
        self.assertEqual(a.scalar, 1.0)

        b = Quaternion.identity_init()
        self.assertEqual(b.vector, Vector3(0.0, 0.0, 0.0))
        self.assertEqual(b.scalar, 1.0)

        c = Quaternion.zero_init()
        self.assertEqual(c.vector, Vector3(0.0, 0.0, 0.0))
        self.assertEqual(c.scalar, 0.0)

        d = Quaternion(Vector3(1.0, 2.0, 3.0), 4.0)
        self.assertEqual(d.vector, Vector3(1.0, 2.0, 3.0))
        self.assertEqual(d.scalar, 4.0)

        e = Quaternion(((1.0, 2.0, 3.0), 4.0))
        self.assertEqual(e.vector, Vector3(1.0, 2.0, 3.0))
        self.assertEqual(e.scalar, 4.0)

    def test_convert(self):
        a = Quaterniond(Quaternion((1.0, 2.0, 3.0), 4.0))
        self.assertEqual(a, Quaterniond((1.0, 2.0, 3.0), 4.0))

    def test_static_methods(self):
        a = Quaternion.rotation(Deg(45.0), Vector3.x_axis())
        self.assertEqual(a, Quaternion((0.382683, 0.0, 0.0), 0.92388))
        self.assertEqual(a.to_matrix(), Matrix4.rotation_x(Deg(45.0)).rotation_scaling())

    def test_methods(self):
        a = Quaternion.rotation(Deg(45.0), Vector3.x_axis())
        self.assertEqual(a.inverted(),
            Quaternion.rotation(Deg(45.0), -Vector3.x_axis()))
        self.assertAlmostEqual(float(Deg(a.angle())), float(Deg(45.0)), 4)

    def test_functions(self):
        a = math.angle(Quaterniond.rotation(Deg(45.0), Vector3d.x_axis()),
                       Quaterniond.rotation(Deg(75.0), Vector3d.x_axis()))
        self.assertEqual(Deg(a), Deg(15.0))

    def test_properties(self):
        a = Quaternion()
        a.vector = (1.0, 2.0, 3.0)
        a.scalar = 4.0
        self.assertEqual(a.vector, Vector3(1.0, 2.0, 3.0))
        self.assertEqual(a.scalar, 4.0)
        self.assertEqual(a, Quaternion((1.0, 2.0, 3.0), 4.0))

    def test_repr(self):
        a = Quaternion.rotation(Deg(45.0), Vector3.x_axis())
        self.assertEqual(repr(a), 'Quaternion({0.382683, 0, 0}, 0.92388)')

class Range(unittest.TestCase):
    def test_init(self):
        a = Range1Di()
        self.assertEqual(a.min, 0)
        self.assertEqual(a.max, 0)

        b = Range1D(3.5, 5.0)
        self.assertEqual(b.min, 3.5)
        self.assertEqual(b.max, 5.0)

        c1 = Range2D(Vector2(1.0, 0.3), Vector2(2.0, 0.6))
        self.assertEqual(c1.min, Vector2(1.0, 0.3))
        self.assertEqual(c1.max, Vector2(2.0, 0.6))

        c2 = Range2D((1.0, 0.3), (2.0, 0.6))
        self.assertEqual(c2.min, Vector2(1.0, 0.3))
        self.assertEqual(c2.max, Vector2(2.0, 0.6))

        c3 = Range2D(((1.0, 0.3), (2.0, 0.6)))
        self.assertEqual(c3.min, Vector2(1.0, 0.3))
        self.assertEqual(c3.max, Vector2(2.0, 0.6))

        d1 = Range3Dd(Vector3d(1.0, 0.2, 0.3), Vector3d(1.0, 2.0, 3.0))
        self.assertEqual(d1.min, Vector3d(1.0, 0.2, 0.3))
        self.assertEqual(d1.max, Vector3d(1.0, 2.0, 3.0))

        d2 = Range3Dd((1.0, 0.2, 0.3), (1.0, 2.0, 3.0))
        self.assertEqual(d2.min, Vector3d(1.0, 0.2, 0.3))
        self.assertEqual(d2.max, Vector3d(1.0, 2.0, 3.0))

        d3 = Range3Dd(((1.0, 0.2, 0.3), (1.0, 2.0, 3.0)))
        self.assertEqual(d3.min, Vector3d(1.0, 0.2, 0.3))
        self.assertEqual(d3.max, Vector3d(1.0, 2.0, 3.0))

    def test_convert(self):
        a = Range2Dd(Range2Di((3, 5), (8, 7)))
        self.assertEqual(a, Range2Dd((3.0, 5.0), (8.0, 7.0)))

    def test_static_methods(self):
        a = Range2D.zero_init()
        self.assertEqual(a.min, Vector2())
        self.assertEqual(a.max, Vector2())

        b = Range2D.from_size((3.0, 1.0), (2.0, 0.5))
        self.assertEqual(b.min, Vector2(3.0, 1.0))
        self.assertEqual(b.max, Vector2(5.0, 1.5))

        c = Range2D.from_center((4.0, 1.25), (1.0, 0.25))
        self.assertEqual(c.min, Vector2(3.0, 1.0))
        self.assertEqual(c.max, Vector2(5.0, 1.5))

    def test_properties(self):
        a = Range2D((1.0, 0.2), (2.0, 0.4))
        self.assertEqual(a.bottom_right, Vector2(2.0, 0.2))
        self.assertEqual(a.top_left, Vector2(1.0, 0.4))

        a.bottom_right = Vector2(3.0, 0.3)
        a.top = 7.0
        self.assertEqual(a, Range2D((1.0, 0.3), (3.0, 7.0)))

        b = Range3D((1.0, 0.2, -1.0), (2.0, 0.4, 5.0))
        self.assertEqual(b.front_bottom_right, Vector3(2.0, 0.2, 5.0))
        self.assertEqual(b.back_top_left, Vector3(1.0, 0.4, -1.0))

        b.back_bottom_right = Vector3(3.0, 0.3, -1.5)
        b.front = 7.0
        b.left = 1.1
        self.assertEqual(b, Range3D((1.1, 0.3, -1.5), (3.0, 0.4, 7.0)))

    def test_methods(self):
        a = math.join(Range2D(Vector2(), (3.0, 5.0)).translated((1.5, 0.7)),
                      Range2D((0.3, 2.0), (0.4, 2.1)))
        self.assertEqual(a, Range2D((0.3, 0.7), (4.5, 5.7)))
        self.assertEqual(a.center(), Vector2(2.4, 3.2))
