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
from magnum import primitives

class Axis(unittest.TestCase):
    def test_2d(self):
        a = primitives.axis2d()
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertTrue(a.is_indexed)
        self.assertEqual(a.attribute_count, 2)

    def test_3d(self):
        a = primitives.axis3d()
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertTrue(a.is_indexed)
        self.assertEqual(a.attribute_count, 2)

class Capsule(unittest.TestCase):
    def test_2d_wireframe(self):
        a = primitives.capsule2d_wireframe(3, 3, 2.0)
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertEqual(a.attribute_count, 1)

    def test_3d_solid(self):
        a = primitives.capsule3d_solid(3, 3, 10, 2.0, primitives.CapsuleFlags.TEXTURE_COORDINATES|primitives.CapsuleFlags.TANGENTS)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(a.is_indexed)
        self.assertEqual(a.attribute_count, 4)

        b = primitives.capsule3d_solid(3, 3, 10, 2.0)
        self.assertEqual(b.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(b.is_indexed)
        self.assertEqual(b.attribute_count, 2)

    def test_3d_wireframe(self):
        a = primitives.capsule3d_wireframe(5, 3, 12, 0.3)
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertTrue(a.is_indexed)

class Circle(unittest.TestCase):
    def test_2d_solid(self):
        a = primitives.circle2d_solid(5, primitives.Circle2DFlags.TEXTURE_COORDINATES)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_FAN)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 2)

        b = primitives.circle2d_solid(5)
        self.assertEqual(b.primitive, MeshPrimitive.TRIANGLE_FAN)
        self.assertFalse(b.is_indexed)
        self.assertEqual(b.attribute_count, 1)

    def test_2d_wireframe(self):
        a = primitives.circle2d_wireframe(5)
        self.assertEqual(a.primitive, MeshPrimitive.LINE_LOOP)
        self.assertFalse(a.is_indexed)

    def test_3d_solid(self):
        a = primitives.circle3d_solid(5, primitives.Circle3DFlags.TEXTURE_COORDINATES|primitives.Circle3DFlags.TANGENTS)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_FAN)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 4)

        b = primitives.circle3d_solid(5)
        self.assertEqual(b.primitive, MeshPrimitive.TRIANGLE_FAN)
        self.assertFalse(b.is_indexed)
        self.assertEqual(b.attribute_count, 2)

    def test_3d_wireframe(self):
        a = primitives.circle3d_wireframe(5)
        self.assertEqual(a.primitive, MeshPrimitive.LINE_LOOP)
        self.assertFalse(a.is_indexed)

class Cone(unittest.TestCase):
    def test_solid(self):
        a = primitives.cone_solid(5, 7, 7.1, primitives.ConeFlags.TEXTURE_COORDINATES|primitives.ConeFlags.CAP_END)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(a.is_indexed)
        self.assertEqual(a.attribute_count, 3)

        b = primitives.cone_solid(5, 7, 7.1)
        self.assertEqual(b.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(b.is_indexed)
        self.assertEqual(b.attribute_count, 2)

    def test_wireframe(self):
        a = primitives.cone_wireframe(16, 7.1)
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertTrue(a.is_indexed)

class Crosshair(unittest.TestCase):
    def test_2d(self):
        a = primitives.crosshair2d()
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertFalse(a.is_indexed)

    def test_3d(self):
        a = primitives.crosshair3d()
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertFalse(a.is_indexed)

class Cube(unittest.TestCase):
    def test_solid(self):
        a = primitives.cube_solid()
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(a.is_indexed)

    def test_solid_strip(self):
        a = primitives.cube_solid_strip()
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)

    def test_wireframe(self):
        a = primitives.cube_wireframe()
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertTrue(a.is_indexed)

class Cylinder(unittest.TestCase):
    def test_solid(self):
        a = primitives.cylinder_solid(7, 12, 0.2, primitives.CylinderFlags.TEXTURE_COORDINATES|primitives.CylinderFlags.CAP_ENDS)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(a.is_indexed)
        self.assertEqual(a.attribute_count, 3)

        b = primitives.cylinder_solid(7, 12, 0.2)
        self.assertEqual(b.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(b.is_indexed)
        self.assertEqual(b.attribute_count, 2)

    def test_wireframe(self):
        a = primitives.cylinder_wireframe(8, 16, 1.1)
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertTrue(a.is_indexed)

class Gradient(unittest.TestCase):
    def test_gradient2d(self):
        a = primitives.gradient2d((3.1, 2.0), Color3(), (0.2, 1.1), Color4())
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 2)

    def test_gradient2d_horizontal(self):
        a = primitives.gradient2d_horizontal(Color4(), Color3())
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 2)

    def test_gradient2d_vertical(self):
        a = primitives.gradient2d_vertical(Color4(), Color3())
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 2)

    def test_gradient3d(self):
        a = primitives.gradient3d((3.1, 2.0, 0.1), Color3(), (0.2, 1.1, 1.2), Color4())
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 3)

    def test_gradient3d_horizontal(self):
        a = primitives.gradient3d_horizontal(Color4(), Color3())
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 3)

    def test_gradient3d_vertical(self):
        a = primitives.gradient3d_vertical(Color4(), Color3())
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 3)

class Grid(unittest.TestCase):
    def test_solid(self):
        a = primitives.grid3d_solid((4, 5), primitives.GridFlags.NORMALS|primitives.GridFlags.TANGENTS)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(a.is_indexed)
        self.assertEqual(a.attribute_count, 3)

    def test_wireframe(self):
        a = primitives.grid3d_wireframe((2, 7))
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertTrue(a.is_indexed)

class Icosphere(unittest.TestCase):
    def test(self):
        a = primitives.icosphere_solid(2)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(a.is_indexed)

class Line(unittest.TestCase):
    def test_2d(self):
        a = primitives.line2d((1.0, 2.0), (7.0, 3.2))
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertFalse(a.is_indexed)

    def test_2d_identity(self):
        a = primitives.line2d()
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertFalse(a.is_indexed)

    def test_3d(self):
        a = primitives.line3d((1.0, 2.0, 1.1), (7.0, 3.2, 1.1))
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertFalse(a.is_indexed)

    def test_3d_identity(self):
        a = primitives.line3d()
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertFalse(a.is_indexed)

class Plane(unittest.TestCase):
    def test_solid(self):
        a = primitives.plane_solid(primitives.PlaneFlags.TEXTURE_COORDINATES|primitives.PlaneFlags.TANGENTS)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 4)

        b = primitives.plane_solid()
        self.assertEqual(b.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(b.is_indexed)
        self.assertEqual(b.attribute_count, 2)

    def test_wireframe(self):
        a = primitives.plane_wireframe()
        self.assertEqual(a.primitive, MeshPrimitive.LINE_LOOP)
        self.assertFalse(a.is_indexed)

class Square(unittest.TestCase):
    def test_solid(self):
        a = primitives.square_solid(primitives.SquareFlags.TEXTURE_COORDINATES)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(a.is_indexed)
        self.assertEqual(a.attribute_count, 2)

        b = primitives.square_solid()
        self.assertEqual(b.primitive, MeshPrimitive.TRIANGLE_STRIP)
        self.assertFalse(b.is_indexed)
        self.assertEqual(b.attribute_count, 1)

    def test_wireframe(self):
        a = primitives.square_wireframe()
        self.assertEqual(a.primitive, MeshPrimitive.LINE_LOOP)
        self.assertFalse(a.is_indexed)

class UVSphere(unittest.TestCase):
    def test_solid(self):
        a = primitives.uv_sphere_solid(3, 7, primitives.UVSphereFlags.TEXTURE_COORDINATES|primitives.UVSphereFlags.TANGENTS)
        self.assertEqual(a.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(a.is_indexed)
        self.assertEqual(a.attribute_count, 4)

        b = primitives.uv_sphere_solid(3, 7)
        self.assertEqual(b.primitive, MeshPrimitive.TRIANGLES)
        self.assertTrue(b.is_indexed)
        self.assertEqual(b.attribute_count, 2)

    def test_wireframe(self):
        a = primitives.uv_sphere_wireframe(6, 8)
        self.assertEqual(a.primitive, MeshPrimitive.LINES)
        self.assertTrue(a.is_indexed)
