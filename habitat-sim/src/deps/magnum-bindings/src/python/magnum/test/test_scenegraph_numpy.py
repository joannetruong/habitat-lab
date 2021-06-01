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
from magnum import scenegraph
from magnum.scenegraph.matrix import Object3D, Scene3D

try:
    import numpy as np
except ModuleNotFoundError:
    raise unittest.SkipTest("numpy not installed")

class Object(unittest.TestCase):
    def test_transformation(self):
        scene = Scene3D()

        a = Object3D(scene)

        # like a.rotate_local(Deg(35.0), Vector3.x_axis()), but way uglier,
        # another could be scipy.spatial.transform.Rotation but that's meh as
        # well
        a.transform_local(np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 0.819152, -0.573576, 0.0],
             [0.0, 0.573576, 0.819152, 0.0],
             [0.0, 0.0, 0.0, 1.0]]))
        self.assertEqual(a.transformation, Matrix4.rotation_x(Deg(35.0)))
        self.assertEqual(a.absolute_transformation(), Matrix4.rotation_x(Deg(35.0)))

        b = Object3D(a)
        b.translate(np.array([3.0, 4.0, 5.0], dtype='float32'))
        self.assertEqual(b.transformation, Matrix4.translation((3.0, 4.0, 5.0)))
        self.assertEqual(b.absolute_transformation(),
            Matrix4.rotation_x(Deg(35.0))@
            Matrix4.translation((3.0, 4.0, 5.0)))

        c = Object3D(scene)
        self.assertEqual(c.transformation, Matrix4.identity_init())
        self.assertEqual(c.absolute_transformation(), Matrix4.identity_init())
