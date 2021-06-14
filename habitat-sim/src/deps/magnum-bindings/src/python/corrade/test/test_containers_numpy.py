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

from corrade import containers
import test_stridedarrayview

try:
    import numpy as np
except ModuleNotFoundError:
    raise unittest.SkipTest("numpy not installed")

class StridedArrayViewCustomType(unittest.TestCase):
    # short and mutable_int tested in test_containers, as for those memoryview
    # works well... well, for one dimension it does

    def test_mutable_vector3d(self):
        a = test_stridedarrayview.MutableContainer3d()
        self.assertEqual(type(a.view), containers.MutableStridedArrayView2D)
        self.assertEqual(a.view.format, 'ddd')
        self.assertEqual(a.list, [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        a.view[0][1] = [-765.6581, 3.5, 1.125]
        a.view[1][2] = [4.666, 0.25, -7.5]
        self.assertEqual(a.list, [
            [0.0, 0.0, 0.0],
            [-765.6581, 3.5, 1.125],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [4.666, 0.25, -7.5]
        ])

        # memoryview ... doesn't understand the type. HAH
        mav = memoryview(a.view[0])
        with self.assertRaisesRegex(NotImplementedError, "unsupported format ddd"):
            self.assertEqual(mav[1], [-765.6581, 3.5, 1.125])

        # Test that numpy understands the type and has changes reflected
        av = np.array(a.view, copy=False)
        a.view[1][0] = [-3.33, 1.0, 0.0]
        # Converting to a tuple, otherwise numpy always compares to False
        self.assertEqual(tuple(av[1][0]), (-3.33, 1.0, 0.0))
        self.assertEqual(tuple(av[1][1]), (0.0, 0.0, 0.0))
        self.assertEqual(tuple(av[1][2]), (4.666, 0.25, -7.5))

        # And the other way around as well
        av[1][1] = (1.0, 0.125, 1.125)
        self.assertEqual(a.list, [
            [0.0, 0.0, 0.0],
            [-765.6581, 3.5, 1.125],
            [0.0, 0.0, 0.0],
            [-3.33, 1.0, 0.0],
            [1.0, 0.125, 1.125],
            [4.666, 0.25, -7.5]
        ])

    def test_mutable_long_float(self):
        a = test_stridedarrayview.MutableContainerlf()
        self.assertEqual(type(a.view), containers.MutableStridedArrayView2D)
        self.assertEqual(a.view.format, 'Qf')
        self.assertEqual(a.list, [
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0)
        ])
        a.view[0][1] = (7656581356781257, 1.125)
        a.view[1][2] = (4666025, -7.5)
        self.assertEqual(a.list, [
            (0, 0.0),
            (7656581356781257, 1.125),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (4666025, -7.5)
        ])

        # memoryview ... doesn't understand the type. HAH
        mav = memoryview(a.view[0])
        with self.assertRaisesRegex(NotImplementedError, "unsupported format Qf"):
            self.assertEqual(mav[1], (7656581356781257, 1.125))

        # Test that numpy understands the type and has changes reflected
        av = np.array(a.view, copy=False)
        a.view[1][0] = (333106832, 0.0)
        # Converting to a tuple, otherwise numpy always compares to False
        self.assertEqual(tuple(av[1][0]), (333106832, 0.0))
        self.assertEqual(tuple(av[1][1]), (0, 0.0))
        self.assertEqual(tuple(av[1][2]), (4666025, -7.5))

        # And the other way around as well
        av[1][1] = (1001, 1.125)
        self.assertEqual(a.list, [
            (0, 0.0),
            (7656581356781257, 1.125),
            (0, 0.0),
            (333106832, 0.0),
            (1001, 1.125),
            (4666025, -7.5)
        ])
