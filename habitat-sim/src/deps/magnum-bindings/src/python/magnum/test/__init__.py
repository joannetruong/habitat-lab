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

import os
import sys
import unittest

# TODO: do this differently / more robustly
sys.path = [os.path.join(os.path.dirname(__file__), os.environ.get('CMAKE_BINARY_DIR', '../../../../build'), 'src/python')] + sys.path

from magnum import *
from magnum import gl, platform

try:
    from magnum.platform.glx import WindowlessApplication
except ImportError:
    try:
        from magnum.platform.wgl import WindowlessApplication
    except ImportError:
        try:
            from magnum.platform.egl import WindowlessApplication
        except ImportError:
            WindowlessApplication = None

def setUpModule():
    if os.environ.get('MAGNUM_SKIP_GL_TESTS') == 'ON':
        raise unittest.SkipTest('GL tests skipped')

    if not WindowlessApplication:
        raise unittest.SkipTest('no WindowlessApplication found')

class GLTestCase(unittest.TestCase):
    app = None

    @classmethod
    def setUpClass(cls):
        if not GLTestCase.app:
            GLTestCase.app = WindowlessApplication()

    def assertNoGLError(self):
        self.assertEqual(gl.Renderer.error, gl.Renderer.Error.NO_ERROR)

    def setUp(self):
        self.assertNoGLError()

    def tearDown(self):
        self.assertNoGLError()
