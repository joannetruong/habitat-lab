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

import magnum
from magnum import gl

class Attribute(unittest.TestCase):
    def test_init(self):
        a = gl.Attribute(gl.Attribute.Kind.GENERIC, 2, gl.Attribute.Components.TWO, gl.Attribute.DataType.FLOAT)
        self.assertEqual(a.kind, gl.Attribute.Kind.GENERIC)
        self.assertEqual(a.location, 2)
        self.assertEqual(a.components, gl.Attribute.Components.TWO)
        self.assertEqual(a.data_type, gl.Attribute.DataType.FLOAT)

class Context(unittest.TestCase):
    def test_no_current(self):
        self.assertFalse(gl.Context.has_current)

        with self.assertRaisesRegex(RuntimeError, "no current context"):
            gl.Context.current

class FramebufferClear(unittest.TestCase):
    def test_ops(self):
        self.assertEqual(gl.FramebufferClear.COLOR|gl.FramebufferClear.COLOR, gl.FramebufferClear.COLOR)
        self.assertFalse(gl.FramebufferClear.COLOR & ~gl.FramebufferClear.COLOR)

class Version(unittest.TestCase):
    def test_enum_to_major_minor(self):
        if magnum.TARGET_GLES:
            self.assertEqual(gl.version(gl.Version.GLES200), (2, 0))
        else:
            self.assertEqual(gl.version(gl.Version.GL430), (4, 3))

    def test_major_minor_to_enum(self):
        if magnum.TARGET_GLES:
            self.assertEqual(gl.version(3, 0), gl.Version.GLES300)
        else:
            self.assertEqual(gl.version(4, 3), gl.Version.GL430)
