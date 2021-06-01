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

import sys
import unittest

from magnum import *

class PixelStorage_(unittest.TestCase):
    def test_init(self):
        a = PixelStorage()
        self.assertEqual(a.alignment, 4)
        self.assertEqual(a.row_length, 0)
        self.assertEqual(a.image_height, 0)
        self.assertEqual(a.skip, Vector3i())

    def test_properties(self):
        a = PixelStorage()
        a.alignment = 1
        a.row_length = 64
        a.image_height = 256
        a.skip = (3, 1, 2)
        self.assertEqual(a.alignment, 1)
        self.assertEqual(a.row_length, 64)
        self.assertEqual(a.image_height, 256)
        self.assertEqual(a.skip, Vector3i(3, 1, 2))

class Image(unittest.TestCase):
    def test_init(self):
        storage = PixelStorage()
        storage.alignment = 1
        a = Image2D(storage, PixelFormat.RGB8_UNORM)
        self.assertEqual(a.storage.alignment, 1)
        self.assertEqual(a.size, Vector2i())
        self.assertEqual(a.format, PixelFormat.RGB8_UNORM)
        self.assertEqual(len(a.data), 0)

        b = Image2D(PixelFormat.R8I)
        self.assertEqual(b.storage.alignment, 4)
        self.assertEqual(b.size, Vector2i())
        self.assertEqual(b.format, PixelFormat.R8I)
        self.assertEqual(len(b.data), 0)

    @unittest.skip("No way to create a non-empty Image at the moment")
    def test_data(self):
        # Tested in test_gl_gl.Framebuffer.test_read_image instead
        a = Image2D(PixelFormat.R8I, Vector2i(3, 17)) # TODO
        a_refcount = sys.getrefcount(a)

        data = a.data
        self.assertEqual(len(data), 3*17*1)
        self.assertIs(data.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        del data
        self.assertEqual(sys.getrefcount(a), a_refcount)

    def test_data_empty(self):
        a = Image2D(PixelFormat.R8I)
        a_refcount = sys.getrefcount(a)

        data = a.data
        self.assertEqual(len(data), 0)
        self.assertIs(data.owner, None)
        self.assertEqual(sys.getrefcount(a), a_refcount)

    @unittest.skip("No way to create a non-empty Image at the moment")
    def test_pixels(self):
        # Tested in test_gl_gl.Framebuffer.test_read_image instead
        a = Image2D(PixelFormat.RG32UI, Vector2i(3, 17)) # TODO
        a_refcount = sys.getrefcount(a)

        pixels = a.pixels
        self.assertEqual(pixels.size, (3, 17, 8))
        self.assertIs(pixels.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        del pixels
        self.assertEqual(sys.getrefcount(a), a_refcount)

    def test_pixels_empty(self):
        a = Image2D(PixelFormat.R8I)
        a_refcount = sys.getrefcount(a)

        pixels = a.pixels
        self.assertEqual(pixels.size, (0, 0, 1))
        self.assertIs(pixels.owner, None)
        self.assertEqual(sys.getrefcount(a), a_refcount)

class ImageView(unittest.TestCase):
    def test_init(self):
        # 2x4 RGB pixels, padded for alignment
        data = (b'rgbRGB  '
                b'abcABC  '
                b'defDEF  '
                b'ijkIJK  ')
        data_refcount = sys.getrefcount(data)

        a = ImageView2D(PixelFormat.RGB8_UNORM, (2, 4), data)
        self.assertEqual(a.storage, PixelStorage())
        self.assertEqual(a.size, Vector2i(2, 4))
        self.assertEqual(a.format, PixelFormat.RGB8_UNORM)
        self.assertEqual(a.pixel_size, 3)
        self.assertEqual(len(a.data), 32)
        self.assertIs(a.owner, data)
        self.assertEqual(sys.getrefcount(data), data_refcount + 1)

        # Third row, second pixel, green channel
        self.assertEqual(a.pixels[2][1][1], 'E')

        # Deleting the original data shouldn't make the image invalid
        del data
        self.assertEqual(a.pixels[2][1][1], 'E')

    def test_init_storage(self):
        # 2x2x2 RGB pixels
        data = (b'rgbRGB'
                b'abcABC'
                b'defDEF'
                b'ijkIJK')
        data_refcount = sys.getrefcount(data)

        storage = PixelStorage()
        storage.alignment = 2
        a = ImageView3D(storage, PixelFormat.RGB8_UNORM, (2, 2, 2), data)
        self.assertEqual(a.storage.alignment, 2)
        self.assertEqual(a.size, Vector3i(2, 2, 2))
        self.assertEqual(len(a.data), 24)

        # Second image, first row, second pixel, green channel
        self.assertEqual(a.pixels[1][0][1][1], 'E')

    def test_init_empty(self):
        a = MutableImageView1D(PixelFormat.RG16UI, 32)
        self.assertEqual(a.storage.alignment, 4)
        self.assertEqual(a.size, 32)
        self.assertEqual(len(a.data), 0)
        self.assertEqual(a.owner, None)

        storage = PixelStorage()
        storage.alignment = 2
        b = ImageView2D(storage, PixelFormat.R32F, (8, 8))
        self.assertEqual(b.storage.alignment, 2)
        self.assertEqual(b.size, Vector2i(8, 8))
        self.assertEqual(len(b.data), 0)
        self.assertEqual(b.owner, None)

    def test_init_mutable(self):
        # 2x4 RGB pixels, padded for alignment
        data = bytearray(b'rgbRGB  '
                         b'abcABC  '
                         b'defDEF  '
                         b'ijkIJK  ')
        data_refcount = sys.getrefcount(data)

        a = MutableImageView2D(PixelFormat.RGB8_UNORM, (2, 4), data)
        self.assertEqual(sys.getrefcount(data), data_refcount + 1)

        a.pixels[1, 1, 1] = '_'
        a.pixels[1, 0, 1] = '_'
        a.pixels[2, 1, 1] = '_'
        a.pixels[2, 0, 1] = '_'

        self.assertEqual(data, b'rgbRGB  '
                               b'a_cA_C  '
                               b'd_fD_F  '
                               b'ijkIJK  ')

        # Back to immutable
        b = ImageView2D(a)
        self.assertEqual(b.size, Vector2i(2, 4))
        self.assertEqual(b.format, PixelFormat.RGB8_UNORM)
        self.assertEqual(b.pixel_size, 3)
        self.assertEqual(len(b.data), 32)
        self.assertIs(b.owner, data)
        self.assertEqual(sys.getrefcount(data), data_refcount + 2)

    @unittest.skip("No way to create a non-empty Image at the moment")
    def test_init_image(self):
        # Tested in test_gl_gl.Framebuffer.test_read_image instead
        a = Image2D(PixelFormat.R32F, Vector2i(3, 17)) # TODO
        a_refcount = sys.getrefcount(a)

        view = ImageView2D(a)
        self.assertEqual(view.size, (3, 17))
        self.assertIs(view.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        del view
        self.assertEqual(sys.getrefcount(a), a_refcount)

        mview = MutableImageView2D(a)
        self.assertEqual(mview.size, (3, 17))
        self.assertIs(mview.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        del mview
        self.assertEqual(sys.getrefcount(a), a_refcount)

    def test_init_image_empty(self):
        a = Image2D(PixelFormat.R32F)
        a_refcount = sys.getrefcount(a)

        view = ImageView2D(a)
        self.assertEqual(view.size, (0, 0))
        self.assertIs(view.owner, None)
        self.assertEqual(sys.getrefcount(a), a_refcount)

        mview = MutableImageView2D(a)
        self.assertEqual(mview.size, (0, 0))
        self.assertIs(mview.owner, None)
        self.assertEqual(sys.getrefcount(a), a_refcount)

    def test_set_data(self):
        # 2x4 RGB pixels, padded for alignment
        data = (b'rgbRGB  '
                b'abcABC  '
                b'defDEF  '
                b'ijkIJK  ')
        data_refcount = sys.getrefcount(data)

        a = ImageView2D(PixelFormat.RGB8_UNORM, (2, 4), data)
        self.assertIs(a.owner, data)
        self.assertEqual(sys.getrefcount(data), data_refcount + 1)

        data2 = (b'ijkIJK  '
                 b'defDEF  '
                 b'abcABC  '
                 b'rgbRGB  ')
        data2_refcount = sys.getrefcount(data2)

        # Replacing the data should disown the original object and point to the
        # new one
        a.data = data2
        self.assertEqual(bytes(a.data), (
            b'ijkIJK  '
            b'defDEF  '
            b'abcABC  '
            b'rgbRGB  '))
        self.assertIs(a.owner, data2)
        self.assertEqual(sys.getrefcount(data), data_refcount)
        self.assertEqual(sys.getrefcount(data2), data2_refcount + 1)
