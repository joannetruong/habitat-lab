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
import sys
import unittest

from corrade import containers

class ArrayView(unittest.TestCase):
    def test_init(self):
        a = containers.ArrayView()
        b = containers.MutableArrayView()
        self.assertIs(a.owner, None)
        self.assertIs(b.owner, None)
        self.assertEqual(len(a), 0)
        self.assertEqual(len(b), 0)
        self.assertEqual(bytes(a), b'')
        self.assertEqual(bytes(b), b'')

    def test_init_buffer(self):
        a = b'hello'
        a_refcount = sys.getrefcount(a)

        b = containers.ArrayView(a)
        self.assertIs(b.owner, a)
        self.assertEqual(len(b), 5)
        self.assertEqual(bytes(b), b'hello')
        self.assertEqual(b[2], 'l')
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        # Not mutable
        with self.assertRaisesRegex(TypeError, "object does not support item assignment"):
            b[4] = '!'

        # b should keep a reference to a, so deleting the local reference
        # shouldn't affect it
        del a
        self.assertTrue(sys.getrefcount(b.owner), a_refcount)
        self.assertEqual(b[2], 'l')

        # Now, if we delete b, a should not be referenced by anything anymore
        a = b.owner
        del b
        self.assertTrue(sys.getrefcount(a), a_refcount)

    def test_init_buffer_empty(self):
        a = b''
        a_refcount = sys.getrefcount(a)

        b = containers.ArrayView(a)
        self.assertIs(b.owner, None)
        self.assertEqual(len(b), 0)
        self.assertEqual(sys.getrefcount(a), a_refcount)

    def test_init_buffer_memoryview_obj(self):
        a = b'hello'
        v = memoryview(a)
        b = containers.ArrayView(v)
        # memoryview's buffer protocol returns itself, not the underlying
        # bytes, as it manages the Py_buffer instance. So this is expected.
        self.assertIs(b.owner, v)

    def test_init_buffer_mutable(self):
        a = bytearray(b'hello')
        b = containers.MutableArrayView(a)
        b[4] = '!'
        self.assertEqual(b[4], '!')
        self.assertEqual(bytes(b), b'hell!')

    def test_init_array(self):
        a = array.array('f', [1.0, 4.5, 7.9])
        b = containers.ArrayView(a)
        self.assertIs(b.owner, a)
        self.assertEqual(len(b), 3*4)

    def test_init_buffer_unexpected_stride(self):
        a = memoryview(b'hello')[::2]
        self.assertEqual(bytes(a), b'hlo')
        # Error emitted by memoryview, not us
        with self.assertRaisesRegex(BufferError, "memoryview: underlying buffer is not C-contiguous"):
            b = containers.ArrayView(a)

    def test_init_buffer_mutable_from_immutable(self):
        a = b'hello'
        with self.assertRaisesRegex(BufferError, "Object is not writable."):
            b = containers.MutableArrayView(a)

    def test_slice(self):
        a = b'World is hell!'
        a_refcount = sys.getrefcount(a)

        b = containers.ArrayView(a)
        b_refcount = sys.getrefcount(b)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        # When slicing, b's refcount should not change but a's refcount should
        # increase
        c = b[4:-4]
        self.assertIsInstance(c, containers.ArrayView)
        self.assertEqual(bytes(c), b'd is h')
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 2)

        # Deleting a slice should reduce a's refcount again, keep b's unchanged
        del c
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

    def test_slice_empty(self):
        data = b'hello'
        data_refcount = sys.getrefcount(data)

        # slice.start = slice.stop
        a = containers.ArrayView(data)[7:8]
        self.assertEqual(len(a), 0)

        # Empty view, original data not referenced at all
        self.assertIs(a.owner, None)
        self.assertEqual(sys.getrefcount(data), data_refcount)

    def test_slice_invalid(self):
        with self.assertRaisesRegex(ValueError, "slice step cannot be zero"):
            containers.ArrayView()[::0]

    def test_slice_stride(self):
        a = b'World_ _i_s_ _hell!'
        a_refcount = sys.getrefcount(a)

        b = containers.ArrayView(a)
        b_refcount = sys.getrefcount(b)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        # When slicing to a strided array view, b's refcount should not change
        # but a's refcount should increase. Check consistency with slices on
        # bytes, slicing bytes will make a copy so it doesn't affect the
        # refcount.
        c1 = a[4:-4:2]
        c2 = b[4:-4:2]
        self.assertIsInstance(c2, containers.StridedArrayView1D)
        self.assertEqual(len(c1), 6)
        self.assertEqual(len(c2), 6)
        self.assertEqual(bytes(c1), b'd is h')
        self.assertEqual(bytes(c2), b'd is h')
        self.assertEqual(c2.size, (6,))
        self.assertEqual(c2.stride, (2,))
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 2)

        # Deleting a slice should reduce a's refcount again, keep b's unchanged
        del c2
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

    def test_slice_stride_empty(self):
        data = b'hello'
        data_refcount = sys.getrefcount(data)

        # slice.start = slice.stop
        a = containers.ArrayView(data)[7:8:2]
        self.assertEqual(len(a), 0)

        # Empty view, original data not referenced at all
        self.assertIs(a.owner, None)
        self.assertEqual(sys.getrefcount(data), data_refcount)

    def test_slice_stride_negative(self):
        a = b'World_ _i_s_ _hell!'
        b = containers.ArrayView(a)

        # Check consistency with slices on bytes
        c1 = a[-5:3:-2] # like [4:-4:2] above, but reverted
        c2 = b[-5:3:-2]
        self.assertEqual(len(c1), 6)
        self.assertEqual(len(c2), 6)
        self.assertEqual(bytes(c1), b'h si d') # like b'd is h' but reverted
        self.assertEqual(bytes(c2), b'h si d')
        self.assertEqual(c2.size, (6,))
        self.assertEqual(c2.stride, (-2,))

    def test_slice_stride_reverse(self):
        # slice.stop = -1
        a = containers.ArrayView(b'hello')[::-1]
        self.assertEqual(len(a), 5)
        self.assertEqual(bytes(a), b'olleh')

    def test_convert_memoryview(self):
        a = b'World is hell!'
        a_refcount = sys.getrefcount(a)

        b = containers.ArrayView(a)
        b_refcount = sys.getrefcount(b)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        c = memoryview(b)
        # Unlike slicing, ArrayView's buffer protocol returns a reference to
        # itself -- it needs to be kept around because the Py_buffer refers to
        # its internals for size. Also returning a reference to the underlying
        # buffer would mean the underlying buffer's releasebuffer function gets
        # called instead of ours which is *not* wanted.
        self.assertIs(c.obj, b)
        self.assertEqual(sys.getrefcount(b), b_refcount + 1)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        with self.assertRaisesRegex(TypeError, "cannot modify read-only memory"):
            c[-1] = ord('?')

    def test_convert_mutable_memoryview(self):
        a = bytearray(b'World is hell!')
        b = memoryview(containers.MutableArrayView(a))
        b[-1] = ord('?')
        self.assertEqual(a, b'World is hell?')

class StridedArrayView1D(unittest.TestCase):
    def test_init(self):
        a = containers.StridedArrayView1D()
        b = containers.MutableStridedArrayView1D()
        self.assertIs(a.owner, None)
        self.assertIs(b.owner, None)
        self.assertEqual(len(a), 0)
        self.assertEqual(len(b), 0)
        self.assertEqual(bytes(a), b'')
        self.assertEqual(bytes(b), b'')
        self.assertEqual(a.size, (0, ))
        self.assertEqual(b.size, (0, ))
        self.assertEqual(a.stride, (0, ))
        self.assertEqual(b.stride, (0, ))
        self.assertEqual(a.dimensions, 1)
        self.assertEqual(b.dimensions, 1)

    def test_init_buffer(self):
        a = b'hello'
        a_refcount = sys.getrefcount(a)

        b = containers.StridedArrayView1D(a)
        self.assertIs(b.owner, a)
        self.assertEqual(len(b), 5)
        self.assertEqual(bytes(b), b'hello')
        self.assertEqual(b.size, (5, ))
        self.assertEqual(b.stride, (1, ))
        self.assertEqual(b[2], 'l')
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        # Not mutable
        with self.assertRaisesRegex(TypeError, "object does not support item assignment"):
            b[4] = '!'

        # b should keep a reference to a, so deleting the local reference
        # shouldn't affect it
        del a
        self.assertTrue(sys.getrefcount(b.owner), a_refcount)
        self.assertEqual(b[2], 'l')

        # Now, if we delete b, a should not be referenced by anything anymore
        a = b.owner
        del b
        self.assertTrue(sys.getrefcount(a), a_refcount)

    def test_init_buffer_empty(self):
        a = b''
        a_refcount = sys.getrefcount(a)

        b = containers.StridedArrayView1D(a)
        self.assertIs(b.owner, None)
        self.assertEqual(len(b), 0)
        self.assertEqual(sys.getrefcount(a), a_refcount)

    def test_init_buffer_memoryview_obj(self):
        a = b'hello'
        v = memoryview(a)
        b = containers.StridedArrayView1D(v)
        # memoryview's buffer protocol returns itself, not the underlying
        # bytes, as it manages the Py_buffer instance. So this is expected.
        self.assertIs(b.owner, v)

    def test_init_buffer_mutable(self):
        a = bytearray(b'hello')
        b = containers.MutableStridedArrayView1D(a)
        b[4] = '!'
        self.assertEqual(b[4], '!')
        self.assertEqual(bytes(b), b'hell!')

    def test_init_buffer_unexpected_dimensions(self):
        a = memoryview(b'123456').cast('b', shape=[2, 3])
        self.assertEqual(bytes(a), b'123456')
        with self.assertRaisesRegex(BufferError, "expected 1 dimensions but got 2"):
            b = containers.StridedArrayView1D(a)

    def test_init_buffer_stride(self):
        a = memoryview(b'hello')[::2]
        self.assertEqual(bytes(a), b'hlo')
        b = containers.StridedArrayView1D(a)
        self.assertEqual(len(b), 3)
        self.assertEqual(bytes(b), b'hlo')
        self.assertEqual(b.size, (3, ))
        self.assertEqual(b.stride, (2, ))
        self.assertEqual(b[2], 'o')

    def test_init_buffer_mutable_from_immutable(self):
        a = b'hello'
        with self.assertRaisesRegex(BufferError, "Object is not writable."):
            b = containers.MutableStridedArrayView1D(a)

    def test_slice(self):
        a = b'World is hell!'
        a_refcount = sys.getrefcount(a)

        b = containers.StridedArrayView1D(a)
        b_refcount = sys.getrefcount(b)
        self.assertIs(b.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        # When slicing, b's refcount should not change but a's refcount should
        # increase
        c = b[4:-4]
        self.assertEqual(c.size, (6,))
        self.assertEqual(c.stride, (1,))
        self.assertIs(c.owner, a)
        self.assertIsInstance(c, containers.StridedArrayView1D)
        self.assertEqual(bytes(c), b'd is h')
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 2)

        # Deleting a slice should reduce a's refcount again, keep b's unchanged
        del c
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

    def test_slice_empty(self):
        data = b'hello'
        data_refcount = sys.getrefcount(data)

        # slice.start = slice.stop
        a = containers.StridedArrayView1D(data)[7:8]
        self.assertEqual(a.size, (0, ))

        # Empty view, original data not referenced at all
        self.assertIs(a.owner, None)
        self.assertEqual(sys.getrefcount(data), data_refcount)

    def test_slice_invalid(self):
        with self.assertRaisesRegex(TypeError, "indices must be integers"):
            containers.StridedArrayView1D()[-5:3:"boo"]

    def test_slice_stride(self):
        a = b'World_ _i_s_ _hell!'
        b = containers.StridedArrayView1D(a)

        # Check consistency with slices on bytes
        c1 = a[4:-4:2]
        c2 = b[4:-4:2]
        self.assertIsInstance(c2, containers.StridedArrayView1D)
        self.assertEqual(len(c1), 6)
        self.assertEqual(len(c2), 6)
        self.assertEqual(bytes(c1), b'd is h')
        self.assertEqual(bytes(c2), b'd is h')
        self.assertEqual(c2.size, (6,))
        self.assertEqual(c2.stride, (2,))

    def test_slice_stride_negative(self):
        a = b'World_ _i_s_ _hell!'
        b = containers.StridedArrayView1D(a)

        # Check consistency with slices on bytes
        c1 = a[-5:3:-2] # like [4:-4:2] above, but reverted
        c2 = b[-5:3:-2]
        self.assertEqual(len(c1), 6)
        self.assertEqual(len(c2), 6)
        self.assertEqual(bytes(c1), b'h si d') # like b'd is h' but reverted
        self.assertEqual(bytes(c2), b'h si d')
        self.assertEqual(c2.size, (6,))
        self.assertEqual(c2.stride, (-2,))

    def test_convert_memoryview(self):
        a = b'World is hell!'
        a_refcount = sys.getrefcount(a)

        b = containers.StridedArrayView1D(a)
        b_refcount = sys.getrefcount(b)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        c = memoryview(b)
        self.assertEqual(c.ndim, 1)
        self.assertEqual(len(c), len(a))
        self.assertEqual(bytes(c), a)
        # Unlike slicing, StridedArrayView's buffer protocol returns a
        # reference to itself and not the underlying buffer -- it needs to be
        # kept around because the Py_buffer refers to its internals for size.
        # Also returning a reference to the underlying buffer would mean the
        # underlying buffer's releasebuffer function gets called instead of
        # ours which is *not* wanted.
        self.assertIs(c.obj, b)
        self.assertEqual(sys.getrefcount(b), b_refcount + 1)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        with self.assertRaisesRegex(TypeError, "cannot modify read-only memory"):
            c[-1] = ord('?')

    def test_convert_mutable_memoryview(self):
        a = bytearray(b'World is hell!')
        b = memoryview(containers.MutableStridedArrayView1D(a))
        b[-1] = ord('?')
        self.assertEqual(a, b'World is hell?')

class StridedArrayView2D(unittest.TestCase):
    def test_init(self):
        a = containers.StridedArrayView2D()
        b = containers.MutableStridedArrayView2D()
        self.assertIs(a.owner, None)
        self.assertIs(b.owner, None)
        self.assertEqual(len(a), 0)
        self.assertEqual(len(b), 0)
        self.assertEqual(bytes(a), b'')
        self.assertEqual(bytes(b), b'')
        self.assertEqual(a.size, (0, 0))
        self.assertEqual(b.size, (0, 0))
        self.assertEqual(a.stride, (0, 0))
        self.assertEqual(b.stride, (0, 0))
        self.assertEqual(a.dimensions, 2)
        self.assertEqual(b.dimensions, 2)

    def test_init_buffer(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef')
        a_refcount = sys.getrefcount(a)

        v = memoryview(a).cast('b', shape=[3, 8])
        b = containers.StridedArrayView2D(v)
        self.assertEqual(len(b), 3)
        self.assertEqual(bytes(b), b'01234567'
                                   b'456789ab'
                                   b'89abcdef')
        self.assertEqual(b.size, (3, 8))
        self.assertEqual(b.stride, (8, 1))
        self.assertIsInstance(b[1], containers.StridedArrayView1D)
        self.assertEqual(bytes(b[1]), b'456789ab')
        self.assertEqual(b[1, 2], '6')
        self.assertEqual(b[1][2], '6')
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        # Not mutable
        with self.assertRaisesRegex(TypeError, "object does not support item assignment"):
            b[1, 2] = '!'

        # b should keep a reference to a, so deleting the local reference
        # shouldn't affect it
        del a
        self.assertTrue(sys.getrefcount(b.owner), a_refcount)
        self.assertEqual(b[1][2], '6')

        # Now, if we delete b, a should not be referenced by anything anymore
        a = b.owner
        del b
        self.assertTrue(sys.getrefcount(a), a_refcount)

    def test_init_buffer_mutable(self):
        a = bytearray(b'01234567'
                      b'456789ab'
                      b'89abcdef')
        b = containers.MutableStridedArrayView2D(memoryview(a).cast('b', shape=[3, 8]))
        b[0, 7] = '!'
        b[1, 7] = '!'
        b[2, 7] = '!'
        self.assertEqual(b[0][7], '!')
        self.assertEqual(bytes(b), b'0123456!'
                                   b'456789a!'
                                   b'89abcde!')

    def test_init_buffer_unexpected_dimensions(self):
        a = b'123456'
        with self.assertRaisesRegex(BufferError, "expected 2 dimensions but got 1"):
            b = containers.StridedArrayView2D(a)

    def test_init_buffer_stride(self):
        a = memoryview(b'01234567'
                       b'456789ab'
                       b'89abcdef').cast('b', shape=[3, 8])[::2]
        self.assertEqual(bytes(a), b'0123456789abcdef')
        b = containers.StridedArrayView2D(a)
        self.assertEqual(len(b), 2)
        self.assertEqual(bytes(b), b'0123456789abcdef')
        self.assertEqual(b.size, (2, 8))
        self.assertEqual(b.stride, (16, 1))
        self.assertEqual(bytes(b[1]), b'89abcdef')
        self.assertEqual(b[1][3], 'b')

    def test_init_buffer_mutable_from_immutable(self):
        a = memoryview(b'01234567'
                       b'456789ab'
                       b'89abcdef').cast('b', shape=[3, 8])
        with self.assertRaisesRegex(BufferError, "underlying buffer is not writable"):
            b = containers.MutableStridedArrayView2D(a)

    def test_slice(self):
        a = memoryview(b'01234567'
                       b'456789ab'
                       b'89abcdef').cast('b', shape=[3, 8])
        a_refcount = sys.getrefcount(a)

        b = containers.StridedArrayView2D(a)
        b_refcount = sys.getrefcount(b)
        # memoryview's buffer protocol returns itself, not the underlying
        # bytes, as it manages the Py_buffer instance. So this is expected.
        self.assertIs(b.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        # When slicing, b's refcount should not change but a's refcount should
        # increase
        c = b[0:-1]
        self.assertIsInstance(c, containers.StridedArrayView2D)
        self.assertEqual(c.size, (2, 8))
        self.assertEqual(c.stride, (8, 1))
        self.assertEqual(bytes(c), b'01234567456789ab')
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 2)

        # Deleting a slice should reduce a's refcount again, keep b's unchanged
        del c
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

    def test_slice_multidimensional(self):
        a = memoryview(b'01234567'
                       b'456789ab'
                       b'89abcdef').cast('b', shape=[3, 8])
        a_refcount = sys.getrefcount(a)

        b = containers.StridedArrayView2D(a)
        b_refcount = sys.getrefcount(b)
        # memoryview's buffer protocol returns itself, not the underlying
        # bytes, as it manages the Py_buffer instance. So this is expected.
        self.assertIs(b.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        # When slicing, b's refcount should not change but a's refcount should
        # increase
        c = b[1:3,4:7]
        self.assertIsInstance(c, containers.StridedArrayView2D)
        self.assertEqual(c.size, (2, 3))
        self.assertEqual(c.stride, (8, 1))
        self.assertEqual(bytes(c[0]), b'89a')
        self.assertEqual(bytes(c[1]), b'cde')
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 2)

        # Deleting a slice should reduce a's refcount again, keep b's unchanged
        del c
        self.assertEqual(sys.getrefcount(b), b_refcount)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

    def test_slice_multidimensional_empty(self):
        a = memoryview(b'01234567'
                       b'456789ab'
                       b'89abcdef').cast('b', shape=[3, 8])
        a_refcount = sys.getrefcount(a)

        b = containers.StridedArrayView2D(a)[1:1,2:2]
        self.assertEqual(b.size, (0, 0))

        # Empty view, original data not referenced at all
        self.assertIs(b.owner, None)
        self.assertEqual(sys.getrefcount(a), a_refcount)

    def test_slice_invalid(self):
        with self.assertRaisesRegex(ValueError, "slice step cannot be zero"):
            containers.StridedArrayView1D()[-5:3:0]

    def test_slice_stride(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef')
        v = memoryview(a).cast('b', shape=[3, 8])
        b = containers.StridedArrayView2D(v)

        # Check consistency with slices on memoryview
        c1 = v[0:3:2]
        c2 = b[0:3:2]
        self.assertEqual(len(c1), 2)
        self.assertEqual(len(c2), 2)
        self.assertIsInstance(c2, containers.StridedArrayView2D)
        self.assertEqual(bytes(c1), b'0123456789abcdef')
        self.assertEqual(bytes(c2), b'0123456789abcdef')
        self.assertEqual(c2.size, (2, 8))
        self.assertEqual(c2.stride, (16, 1))
        self.assertEqual(bytes(c2[1]), b'89abcdef')

    def test_slice_stride_negative(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef')
        v = memoryview(a).cast('b', shape=[3, 8])
        b = containers.StridedArrayView2D(v)

        # Check consistency with slices on memoryview
        self.assertEqual(v.shape, (3, 8))
        self.assertEqual(b.size, (3, 8))
        self.assertEqual(v.strides, (8, 1))
        self.assertEqual(b.stride, (8, 1))
        c1 = v[-1:None:-2] # like [0:3:2] above, but reverted
        c2 = b[-1:None:-2]
        self.assertEqual(len(c1), 2)
        self.assertEqual(len(c2), 2)
        self.assertEqual(bytes(c1), b'89abcdef01234567') # like above but reverted
        self.assertEqual(bytes(c2), b'89abcdef01234567')
        self.assertEqual(c1.shape, (2, 8))
        self.assertEqual(c2.size, (2, 8))
        self.assertEqual(c1.strides, (-16, 1))
        self.assertEqual(c2.stride, (-16, 1))

    def test_slice_stride_negative_multidimensional(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef')
        v = memoryview(a).cast('b', shape=[3, 8])
        b = containers.StridedArrayView2D(v)

        # Check consistency with slices on memoryview
        self.assertEqual(v.shape, (3, 8))
        self.assertEqual(b.size, (3, 8))
        self.assertEqual(v.strides, (8, 1))
        self.assertEqual(b.stride, (8, 1))

        with self.assertRaises(NotImplementedError):
            c1 = v[-1:None:-2, -2:2:-3] # HAH!

        c2 = b[-1:None:-2, -2:2:-3]
        self.assertEqual(len(c2), 2)
        self.assertEqual(bytes(c2), b'eb63')
        self.assertEqual(c2.size, (2, 2))
        self.assertEqual(c2.stride, (-16, -3))

    def test_ops(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef')
        v = memoryview(a).cast('b', shape=[3, 8])

        b = containers.StridedArrayView2D(v).transposed(0, 1).flipped(0)
        self.assertEqual(b.size, (8, 3))
        self.assertEqual(b.stride, (-1, 8))
        self.assertEqual(bytes(b), b'7bf6ae59d48c37b26a159048')

        c = containers.StridedArrayView2D(v).transposed(1, 0).flipped(1)
        self.assertEqual(c.size, (8, 3))
        self.assertEqual(c.stride, (1, -8))
        self.assertEqual(bytes(c), b'840951a62b73c84d95ea6fb7')

        d = containers.StridedArrayView2D(v).transposed(0, 1)[3:4].broadcasted(0, 5)
        self.assertEqual(d.size, (5, 3))
        self.assertEqual(d.stride, (0, 8))
        self.assertEqual(bytes(d), b'37b37b37b37b37b')

        d = containers.StridedArrayView2D(v)[:, 3:4].broadcasted(1, 2)
        self.assertEqual(d.size, (3, 2))
        self.assertEqual(d.stride, (8, 0))
        self.assertEqual(bytes(d), b'3377bb')

    def test_convert_memoryview(self):
        a = memoryview(b'01234567'
                       b'456789ab'
                       b'89abcdef').cast('b', shape=[3, 8])
        a_refcount = sys.getrefcount(a)

        b = containers.StridedArrayView2D(a)
        b_refcount = sys.getrefcount(b)
        # memoryview's buffer protocol returns itself, not the underlying
        # bytes, as it manages the Py_buffer instance. So this is expected.
        self.assertIs(b.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        c = memoryview(b)
        self.assertEqual(c.ndim, 2)
        self.assertEqual(c.shape, (3, 8))
        self.assertEqual(c.strides, (8, 1))
        self.assertIs(c.obj, b)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)
        self.assertEqual(sys.getrefcount(b), b_refcount + 1)

class StridedArrayView3D(unittest.TestCase):
    def test_init_buffer(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef'

             b'cdef0123'
             b'01234567'
             b'456789ab')
        b = containers.StridedArrayView3D(memoryview(a).cast('b', shape=[2, 3, 8]))
        self.assertEqual(len(b), 2)
        self.assertEqual(bytes(b), b'01234567456789ab89abcdefcdef012301234567456789ab')
        self.assertEqual(b.size, (2, 3, 8))
        self.assertEqual(b.stride, (24, 8, 1))
        self.assertEqual(b[1, 2, 3], '7')
        self.assertEqual(b[1][2][3], '7')

    def test_init_buffer_mutable(self):
        a = bytearray(b'01234567'
                      b'456789ab'
                      b'89abcdef'

                      b'cdef0123'
                      b'01234567'
                      b'456789ab')
        b = containers.MutableStridedArrayView3D(memoryview(a).cast('b', shape=[2, 3, 8]))
        b[0, 0, 7] = '!'
        b[0, 1, 7] = '!'
        b[0, 2, 7] = '!'
        b[1, 0, 7] = '!'
        b[1, 1, 7] = '!'
        b[1, 2, 7] = '!'
        self.assertEqual(b[1][1][7], '!')
        self.assertEqual(bytes(b), b'0123456!'
                                   b'456789a!'
                                   b'89abcde!'

                                   b'cdef012!'
                                   b'0123456!'
                                   b'456789a!')

    def test_ops(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef'

             b'cdef0123'
             b'01234567'
             b'456789ab')
        v = memoryview(a).cast('b', shape=[2, 3, 8])

        b = containers.StridedArrayView3D(v).transposed(0, 1).flipped(0)
        self.assertEqual(b.size, (3, 2, 8))
        self.assertEqual(b.stride, (-8, 24, 1))
        self.assertEqual(bytes(b), b'89abcdef456789ab456789ab0123456701234567cdef0123')

        c = containers.StridedArrayView3D(v).transposed(2, 0).flipped(1)
        self.assertEqual(c.size, (8, 3, 2))
        self.assertEqual(c.stride, (1, -8, 24))
        self.assertEqual(bytes(c), b'84400c95511da6622eb7733fc88440d99551eaa662fbb773')

        d = containers.StridedArrayView3D(v).transposed(1, 2)[0:1, 3:5, :].broadcasted(0, 5)
        self.assertEqual(d.size, (5, 2, 3))
        self.assertEqual(d.stride, (0, 1, 8))
        self.assertEqual(bytes(d), b'37b48c37b48c37b48c37b48c37b48c')

        e = containers.StridedArrayView3D(v)[:, 1:2, 3:4].flipped(2).broadcasted(1, 2)
        self.assertEqual(e.size, (2, 2, 1))
        self.assertEqual(e.stride, (24, 0, -1))
        self.assertEqual(bytes(e), b'7733')

        f = containers.StridedArrayView3D(v)[:, :, 0:1].broadcasted(2, 5)
        self.assertEqual(f.size, (2, 3, 5))
        self.assertEqual(f.stride, (24, 8, 0))
        self.assertEqual(bytes(f), b'000004444488888ccccc0000044444')

# This is just a dumb copy of the above with one dimension inserted at the
# second place.
class StridedArrayView4D(unittest.TestCase):
    def test_init_buffer(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef'

             b'cdef0123'
             b'01234567'
             b'456789ab')
        b = containers.StridedArrayView4D(memoryview(a).cast('b', shape=[2, 1, 3, 8]))
        self.assertEqual(len(b), 2)
        self.assertEqual(bytes(b), b'01234567456789ab89abcdefcdef012301234567456789ab')
        self.assertEqual(b.size, (2, 1, 3, 8))
        self.assertEqual(b.stride, (24, 24, 8, 1))
        self.assertEqual(b[1, 0, 2, 3], '7')
        self.assertEqual(b[1][0][2][3], '7')

    def test_init_buffer_mutable(self):
        a = bytearray(b'01234567'
                      b'456789ab'
                      b'89abcdef'

                      b'cdef0123'
                      b'01234567'
                      b'456789ab')
        b = containers.MutableStridedArrayView4D(memoryview(a).cast('b', shape=[2, 1, 3, 8]))
        b[0, 0, 0, 7] = '!'
        b[0, 0, 1, 7] = '!'
        b[0, 0, 2, 7] = '!'
        b[1, 0, 0, 7] = '!'
        b[1, 0, 1, 7] = '!'
        b[1, 0, 2, 7] = '!'
        self.assertEqual(b[1][0][1][7], '!')
        self.assertEqual(bytes(b), b'0123456!'
                                   b'456789a!'
                                   b'89abcde!'

                                   b'cdef012!'
                                   b'0123456!'
                                   b'456789a!')

    def test_ops(self):
        a = (b'01234567'
             b'456789ab'
             b'89abcdef'

             b'cdef0123'
             b'01234567'
             b'456789ab')
        v = memoryview(a).cast('b', shape=[2, 1, 3, 8])

        b = containers.StridedArrayView4D(v).transposed(0, 2).flipped(0)
        self.assertEqual(b.size, (3, 1, 2, 8))
        self.assertEqual(b.stride, (-8, 24, 24, 1))
        self.assertEqual(bytes(b), b'89abcdef456789ab456789ab0123456701234567cdef0123')

        c = containers.StridedArrayView4D(v).transposed(3, 0).flipped(2)
        self.assertEqual(c.size, (8, 1, 3, 2))
        self.assertEqual(c.stride, (1, 24, -8, 24))
        self.assertEqual(bytes(c), b'84400c95511da6622eb7733fc88440d99551eaa662fbb773')

        d = containers.StridedArrayView4D(v).transposed(2, 3)[0:1, :, 3:5, :].broadcasted(0, 5)
        self.assertEqual(d.size, (5, 1, 2, 3))
        self.assertEqual(d.stride, (0, 24, 1, 8))
        self.assertEqual(bytes(d), b'37b48c37b48c37b48c37b48c37b48c')

        e = containers.StridedArrayView4D(v)[:, :, 1:2, 3:4].flipped(3).broadcasted(2, 2)
        self.assertEqual(e.size, (2, 1, 2, 1))
        self.assertEqual(e.stride, (24, 24, 0, -1))
        self.assertEqual(bytes(e), b'7733')

        f = containers.StridedArrayView4D(v)[:, :, :, 0:1].broadcasted(3, 5)
        self.assertEqual(f.size, (2, 1, 3, 5))
        self.assertEqual(f.stride, (24, 24, 8, 0))
        self.assertEqual(bytes(f), b'000004444488888ccccc0000044444')
