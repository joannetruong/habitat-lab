..
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
..

.. doctest setup
    >>> from corrade import containers

.. py:class:: corrade.containers.ArrayView

    Provides one-dimensional tightly packed view on a memory range. Convertible
    both from and to Python objects supporting the Buffer Protocol, with one
    dimension and stride of :py:`1`. See :ref:`StridedArrayView1D` and others
    for more generic views. :ref:`ArrayView` is immutable, see
    :ref:`MutableArrayView` for the mutable alterantive. All slicing operations
    are supported, specifying a non-trivial stride will return
    :ref:`StridedArrayView1D` instead of :ref:`ArrayView`. Example usage:

    .. code:: pycon

        >>> a = b'hello'
        >>> b = containers.ArrayView(a)
        >>> b[2]
        'l'
        >>> bytes(b[1:4])
        b'ell'

    `Memory ownership and reference counting`_
    ==========================================

    Unlike in C++, the view keeps a reference to the original memory owner
    object in the :ref:`owner` field, meaning that calling :py:`del` on the
    original object will *not* invalidate the view. Slicing a view creates a
    new view referencing the same original object, without any dependency on
    the previous view. That means a long chained slicing operation will not cause
    increased memory usage.

    .. code:: pycon

        >>> b.owner is a
        True
        >>> b[1:4][:-1].owner is a
        True

    The :py:`owner` is :py:`None` if the view is empty.

    `Comparison to Python's memoryview`_
    ====================================

    The :ref:`ArrayView` class is equivalent to one-dimensional
    :ref:`memoryview` with a stride of :py:`1`. For multiple dimensions and
    non-trivial strides, :ref:`StridedArrayView1D` and friends provide a
    superset of :ref:`memoryview` features.

.. py:class:: corrade.containers.MutableArrayView

    Equivalent to :ref:`ArrayView`, but implementing :ref:`__setitem__()` as
    well.

.. py:class:: corrade.containers.StridedArrayView1D

    Provides one-dimensional read-only view on a memory range with custom
    stride values. See :ref:`StridedArrayView2D`, :ref:`StridedArrayView3D`,
    :ref:`MutableStridedArrayView1D` and others for multi-dimensional and
    mutable equivalents.

    `Comparison to Python's memoryview`_
    ====================================

    The :ref:`StridedArrayView1D` and its multi-dimensional variants are
    equivalent to any :ref:`memoryview`, but additionally supporting
    multi-dimensional slicing as well (which raises :ref:`NotImplementedError`
    in Py3.7 :ref:`memoryview`).

.. py:class:: corrade.containers.MutableStridedArrayView1D

    Equivalent to :ref:`StridedArrayView1D`, but implementing
    :ref:`__setitem__()` as well.

.. py:class:: corrade.containers.StridedArrayView2D

    See :ref:`StridedArrayView1D` for more information.

.. py:class:: corrade.containers.MutableStridedArrayView2D

    See :ref:`StridedArrayView1D` and :ref:`MutableStridedArrayView1D` for more
    information.

.. py:class:: corrade.containers.StridedArrayView3D

    See :ref:`StridedArrayView1D` for more information.

.. py:class:: corrade.containers.MutableStridedArrayView3D

    See :ref:`StridedArrayView1D` and :ref:`MutableStridedArrayView1D` for more
    information.
