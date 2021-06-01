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

.. py:class:: magnum.trade.ImageData1D

    See :ref:`ImageData2D` for more information.

.. py:class:: magnum.trade.ImageData2D

    Similarly to :ref:`Image2D`, holds its own data buffer, thus doesn't have
    an equivalent to :ref:`ImageView2D.owner`. Implicitly convertible to
    :ref:`ImageView2D` / :ref:`MutableImageView2D`, so all APIs consuming image
    views work with this type as well.

.. py:class:: magnum.trade.ImageData3D

    See :ref:`ImageData2D` for more information.

.. py:property:: magnum.trade.ImageData1D.storage
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`
.. py:property:: magnum.trade.ImageData2D.storage
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`
.. py:property:: magnum.trade.ImageData3D.storage
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`

.. py:property:: magnum.trade.ImageData1D.format
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`
.. py:property:: magnum.trade.ImageData2D.format
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`
.. py:property:: magnum.trade.ImageData3D.format
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`

.. py:property:: magnum.trade.ImageData1D.pixel_size
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`
.. py:property:: magnum.trade.ImageData2D.pixel_size
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`
.. py:property:: magnum.trade.ImageData3D.pixel_size
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`

.. py:property:: magnum.trade.ImageData1D.pixels
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`
.. py:property:: magnum.trade.ImageData2D.pixels
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`
.. py:property:: magnum.trade.ImageData3D.pixels
    :raise AttributeError: If :ref:`is_compressed` is :py:`True`

.. py:class:: magnum.trade.ImporterManager
    :summary: Manager for :ref:`AbstractImporter` plugin instances

    Each plugin returned by :ref:`instantiate()` or :ref:`load_and_instantiate()`
    references its owning :ref:`ImporterManager` through
    :ref:`AbstractImporter.manager`, ensuring the manager is not deleted before
    the plugin instances are.

.. py:class:: magnum.trade.AbstractImporter

    Similarly to C++, importer plugins are loaded through :ref:`ImporterManager`:

    ..
        >>> from magnum import trade

    .. code:: py

        >>> manager = trade.ImporterManager()
        >>> importer = manager.load_and_instantiate('PngImporter')

    Unlike C++, errors in both API usage and file parsing are reported by
    raising an exception. See particular function documentation for detailed
    behavior.

.. py:function:: magnum.trade.AbstractImporter.open_data
    :raise RuntimeError: If file opening fails

.. py:function:: magnum.trade.AbstractImporter.open_file
    :raise RuntimeError: If file opening fails

.. py:property:: magnum.trade.AbstractImporter.mesh_count
    :raise RuntimeError: If no file is opened
.. py:function:: magnum.trade.AbstractImporter.mesh_level_count
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than :ref:`mesh_count`
.. py:function:: magnum.trade.AbstractImporter.mesh_for_name
    :raise RuntimeError: If no file is opened
.. py:function:: magnum.trade.AbstractImporter.mesh_name
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than :ref:`mesh_count`
.. py:function:: magnum.trade.AbstractImporter.mesh
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than :ref:`mesh_count`

.. py:property:: magnum.trade.AbstractImporter.image1d_count
    :raise RuntimeError: If no file is opened
.. py:property:: magnum.trade.AbstractImporter.image2d_count
    :raise RuntimeError: If no file is opened
.. py:property:: magnum.trade.AbstractImporter.image3d_count
    :raise RuntimeError: If no file is opened

.. py:function:: magnum.trade.AbstractImporter.image1d_level_count
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image1d_count`
.. py:function:: magnum.trade.AbstractImporter.image2d_level_count
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image2d_count`
.. py:function:: magnum.trade.AbstractImporter.image3d_level_count
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image3d_count`

.. py:function:: magnum.trade.AbstractImporter.image1d_for_name
    :raise RuntimeError: If no file is opened
.. py:function:: magnum.trade.AbstractImporter.image2d_for_name
    :raise RuntimeError: If no file is opened
.. py:function:: magnum.trade.AbstractImporter.image3d_for_name
    :raise RuntimeError: If no file is opened

.. py:function:: magnum.trade.AbstractImporter.image1d_name
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image1d_count`
.. py:function:: magnum.trade.AbstractImporter.image2d_name
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image2d_count`
.. py:function:: magnum.trade.AbstractImporter.image3d_name
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image3d_count`

.. py:function:: magnum.trade.AbstractImporter.image1d
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image1d_count`
.. py:function:: magnum.trade.AbstractImporter.image2d
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image2d_count`
.. py:function:: magnum.trade.AbstractImporter.image3d
    :raise RuntimeError: If no file is opened
    :raise ValueError: If :p:`id` is negative or not less than
        :ref:`image3d_count`
