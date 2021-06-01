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

.. py:module:: magnum.gl
    :data default_framebuffer: Default framebuffer instance

    :TODO: remove this line once m.css stops ignoring first caption on a page

    `NoCreate constructors`_
    ========================

    .. TODO: link to NoCreate once m.dox handles variables properly

    Compared to C++, the Python APIs don't have an alternative to the
    :dox:`NoCreate <NoCreateT>` constructor tag. In C++ these make it possible
    to have class members initialized before a GL context is present, but in
    Python there's no such limitation so these don't make sense.

.. py:function:: magnum.gl.AbstractShaderProgram.link
    :raise RuntimeError: If linking fails
.. py:function:: magnum.gl.AbstractShaderProgram.uniform_location
    :raise ValueError: If there's no uniform of that name
.. py:function:: magnum.gl.AbstractShaderProgram.uniform_block_index
    :raise ValueError: If there's no uniform block of that name
.. py:function:: magnum.gl.Shader.compile
    :raise RuntimeError: If compilation fails

.. py:class:: magnum.gl.Mesh

    :TODO: remove this line once m.css stops ignoring first caption on a page

    `Buffer ownership and reference counting`_
    ==========================================

    Unlike in C++, where a :dox:`GL::Buffer` is either :dox:`std::move()`\ d
    into the mesh or referenced externally (with the user being responsible for
    its lifetime), the :ref:`gl.Mesh` object keeps references to all buffers
    added to it.

.. py:property:: magnum.gl.Mesh.primitive

    While querying this property will always give back a :ref:`gl.MeshPrimitive`,
    this property can be set using either :ref:`magnum.MeshPrimitive` or
    :ref:`gl.MeshPrimitive`, similarly to how the overloaded
    :dox:`GL::Mesh::setPrimitive()` works.

.. py:property:: magnum.gl.Texture1D.minification_filter

    See :ref:`Texture2D.minification_filter` for more information.

.. py:property:: magnum.gl.Texture2D.minification_filter

    This property accepts either a tuple of :ref:`magnum.SamplerFilter` /
    :ref:`gl.SamplerFilter` and :ref:`magnum.SamplerMipmap` /
    :ref:`gl.SamplerMipmap` values or just :ref:`magnum.SamplerFilter` / :ref:`gl.SamplerFilter` alone in which case :ref:`gl.SamplerMipmap.BASE`
    will be used implicitly; similarly to how the overloaded
    :dox:`GL::Texture::setMinificationFilter()` works.

.. py:property:: magnum.gl.Texture3D.minification_filter

    See :ref:`Texture2D.minification_filter` for more information.

.. py:property:: magnum.gl.Texture1D.magnification_filter

    See :ref:`Texture2D.magnification_filter` for more information.

.. py:property:: magnum.gl.Texture2D.magnification_filter

    This property accepts either :ref:`magnum.SamplerFilter` or
    :ref:`gl.SamplerFilter`, similarly to how the overloaded
    :dox:`GL::Texture::setMagnificationFilter()`
    works.

.. py:property:: magnum.gl.Texture3D.magnification_filter

    See :ref:`Texture2D.magnification_filter` for more information.
