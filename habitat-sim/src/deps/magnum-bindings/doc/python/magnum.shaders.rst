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

.. py:class:: magnum.shaders.FlatGL2D
    :data POSITION: Vertex position
    :data TEXTURE_COORDINATES: 2D texture coordinates
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:class:: magnum.shaders.FlatGL3D
    :data POSITION: Vertex position
    :data TEXTURE_COORDINATES: 2D texture coordinates
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:property:: magnum.shaders.FlatGL2D.alpha_mask
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.ALPHA_MASK`
.. py:property:: magnum.shaders.FlatGL3D.alpha_mask
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.ALPHA_MASK`

.. py:function:: magnum.shaders.FlatGL2D.bind_texture
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.TEXTURED`
.. py:function:: magnum.shaders.FlatGL3D.bind_texture
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.TEXTURED`

.. py:class:: magnum.shaders.VertexColorGL2D
    :data POSITION: Vertex position
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:class:: magnum.shaders.VertexColorGL3D
    :data POSITION: Vertex position
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:class:: magnum.shaders.PhongGL
    :data POSITION: Vertex position
    :data NORMAL: Normal direction
    :data TANGENT: Tangent direction
    :data TANGENT4: Tangent direction with a bitangent sign
    :data BITANGENT: Bitangent direction
    :data TEXTURE_COORDINATES: 2D texture coordinates
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color
    :data TRANSFORMATION_MATRIX: (Instanced) transformation matrix
    :data NORMAL_MATRIX: (Instanced) normal matrix
    :data TEXTURE_OFFSET: (Instanced) texture offset

.. py:property:: magnum.shaders.PhongGL.normal_texture_scale
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.NORMAL_TEXTURE`
.. py:property:: magnum.shaders.PhongGL.alpha_mask
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.ALPHA_MASK`
.. py:property:: magnum.shaders.PhongGL.texture_matrix
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.TEXTURE_TRANSFORMATION`
.. py:property:: magnum.shaders.PhongGL.light_positions
    :raise ValueError: If list length is different from :ref:`light_count`
.. py:property:: magnum.shaders.PhongGL.light_colors
    :raise ValueError: If list length is different from :ref:`light_count`
.. py:property:: magnum.shaders.PhongGL.light_ranges
    :raise ValueError: If list length is different from :ref:`light_count`

.. py:function:: magnum.shaders.PhongGL.bind_ambient_texture
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.AMBIENT_TEXTURE`
.. py:function:: magnum.shaders.PhongGL.bind_diffuse_texture
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.DIFFUSE_TEXTURE`
.. py:function:: magnum.shaders.PhongGL.bind_specular_texture
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.SPECULAR_TEXTURE`
.. py:function:: magnum.shaders.PhongGL.bind_normal_texture
    :raise AttributeError: If the shader was not created with
        :ref:`Flags.NORMAL_TEXTURE`
.. py:function:: magnum.shaders.PhongGL.bind_textures
    :raise AttributeError: If the shader was not created with any of
        :ref:`Flags.AMBIENT_TEXTURE`, :ref:`Flags.DIFFUSE_TEXTURE`,
        :ref:`Flags.SPECULAR_TEXTURE` or :ref:`Flags.NORMAL_TEXTURE`
