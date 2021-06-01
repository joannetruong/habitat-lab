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

# setUpModule gets called before everything else, skipping if GL tests can't
# be run
from . import GLTestCase, setUpModule

from magnum import *
from magnum import gl, shaders

class Flat(GLTestCase):
    def test_init(self):
        a = shaders.Flat3D()
        self.assertEqual(a.flags, shaders.Flat3D.Flags.NONE)

        b = shaders.Flat3D(shaders.Flat3D.Flags.TEXTURED|shaders.Flat3D.Flags.ALPHA_MASK)
        self.assertEqual(b.flags, shaders.Flat3D.Flags.TEXTURED|shaders.Flat3D.Flags.ALPHA_MASK)

    def test_uniforms_bindings(self):
        a = shaders.Flat3D(shaders.Flat3D.Flags.TEXTURED|shaders.Flat3D.Flags.ALPHA_MASK)
        a.color = (0.5, 1.0, 0.9)
        a.transformation_projection_matrix = Matrix4.translation(Vector3.x_axis())
        a.alpha_mask = 0.3

        texture = gl.Texture2D()
        texture.set_storage(1, gl.TextureFormat.RGBA8, Vector2i(8))
        a.bind_texture(texture)

    def test_uniforms_bindings_errors(self):
        a = shaders.Flat2D()
        with self.assertRaisesRegex(AttributeError, "the shader was not created with alpha mask enabled"):
            a.alpha_mask = 0.3

        texture = gl.Texture2D()
        with self.assertRaisesRegex(AttributeError, "the shader was not created with texturing enabled"):
            a.bind_texture(texture)

class VertexColor(GLTestCase):
    def test_init(self):
        a = shaders.VertexColor2D()
        b = shaders.VertexColor3D()

    def test_uniforms(self):
        a = shaders.VertexColor2D()
        a.transformation_projection_matrix = (
            Matrix3.translation(Vector2.x_axis())@
            Matrix3.rotation(Deg(35.0)))

class Phong(GLTestCase):
    def test_init(self):
        a = shaders.Phong()
        self.assertEqual(a.flags, shaders.Phong.Flags.NONE)
        self.assertEqual(a.light_count, 1)

        b = shaders.Phong(shaders.Phong.Flags.DIFFUSE_TEXTURE|shaders.Phong.Flags.ALPHA_MASK)
        self.assertEqual(b.flags, shaders.Phong.Flags.DIFFUSE_TEXTURE|shaders.Phong.Flags.ALPHA_MASK)
        self.assertEqual(b.light_count, 1)

        c = shaders.Phong(shaders.Phong.Flags.NONE, 3)
        self.assertEqual(c.flags, shaders.Phong.Flags.NONE)
        self.assertEqual(c.light_count, 3)

    def test_uniforms_bindings(self):
        a = shaders.Phong(shaders.Phong.Flags.ALPHA_MASK|shaders.Phong.Flags.AMBIENT_TEXTURE|shaders.Phong.Flags.DIFFUSE_TEXTURE|shaders.Phong.Flags.SPECULAR_TEXTURE|shaders.Phong.Flags.NORMAL_TEXTURE|shaders.Phong.Flags.TEXTURE_TRANSFORMATION, 2)
        a.diffuse_color = (0.5, 1.0, 0.9)
        a.transformation_matrix = Matrix4.translation(Vector3.x_axis())
        a.projection_matrix = Matrix4.zero_init()
        a.light_positions = [(0.5, 1.0, 0.3, 1.0), Vector4()]
        a.light_colors = [Color3(), Color3()]
        a.light_specular_colors = [Color3(), Color3()]
        a.light_ranges = [0.5, 100]
        a.normal_texture_scale = 0.3
        a.alpha_mask = 0.3
        a.texture_matrix = Matrix3()

        texture = gl.Texture2D()
        texture.set_storage(1, gl.TextureFormat.RGBA8, Vector2i(8))
        a.bind_ambient_texture(texture)
        a.bind_diffuse_texture(texture)
        a.bind_specular_texture(texture)
        a.bind_normal_texture(texture)
        a.bind_textures(ambient=texture, diffuse=texture, specular=texture, normal=texture)

    def test_uniforms_bindings_errors(self):
        a = shaders.Phong()
        with self.assertRaisesRegex(AttributeError, "the shader was not created with normal texture enabled"):
            a.normal_texture_scale = 0.3
        with self.assertRaisesRegex(AttributeError, "the shader was not created with alpha mask enabled"):
            a.alpha_mask = 0.3
        with self.assertRaisesRegex(AttributeError, "the shader was not created with texture transformation enabled"):
            a.texture_matrix = Matrix3()
        with self.assertRaisesRegex(ValueError, "expected 1 items but got 0"):
            a.light_positions = []
        with self.assertRaisesRegex(ValueError, "expected 1 items but got 0"):
            a.light_colors = []
        with self.assertRaisesRegex(ValueError, "expected 1 items but got 0"):
            a.light_specular_colors = []
        with self.assertRaisesRegex(ValueError, "expected 1 items but got 0"):
            a.light_ranges = []

        texture = gl.Texture2D()
        with self.assertRaisesRegex(AttributeError, "the shader was not created with ambient texture enabled"):
            a.bind_ambient_texture(texture)
        with self.assertRaisesRegex(AttributeError, "the shader was not created with diffuse texture enabled"):
            a.bind_diffuse_texture(texture)
        with self.assertRaisesRegex(AttributeError, "the shader was not created with specular texture enabled"):
            a.bind_specular_texture(texture)
        with self.assertRaisesRegex(AttributeError, "the shader was not created with normal texture enabled"):
            a.bind_normal_texture(texture)
        with self.assertRaisesRegex(AttributeError, "the shader was not created with any textures enabled"):
            a.bind_textures(diffuse=texture)
