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

# setUpModule gets called before everything else, skipping if GL tests can't
# be run
from . import GLTestCase, setUpModule

import magnum
from magnum import *
from magnum import gl

class AbstractShaderProgram(GLTestCase):
    def test(self):
        a = gl.AbstractShaderProgram()

        if magnum.TARGET_GLES2:
            vert = gl.Shader(gl.Version.GLES200, gl.Shader.Type.VERTEX)
        elif magnum.TARGET_GLES:
            vert = gl.Shader(gl.Version.GLES300, gl.Shader.Type.VERTEX)
        else:
            vert = gl.Shader(gl.Version.GL300, gl.Shader.Type.VERTEX)
        if magnum.TARGET_GLES2:
            vert.add_source("""
attribute lowp vec4 position;
uniform lowp mat4 transformationProjectionMatrix;

void main() {
    gl_Position = transformationProjectionMatrix*position;
}
            """.strip())
        else:
            vert.add_source("""
in lowp vec4 position;
uniform lowp mat4 transformationProjectionMatrix;

void main() {
    gl_Position = transformationProjectionMatrix*position;
}
            """.strip())

        vert.compile()
        a.attach_shader(vert)

        if magnum.TARGET_GLES2:
            frag = gl.Shader(gl.Version.GLES200, gl.Shader.Type.FRAGMENT)
        elif magnum.TARGET_GLES:
            frag = gl.Shader(gl.Version.GLES300, gl.Shader.Type.FRAGMENT)
        else:
            frag = gl.Shader(gl.Version.GL300, gl.Shader.Type.FRAGMENT)
        if magnum.TARGET_GLES2:
            frag.add_source("""
void main() {
    gl_FragColor = vec4(0.0);
}
            """.strip())
        else:
            frag.add_source("""
out lowp vec4 color;

void main() {
    color = vec4(0.0);
}
            """.strip())
        frag.compile()
        a.attach_shader(frag)

        a.bind_attribute_location(0, "position")
        a.link()
        location = a.uniform_location("transformationProjectionMatrix")
        self.assertGreaterEqual(location, 0)
        a.set_uniform(location, Matrix4())

    def test_link_fail(self):
        a = gl.AbstractShaderProgram()
        # Link of an empty shader will always fail
        with self.assertRaisesRegex(RuntimeError, "linking failed"):
            a.link()

    def test_uniform_fail(self):
        a = gl.AbstractShaderProgram()
        with self.assertRaisesRegex(ValueError, "location of uniform 'nonexistent' cannot be retrieved"):
            a.uniform_location("nonexistent")
        # Asking for uniform on a non-linked program is an error, eat it so the
        # setup/teardown checks don't complain
        self.assertEqual(gl.Renderer.error, gl.Renderer.Error.INVALID_OPERATION)

        if not magnum.TARGET_GLES2:
            with self.assertRaisesRegex(ValueError, "index of uniform block 'nonexistent' cannot be retrieved"):
                a.uniform_block_index("nonexistent")

class Buffer(GLTestCase):
    def test_init(self):
        a = gl.Buffer()
        self.assertNotEqual(a.id, 0)
        self.assertEqual(a.target_hint, gl.Buffer.TargetHint.ARRAY)

        b = gl.Buffer(gl.Buffer.TargetHint.ELEMENT_ARRAY)
        self.assertNotEqual(b.id, 0)
        self.assertEqual(b.target_hint, gl.Buffer.TargetHint.ELEMENT_ARRAY)

    def test_set_data(self):
        a = gl.Buffer()
        a.set_data(b'hello', gl.BufferUsage.STATIC_DRAW)

    def test_set_data_array(self):
        a = gl.Buffer()
        a.set_data(array.array('f', [0.5, 1.2]))

class Context(GLTestCase):
    def test(self):
        self.assertTrue(gl.Context.has_current)

        # Retrieving the context should hold down the app with one more
        # reference
        app_refcount = sys.getrefcount(self.app)
        current = gl.Context.current
        self.assertEqual(sys.getrefcount(self.app), app_refcount + 1)
        self.assertEqual(current.owner, self.app)

        # Some properties
        self.assertGreater(len(current.renderer_string), 0)
        self.assertGreater(len(current.extension_strings), 0)

        # Interestingly enough, this "just works". I thought I would need to do
        # something extra but apparently not
        current2 = gl.Context.current
        self.assertTrue(id(current) == id(current2))

        del current, current2
        self.assertEqual(sys.getrefcount(self.app), app_refcount)

class DefaultFramebuffer(GLTestCase):
    def test(self):
        # Using it should not crash, leak or cause double-free issues
        self.assertTrue(gl.default_framebuffer is not None)

class Framebuffer(GLTestCase):
    def test(self):
        framebuffer = gl.Framebuffer(((0, 0), (4, 4)))
        self.assertNotEqual(framebuffer.id, 0)
        self.assertEqual(len(framebuffer.attachments), 0)

    def test_attach(self):
        renderbuffer = gl.Renderbuffer()
        renderbuffer.set_storage(gl.RenderbufferFormat.RGBA8, (4, 4))
        renderbuffer_refcount = sys.getrefcount(renderbuffer)

        framebuffer = gl.Framebuffer(((0, 0), (4, 4)))
        framebuffer.attach_renderbuffer(gl.Framebuffer.ColorAttachment(0), renderbuffer)
        self.assertEqual(len(framebuffer.attachments), 1)
        self.assertIs(framebuffer.attachments[0], renderbuffer)
        self.assertEqual(sys.getrefcount(renderbuffer), renderbuffer_refcount + 1)

    def test_read_image(self):
        renderbuffer = gl.Renderbuffer()
        renderbuffer.set_storage(gl.RenderbufferFormat.RGBA8, (4, 4))

        framebuffer = gl.Framebuffer(((0, 0), (4, 4)))
        framebuffer.attach_renderbuffer(gl.Framebuffer.ColorAttachment(0), renderbuffer)

        gl.Renderer.clear_color = Color4(1.0, 0.5, 0.75)
        framebuffer.clear(gl.FramebufferClear.COLOR)

        a = Image2D(PixelFormat.RGBA8_UNORM)
        framebuffer.read(Range2Di.from_size((1, 1), (2, 2)), a)
        self.assertEqual(a.size, Vector2i(2, 2))

        # This tests Image internals because this is the only way how to get a
        # non-empty Image ATM (sorry)
        # TODO: remove once Image can be created non-empty
        a_refcount = sys.getrefcount(a)

        data = a.data
        self.assertIs(data.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        del data
        self.assertEqual(sys.getrefcount(a), a_refcount)

        pixels = a.pixels
        self.assertIs(pixels.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)
        self.assertEqual(ord(a.pixels[0, 0, 0]), 0xff)
        self.assertEqual(ord(a.pixels[0, 1, 1]), 0x80)
        self.assertEqual(ord(a.pixels[1, 0, 2]), 0xbf)

        del pixels
        self.assertEqual(sys.getrefcount(a), a_refcount)

        view = ImageView2D(a)
        self.assertEqual(view.size, (2, 2))
        self.assertIs(view.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        del view
        self.assertEqual(sys.getrefcount(a), a_refcount)

        mview = MutableImageView2D(a)
        self.assertEqual(mview.size, (2, 2))
        self.assertIs(mview.owner, a)
        self.assertEqual(sys.getrefcount(a), a_refcount + 1)

        del mview
        self.assertEqual(sys.getrefcount(a), a_refcount)

    def test_read_view(self):
        renderbuffer = gl.Renderbuffer()
        renderbuffer.set_storage(gl.RenderbufferFormat.RGBA8, (4, 4))

        framebuffer = gl.Framebuffer(((0, 0), (4, 4)))
        framebuffer.attach_renderbuffer(gl.Framebuffer.ColorAttachment(0), renderbuffer)

        gl.Renderer.clear_color = Color4(1.0, 0.5, 0.75)
        framebuffer.clear(gl.FramebufferClear.COLOR)

        a = MutableImageView2D(PixelFormat.RGBA8_UNORM, (2, 2), bytearray(16))
        framebuffer.read(Range2Di.from_size((1, 1), (2, 2)), a)
        self.assertEqual(a.size, Vector2i(2, 2))
        self.assertEqual(ord(a.pixels[0, 0, 0]), 0xff)
        self.assertEqual(ord(a.pixels[0, 1, 1]), 0x80)
        self.assertEqual(ord(a.pixels[1, 0, 2]), 0xbf)

class Mesh(GLTestCase):
    def test_init(self):
        a = gl.Mesh()
        b = gl.Mesh(gl.MeshPrimitive.LINE_LOOP)
        c = gl.Mesh(MeshPrimitive.LINES)
        self.assertNotEqual(a.id, 0)
        self.assertNotEqual(b.id, 0)
        self.assertNotEqual(c.id, 0)
        self.assertEqual(a.primitive, gl.MeshPrimitive.TRIANGLES)
        self.assertEqual(b.primitive, gl.MeshPrimitive.LINE_LOOP)
        self.assertEqual(c.primitive, gl.MeshPrimitive.LINES)

    def test_set_primitive(self):
        a = gl.Mesh()

        a.primitive = gl.MeshPrimitive.TRIANGLE_STRIP
        self.assertEqual(a.primitive, gl.MeshPrimitive.TRIANGLE_STRIP)

        a.primitive = MeshPrimitive.POINTS
        self.assertEqual(a.primitive, gl.MeshPrimitive.POINTS)

    def test_set_primitive_invalid(self):
        a = gl.Mesh()

        with self.assertRaisesRegex(TypeError, "expected MeshPrimitive or gl.MeshPrimitive, got <class 'str'>"):
            a.primitive = "ahaha"

    def test_set_count(self):
        a = gl.Mesh()
        a.count = 15
        self.assertEqual(a.count, 15)

    def test_add_buffer(self):
        buffer = gl.Buffer()
        buffer_refcount = sys.getrefcount(buffer)

        # Adding a buffer to the mesh should increase its ref count
        mesh = gl.Mesh()
        mesh.add_vertex_buffer(buffer, 0, 8, gl.Attribute(gl.Attribute.Kind.GENERIC, 2, gl.Attribute.Components.TWO, gl.Attribute.DataType.FLOAT))
        self.assertEqual(len(mesh.buffers), 1)
        self.assertIs(mesh.buffers[0], buffer)
        self.assertEqual(sys.getrefcount(buffer), buffer_refcount + 1)

        # Deleting the mesh should decrease it again
        del mesh
        self.assertEqual(sys.getrefcount(buffer), buffer_refcount)

class Renderbuffer(GLTestCase):
    def test_init(self):
        renderbuffer = gl.Renderbuffer()
        renderbuffer.set_storage(gl.RenderbufferFormat.RGBA8, (16, 16))
        self.assertNotEqual(renderbuffer.id, 0)

class Renderer(GLTestCase):
    def test_feature(self):
        gl.Renderer.enable(gl.Renderer.Feature.DEPTH_TEST)
        gl.Renderer.disable(gl.Renderer.Feature.FACE_CULLING)
        gl.Renderer.set_feature(gl.Renderer.Feature.STENCIL_TEST, True)

    def test_error(self):
        self.assertEqual(gl.Renderer.error, gl.Renderer.Error.NO_ERROR)

    def test_blend(self):
        gl.Renderer.set_blend_equation(gl.Renderer.BlendEquation.ADD)
        gl.Renderer.set_blend_function(gl.Renderer.BlendFunction.SOURCE_ALPHA, gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA)

class Shader(GLTestCase):
    def test(self):
        if magnum.TARGET_GLES2:
            a = gl.Shader(gl.Version.GLES200, gl.Shader.Type.VERTEX)
        elif magnum.TARGET_GLES:
            a = gl.Shader(gl.Version.GLES300, gl.Shader.Type.VERTEX)
        else:
            a = gl.Shader(gl.Version.GL300, gl.Shader.Type.VERTEX)
        a.add_source("""
        void main() {
            gl_Position = vec4(0.0);
        }
        """)
        a.compile()

    def test_compile_fail(self):
        if magnum.TARGET_GLES2:
            a = gl.Shader(gl.Version.GLES200, gl.Shader.Type.VERTEX)
        elif magnum.TARGET_GLES:
            a = gl.Shader(gl.Version.GLES300, gl.Shader.Type.VERTEX)
        else:
            a = gl.Shader(gl.Version.GL300, gl.Shader.Type.VERTEX)
        a.add_source("error!!!!")
        with self.assertRaisesRegex(RuntimeError, "compilation failed"):
            a.compile()

class AbstractTexture(GLTestCase):
    def test_unbind(self):
        gl.AbstractTexture.unbind(3)

class Texture(GLTestCase):
    def test_minification_filter(self):
        a = gl.Texture2D()

        # Both generic and GL value should work
        a.minification_filter = gl.SamplerFilter.LINEAR
        a.minification_filter = SamplerFilter.LINEAR

        # A tuple as well -- any combination
        a.minification_filter = (gl.SamplerFilter.LINEAR, gl.SamplerMipmap.LINEAR)
        a.minification_filter = (gl.SamplerFilter.LINEAR, SamplerMipmap.LINEAR)
        a.minification_filter = (SamplerFilter.LINEAR, gl.SamplerMipmap.LINEAR)
        a.minification_filter = (SamplerFilter.LINEAR, SamplerMipmap.LINEAR)

    def test_minification_filter_invalid(self):
        a = gl.Texture2D()

        with self.assertRaisesRegex(TypeError, "expected SamplerFilter, gl.SamplerFilter or a two-element tuple"):
            a.minification_filter = 3
        with self.assertRaisesRegex(TypeError, "expected a tuple with SamplerFilter or gl.SamplerFilter as the first element"):
            a.minification_filter = (3, SamplerMipmap.BASE)
        with self.assertRaisesRegex(TypeError, "expected a tuple with SamplerMipmap or gl.SamplerMipmap as the second element"):
            a.minification_filter = (SamplerFilter.NEAREST, 3)
        with self.assertRaisesRegex(TypeError, "expected a tuple with SamplerFilter or gl.SamplerFilter as the first element"):
            a.minification_filter = (3, SamplerMipmap.BASE)

        # List doesn't work ATM, sorry
        with self.assertRaisesRegex(TypeError, "expected SamplerFilter, gl.SamplerFilter or a two-element tuple"):
            a.minification_filter = [gl.SamplerFilter.LINEAR, gl.SamplerMipmap.LINEAR]

    def test_magnification_filter(self):
        a = gl.Texture2D()

        # Both generic and GL value should work
        a.magnification_filter = gl.SamplerFilter.LINEAR
        a.magnification_filter = SamplerFilter.LINEAR

    def test_magnification_filter_invalid(self):
        a = gl.Texture2D()

        with self.assertRaisesRegex(TypeError, "expected SamplerFilter or gl.SamplerFilter"):
            a.magnification_filter = 3

    def test_wrapping(self):
        a = gl.Texture2D()

        # Both generic and GL value should work
        a.wrapping = gl.SamplerWrapping.REPEAT
        a.wrapping = SamplerWrapping.REPEAT

    def test_wrapping_invalid(self):
        a = gl.Texture2D()

        with self.assertRaisesRegex(TypeError, "expected SamplerWrapping or gl.SamplerWrapping"):
            a.wrapping = 3

    # TODO: re-enable on ES when extensions can be checked
    @unittest.skipUnless(not magnum.TARGET_WEBGL and not magnum.TARGET_GLES, "border color is not available on WebGL and requires an extension on ES which we can't check")
    def test_border_color(self):
        a = gl.Texture2D()

        # Both three- and four-component should work
        a.border_color = Color3()
        a.border_color = Color4()

        if not magnum.TARGET_GLES2:
            a.border_color = Vector4ui()
            a.border_color = Vector4i()

    # TODO: re-enable on ES when extensions can be checked
    @unittest.skipUnless(not magnum.TARGET_WEBGL and not magnum.TARGET_GLES, "border color is not available on WebGL and requires an extension on ES which we can't check")
    def test_border_color_invalid(self):
        a = gl.Texture2D()

        if not magnum.TARGET_GLES2:
            with self.assertRaisesRegex(TypeError, "expected Color3, Color4, Vector4ui or Vector4i"):
                a.border_color = 3
        else:
            # On ES2 this is handled by pybind itself, so the message is
            # different
            with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
                a.border_color = 3

            # This should raise a type error on ES2, as only floats are
            # supported
            with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
                a.border_color = Vector4ui()

    def test_set_image(self):
        a = gl.Texture2D()
        a.set_image(level=0, internal_format=gl.TextureFormat.RGBA8,
            image=ImageView2D(PixelFormat.RGBA8_UNORM, Vector2i(16)))

    def test_set_storage_subimage(self):
        a = gl.Texture2D()
        a.set_storage(levels=5, internal_format=gl.TextureFormat.RGBA8,
            size=Vector2i(16))
        a.set_sub_image(0, Vector2i(), ImageView2D(PixelFormat.RGBA8_UNORM, Vector2i(16)))
        a.generate_mipmap()

        if not magnum.TARGET_GLES:
            # This is in ES3.2 too, but we don't have a way to check for
            # extensions / version yet
            self.assertEqual(a.image_size(0), Vector2i(16, 16))
