/*
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
*/

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <Corrade/Containers/StridedArrayView.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/Mesh.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/PixelStorage.h>
#include <Magnum/Sampler.h>

#include "Corrade/PythonBindings.h"
#include "Corrade/Containers/PythonBindings.h"
#include "Corrade/Containers/StridedArrayViewPythonBindings.h"
#include "Magnum/PythonBindings.h"

#include "magnum/bootstrap.h"

#ifdef MAGNUM_BUILD_STATIC
#include "magnum/staticconfigure.h"
#endif

namespace py = pybind11;

namespace magnum { namespace {

template<class T> void image(py::class_<T>& c) {
    c
        /* Constructors. Only the ones taking the generic format and *not*
           taking an Array, as Python has no way to "move" it in */
        .def(py::init<const PixelStorage&, PixelFormat>(), "Construct an image placeholder")
        .def(py::init<PixelFormat>(), "Construct an image placeholder")

        /* Properties */
        .def_property_readonly("storage", &T::storage, "Storage of pixel data")
        .def_property_readonly("format", &T::format, "Format of pixel data")
        /** @todo formatExtra() */
        .def_property_readonly("pixel_size", &T::pixelSize, "Pixel size (in bytes)")
        .def_property_readonly("size", [](T& self) {
            return PyDimensionTraits<T::Dimensions, Int>::from(self.size());
        }, "Image size")
        .def_property_readonly("data", [](T& self) {
            return Containers::pyArrayViewHolder(self.data(), self.data() ? py::cast(self) : py::none{});
        }, "Image data")
        .def_property_readonly("pixels", [](T& self) {
            return Containers::pyArrayViewHolder(Containers::PyStridedArrayView<T::Dimensions + 1, const char>{self.pixels()}, self.data() ? py::cast(self) : py::none{});
        }, "View on pixel data");
}

template<class T> void imageView(py::class_<T, PyImageViewHolder<T>>& c) {
    /*
        Missing APIs:

        Type, ErasedType, Dimensions
    */

    py::implicitly_convertible<Image<T::Dimensions>, T>();

    c
        /* Constructors. The variants *not* taking an array view have to be
           first, otherwise things fail on systems that don't have numpy
           installed:

            ===================================================================
            ERROR: test_init_empty (test.test.ImageView)
            -------------------------------------------------------------------
            Traceback (most recent call last):
              File ".../magnum/test/test.py", line 102, in test_init_empty
                b = ImageView2D(storage, PixelFormat.R32F, (8, 8))
            ModuleNotFoundError: No module named 'numpy'

           This is because of the order in which pybind processes arguments ---
           it would first try to match the (PixelFormat, Vector2i, ArrayView)
           variant and *somehow* getting all the way to the third argument,
           where, because ArrayView is marked as implicitly convertible from
           py::array for numpy compatibility, it ends up doing this in numpy.h:

            module m = module::import("numpy.core.multiarray");
            auto c = m.attr("_ARRAY_API");

           Wonderful, isn't it. */
        .def(py::init([](const PixelStorage& storage, PixelFormat format, const typename PyDimensionTraits<T::Dimensions, Int>::VectorType& size) {
            return T{storage, format, size};
        }), "Construct an empty view")
        .def(py::init([](PixelFormat format, const typename PyDimensionTraits<T::Dimensions, Int>::VectorType& size) {
            return T{format, size};
        }), "Construct an empty view")
        .def(py::init([](const PixelStorage& storage, PixelFormat format, const typename PyDimensionTraits<T::Dimensions, Int>::VectorType& size, const Containers::ArrayView<typename T::Type>& data) {
            return pyImageViewHolder(T{storage, format, size, data}, pyObjectHolderFor<Containers::PyArrayViewHolder>(data).owner);
        }), "Constructor")
        .def(py::init([](PixelFormat format, const typename PyDimensionTraits<T::Dimensions, Int>::VectorType& size, const Containers::ArrayView<typename T::Type>& data) {
            return pyImageViewHolder(T{format, size, data}, pyObjectHolderFor<Containers::PyArrayViewHolder>(data).owner);
        }), "Constructor")
        .def(py::init([](Image<T::Dimensions>& image) {
            return pyImageViewHolder(T{image}, image.data() ? py::cast(image) : py::none{});
        }), "Construct a view on an image")
        .def(py::init([](const ImageView<T::Dimensions, typename T::Type>& other) {
            return pyImageViewHolder(ImageView<T::Dimensions, typename T::Type>(other), pyObjectHolderFor<PyImageViewHolder>(other).owner);
        }), "Construct from any type convertible to an image view")

        /* Properties */
        .def_property_readonly("storage", &T::storage, "Storage of pixel data")
        .def_property_readonly("format", &T::format, "Format of pixel data")
        .def_property_readonly("pixel_size", &T::pixelSize, "Pixel size (in bytes)")
        .def_property_readonly("size", [](T& self) {
            return PyDimensionTraits<T::Dimensions, Int>::from(self.size());
        }, "Image size")
        .def_property("data", [](T& self) {
            return Containers::pyArrayViewHolder(self.data(), pyObjectHolderFor<PyImageViewHolder>(self).owner);
        }, [](T& self, const Containers::ArrayView<typename T::Type>& data) {
            self.setData(data);
            pyObjectHolderFor<PyImageViewHolder>(self).owner =
            pyObjectHolderFor<Containers::PyArrayViewHolder>(data).owner;
        }, "Image data")
        .def_property_readonly("pixels", [](T& self) {
            return Containers::pyArrayViewHolder(Containers::PyStridedArrayView<T::Dimensions + 1, typename T::Type>{self.pixels()}, pyObjectHolderFor<PyImageViewHolder>(self).owner);
        }, "View on pixel data")

        .def_property_readonly("owner", [](T& self) {
            return pyObjectHolderFor<PyImageViewHolder>(self).owner;
        }, "Memory owner");
}

template<class T> void imageViewFromMutable(py::class_<T, PyImageViewHolder<T>>& c) {
    py::implicitly_convertible<BasicMutableImageView<T::Dimensions>, T>();

    c
        .def(py::init([](const BasicMutableImageView<T::Dimensions>& other) {
            return pyImageViewHolder(BasicImageView<T::Dimensions>(other), pyObjectHolderFor<PyImageViewHolder>(other).owner);
        }), "Construct from a mutable view");
}

void magnum(py::module_& m) {
    m.attr("BUILD_STATIC") =
        #ifdef MAGNUM_BUILD_STATIC
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_GL") =
        #ifdef MAGNUM_TARGET_GL
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_GLES") =
        #ifdef MAGNUM_TARGET_GLES
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_GLES2") =
        #ifdef MAGNUM_TARGET_GLES2
        true
        #else
        false
        #endif
        ;
    /** @todo do we need TARGET_GLES3? i hope not */
    m.attr("TARGET_WEBGL") =
        #ifdef MAGNUM_TARGET_WEBGL
        true
        #else
        false
        #endif
        ;
    /* TARGET_DESKTOP_GLES, TARGET_HEADLESS skipped as they make sense only
       on native side (affecting what the builtin utilities use), not really in
       Python */
    m.attr("TARGET_VK") =
        #ifdef MAGNUM_TARGET_VK
        true
        #else
        false
        #endif
        ;

    py::enum_<MeshPrimitive>{m, "MeshPrimitive", "Mesh primitive type"}
        .value("POINTS", MeshPrimitive::Points)
        .value("LINES", MeshPrimitive::Lines)
        .value("LINE_LOOP", MeshPrimitive::LineLoop)
        .value("LINE_STRIP", MeshPrimitive::LineStrip)
        .value("TRIANGLES", MeshPrimitive::Triangles)
        .value("TRIANGLE_STRIP", MeshPrimitive::TriangleStrip)
        .value("TRIANGLE_FAN", MeshPrimitive::TriangleFan);

    py::enum_<MeshIndexType>{m, "MeshIndexType", "Mesh index type"}
        .value("UNSIGNED_BYTE", MeshIndexType::UnsignedByte)
        .value("UNSIGNED_SHORT", MeshIndexType::UnsignedShort)
        .value("UNSIGNED_INT", MeshIndexType::UnsignedInt);

    py::enum_<PixelFormat>{m, "PixelFormat", "Format of pixel data"}
        .value("R8_UNORM", PixelFormat::R8Unorm)
        .value("RG8_UNORM", PixelFormat::RG8Unorm)
        .value("RGB8_UNORM", PixelFormat::RGB8Unorm)
        .value("RGBA8_UNORM", PixelFormat::RGBA8Unorm)
        .value("R8_SNORM", PixelFormat::R8Snorm)
        .value("RG8_SNORM", PixelFormat::RG8Snorm)
        .value("RGB8_SNORM", PixelFormat::RGB8Snorm)
        .value("RGBA8_SNORM", PixelFormat::RGBA8Snorm)
        .value("R8_SRGB", PixelFormat::R8Srgb)
        .value("RG8_SRGB", PixelFormat::RG8Srgb)
        .value("RGB8_SRGB", PixelFormat::RGB8Srgb)
        .value("RGBA8_SRGB", PixelFormat::RGBA8Srgb)
        .value("R8UI", PixelFormat::R8UI)
        .value("RG8UI", PixelFormat::RG8UI)
        .value("RGB8UI", PixelFormat::RGB8UI)
        .value("RGBA8UI", PixelFormat::RGBA8UI)
        .value("R8I", PixelFormat::R8I)
        .value("RG8I", PixelFormat::RG8I)
        .value("RGB8I", PixelFormat::RGB8I)
        .value("RGBA8I", PixelFormat::RGBA8I)
        .value("R16_UNORM", PixelFormat::R16Unorm)
        .value("RG16_UNORM", PixelFormat::RG16Unorm)
        .value("RGB16_UNORM", PixelFormat::RGB16Unorm)
        .value("RGBA16_UNORM", PixelFormat::RGBA16Unorm)
        .value("R16_SNORM", PixelFormat::R16Snorm)
        .value("RG16_SNORM", PixelFormat::RG16Snorm)
        .value("RGB16_SNORM", PixelFormat::RGB16Snorm)
        .value("RGBA16_SNORM", PixelFormat::RGBA16Snorm)
        .value("R16UI", PixelFormat::R16UI)
        .value("RG16UI", PixelFormat::RG16UI)
        .value("RGB16UI", PixelFormat::RGB16UI)
        .value("RGBA16UI", PixelFormat::RGBA16UI)
        .value("R16I", PixelFormat::R16I)
        .value("RG16I", PixelFormat::RG16I)
        .value("RGB16I", PixelFormat::RGB16I)
        .value("RGBA16I", PixelFormat::RGBA16I)
        .value("R32UI", PixelFormat::R32UI)
        .value("RG32UI", PixelFormat::RG32UI)
        .value("RGB32UI", PixelFormat::RGB32UI)
        .value("RGBA32UI", PixelFormat::RGBA32UI)
        .value("R32I", PixelFormat::R32I)
        .value("RG32I", PixelFormat::RG32I)
        .value("RGB32I", PixelFormat::RGB32I)
        .value("RGBA32I", PixelFormat::RGBA32I)
        .value("R16F", PixelFormat::R16F)
        .value("RG16F", PixelFormat::RG16F)
        .value("RGB16F", PixelFormat::RGB16F)
        .value("RGBA16F", PixelFormat::RGBA16F)
        .value("R32F", PixelFormat::R32F)
        .value("RG32F", PixelFormat::RG32F)
        .value("RGB32F", PixelFormat::RGB32F)
        .value("RGBA32F", PixelFormat::RGBA32F);

    py::class_<PixelStorage>{m, "PixelStorage", "Pixel storage parameters"}
        .def(py::init(), "Default constructor")

        /* Comparison */
        .def(py::self == py::self, "Equality comparison")
        .def(py::self != py::self, "Non-equality comparison")

        /* Properties */
        .def_property("alignment",
            &PixelStorage::alignment, &PixelStorage::setAlignment, "Row alignment")
        .def_property("row_length",
            &PixelStorage::rowLength, &PixelStorage::setRowLength, "Row length")
        .def_property("image_height",
            &PixelStorage::imageHeight, &PixelStorage::setImageHeight, "Image height")
        .def_property("skip",
            &PixelStorage::skip, &PixelStorage::setSkip, "Pixel, row and image skip");

    py::class_<Image1D> image1D{m, "Image1D", "One-dimensional image"};
    py::class_<Image2D> image2D{m, "Image2D", "Two-dimensional image"};
    py::class_<Image3D> image3D{m, "Image3D", "Three-dimensional image"};
    image(image1D);
    image(image2D);
    image(image3D);

    py::class_<ImageView1D, PyImageViewHolder<ImageView1D>> imageView1D{m, "ImageView1D", "One-dimensional image view"};
    py::class_<ImageView2D, PyImageViewHolder<ImageView2D>> imageView2D{m, "ImageView2D", "Two-dimensional image view"};
    py::class_<ImageView3D, PyImageViewHolder<ImageView3D>> imageView3D{m, "ImageView3D", "Three-dimensional image view"};
    py::class_<MutableImageView1D, PyImageViewHolder<MutableImageView1D>> mutableImageView1D{m, "MutableImageView1D", "One-dimensional mutable image view"};
    py::class_<MutableImageView2D, PyImageViewHolder<MutableImageView2D>> mutableImageView2D{m, "MutableImageView2D", "Two-dimensional mutable image view"};
    py::class_<MutableImageView3D, PyImageViewHolder<MutableImageView3D>> mutableImageView3D{m, "MutableImageView3D", "Three-dimensional mutable image view"};

    imageView(imageView1D);
    imageView(imageView2D);
    imageView(imageView3D);
    imageView(mutableImageView1D);
    imageView(mutableImageView2D);
    imageView(mutableImageView3D);

    imageViewFromMutable(imageView1D);
    imageViewFromMutable(imageView2D);
    imageViewFromMutable(imageView3D);

    py::enum_<SamplerFilter>{m, "SamplerFilter", "Texture sampler filtering"}
        .value("NEAREST", SamplerFilter::Nearest)
        .value("LINEAR", SamplerFilter::Linear);
    py::enum_<SamplerMipmap>{m, "SamplerMipmap", "Texture sampler mip level selection"}
        .value("BASE", SamplerMipmap::Base)
        .value("NEAREST", SamplerMipmap::Nearest)
        .value("LINEAR", SamplerMipmap::Linear);
    py::enum_<SamplerWrapping>{m, "SamplerWrapping", "Texture sampler wrapping"}
        .value("REPEAT", SamplerWrapping::Repeat)
        .value("MIRRORED_REPEAT", SamplerWrapping::MirroredRepeat)
        .value("CLAMP_TO_EDGE", SamplerWrapping::ClampToEdge)
        .value("CLAMP_TO_BORDER", SamplerWrapping::ClampToBorder)
        .value("MIRROR_CLAMP_TO_EDGE", SamplerWrapping::MirrorClampToEdge);
}

}}

/* TODO: remove declaration when https://github.com/pybind/pybind11/pull/1863
   is released */
extern "C" PYBIND11_EXPORT PyObject* PyInit__magnum();
PYBIND11_MODULE(_magnum, m) {
    m.doc() = "Root Magnum module";

    /* We need ArrayView for images */
    py::module_::import("corrade.containers");

    py::module_ math = m.def_submodule("math");
    magnum::math(m, math);

    /* These need stuff from math, so need to be called after */
    magnum::magnum(m);

    /* In case Magnum is a bunch of static libraries, put everything into a
       single shared lib to make it easier to install (which is the point of
       static builds) and avoid issues with multiply-defined global symbols.

       These need to be defined in the order they depend on. */
    #ifdef MAGNUM_BUILD_STATIC
    #ifdef Magnum_GL_FOUND
    py::module_ gl = m.def_submodule("gl");
    magnum::gl(gl);
    #endif

    #ifdef Magnum_SceneGraph_FOUND
    py::module_ scenegraph = m.def_submodule("scenegraph");
    magnum::scenegraph(scenegraph);
    #endif

    #ifdef Magnum_Trade_FOUND
    py::module_ trade = m.def_submodule("trade");
    magnum::trade(trade);
    #endif

    #ifdef Magnum_MeshTools_FOUND
    /* Depends on trade and gl */
    py::module_ meshtools = m.def_submodule("meshtools");
    magnum::meshtools(meshtools);
    #endif

    #ifdef Magnum_Primitives_FOUND
    /* Depends on trade */
    py::module_ primitives = m.def_submodule("primitives");
    magnum::primitives(primitives);
    #endif

    #ifdef Magnum_Shaders_FOUND
    /* Depends on gl */
    py::module_ shaders = m.def_submodule("shaders");
    magnum::shaders(shaders);
    #endif

    /* Keep the doc in sync with platform/__init__.py */
    py::module_ platform = m.def_submodule("platform");
    platform.doc() = "Platform-specific application and context creation";

    #ifdef Magnum_GlfwApplication_FOUND
    py::module_ glfw = platform.def_submodule("glfw");
    magnum::platform::glfw(glfw);
    #endif

    #ifdef Magnum_Sdl2Application_FOUND
    py::module_ sdl2 = platform.def_submodule("sdl2");
    magnum::platform::sdl2(sdl2);
    #endif

    #ifdef Magnum_WindowlessEglApplication_FOUND
    py::module_ egl = platform.def_submodule("egl");
    magnum::platform::egl(egl);
    #endif

    #ifdef Magnum_WindowlessGlxApplication_FOUND
    py::module_ glx = platform.def_submodule("glx");
    magnum::platform::glx(glx);
    #endif
    #endif
}
