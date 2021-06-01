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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> /* for Mesh.buffers */
#include <Corrade/Containers/ArrayView.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Attribute.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Math/Color.h>

#include "Corrade/PythonBindings.h"
#include "Magnum/PythonBindings.h"
#include "Magnum/GL/PythonBindings.h"

#include "corrade/EnumOperators.h"
#include "magnum/bootstrap.h"

namespace magnum { namespace {

/* Stores additional stuff needed for proper refcounting of applications that
   own current context. The context itself isn't deleted, that's a
   responsibility of the applications instead. For some reason it *has to be*
   templated, otherwise PYBIND11_DECLARE_HOLDER_TYPE doesn't work. */
template<class T> struct ContextHolder: std::unique_ptr<T, pybind11::nodelete> {
    static_assert(std::is_same<T, GL::Context>::value, "context holder has to hold a context");

    /* Used when instantiating a context directly */
    explicit ContextHolder(T* object): std::unique_ptr<T, pybind11::nodelete>{object} {}

    /* Used by Context.current() */
    explicit ContextHolder(T* object, pybind11::object owner) noexcept: std::unique_ptr<T, pybind11::nodelete>{object}, owner{std::move(owner)} {}

    ContextHolder(ContextHolder<T>&&) noexcept = default;
    ContextHolder(const ContextHolder<T>&) = delete;
    ContextHolder<T>& operator=(ContextHolder<T>&&) noexcept = default;
    ContextHolder<T>& operator=(const ContextHolder<T>&) = default;

    pybind11::object owner;
};

/* Otherwise pybind yells that `generic_type: type "Framebuffer" has a
   non-default holder type while its base "Magnum::GL::AbstractFramebuffer"
   does not` -- we're using PyFramebufferHolder for it */
template<class T> struct NonDefaultFramebufferHolder: std::unique_ptr<T, PyNonDestructibleBaseDeleter<T, std::is_destructible<T>::value>> {
    explicit NonDefaultFramebufferHolder(T* object): std::unique_ptr<T, PyNonDestructibleBaseDeleter<T, std::is_destructible<T>::value>>{object} {}
};

}}

PYBIND11_DECLARE_HOLDER_TYPE(T, magnum::ContextHolder<T>)
PYBIND11_DECLARE_HOLDER_TYPE(T, magnum::NonDefaultFramebufferHolder<T>)

namespace magnum {

namespace {

struct PublicizedAbstractShaderProgram: GL::AbstractShaderProgram {
    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    using GL::AbstractShaderProgram::setRetrievableBinary;
    #endif
    #ifndef MAGNUM_TARGET_WEBGL
    using GL::AbstractShaderProgram::setSeparable;
    #endif
    using GL::AbstractShaderProgram::attachShader;
    using GL::AbstractShaderProgram::bindAttributeLocation;
    #ifndef MAGNUM_TARGET_GLES
    using GL::AbstractShaderProgram::bindFragmentDataLocation;
    using GL::AbstractShaderProgram::bindFragmentDataLocationIndexed;
    #endif
    #ifndef MAGNUM_TARGET_GLES2
    using GL::AbstractShaderProgram::setTransformFeedbackOutputs;
    #endif
    using GL::AbstractShaderProgram::link;
    using GL::AbstractShaderProgram::uniformLocation;
    #ifndef MAGNUM_TARGET_GLES2
    using GL::AbstractShaderProgram::uniformBlockIndex;
    #endif
    using GL::AbstractShaderProgram::setUniform;
    #ifndef MAGNUM_TARGET_GLES2
    using GL::AbstractShaderProgram::setUniformBlockBinding;
    #endif
};

template<class T> void setUniform(GL::AbstractShaderProgram& self, Int location, T value) {
    static_cast<PublicizedAbstractShaderProgram&>(self).setUniform(location, value);
}

template<UnsignedInt dimensions> void texture(py::class_<GL::Texture<dimensions>, GL::AbstractTexture>& c) {
    c
        /** @todo limits */
        .def(py::init(), "Constructor")
        /** @todo bindImage(), bindImageLayered */
        #ifndef MAGNUM_TARGET_GLES2
        .def_property("base_level", nullptr, &GL::Texture<dimensions>::setBaseLevel, "Base mip level")
        #endif
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .def_property("max_level", nullptr, &GL::Texture<dimensions>::setMaxLevel, "Max mip level")
        #endif
        .def_property("minification_filter", nullptr,
            [](GL::Texture<dimensions>& self, const py::object& value) {
                if(py::isinstance<SamplerFilter>(value))
                    self.setMinificationFilter(py::cast<SamplerFilter>(value));
                else if(py::isinstance<GL::SamplerFilter>(value))
                    self.setMinificationFilter(py::cast<GL::SamplerFilter>(value));
                else if(py::isinstance<py::tuple>(value) && py::cast<py::tuple>(value).size() == 2) {
                    auto tuple = py::cast<py::tuple>(value);

                    GL::SamplerFilter filter;
                    if(py::isinstance<SamplerFilter>(tuple[0]))
                        filter = GL::samplerFilter(py::cast<SamplerFilter>(tuple[0]));
                    else if(py::isinstance<GL::SamplerFilter>(tuple[0]))
                        filter = py::cast<GL::SamplerFilter>(tuple[0]);
                    else {
                        PyErr_Format(PyExc_TypeError, "expected a tuple with SamplerFilter or gl.SamplerFilter as the first element, got %A", value.get_type().ptr());
                        throw py::error_already_set{};
                    }

                    GL::SamplerMipmap mipmap;
                    if(py::isinstance<SamplerMipmap>(tuple[1]))
                        mipmap = GL::samplerMipmap(py::cast<SamplerMipmap>(tuple[1]));
                    else if(py::isinstance<GL::SamplerMipmap>(tuple[1]))
                        mipmap = py::cast<GL::SamplerMipmap>(tuple[1]);
                    else {
                        PyErr_Format(PyExc_TypeError, "expected a tuple with SamplerMipmap or gl.SamplerMipmap as the second element, got %A", value.get_type().ptr());
                        throw py::error_already_set{};
                    }

                    self.setMinificationFilter(filter, mipmap);
                } else {
                    PyErr_Format(PyExc_TypeError, "expected SamplerFilter, gl.SamplerFilter or a two-element tuple, got %A", value.get_type().ptr());
                    throw py::error_already_set{};
                }
            }, "Minification filter")
        .def_property("magnification_filter", nullptr,
            [](GL::Texture<dimensions>& self, const py::object& filter) {
                if(py::isinstance<SamplerFilter>(filter))
                    self.setMagnificationFilter(py::cast<SamplerFilter>(filter));
                else if(py::isinstance<GL::SamplerFilter>(filter))
                    self.setMagnificationFilter(py::cast<GL::SamplerFilter>(filter));
                else {
                    PyErr_Format(PyExc_TypeError, "expected SamplerFilter or gl.SamplerFilter, got %A", filter.get_type().ptr());
                    throw py::error_already_set{};
                }
            }, "Magnification filter")
        #ifndef MAGNUM_TARGET_GLES2
        .def_property("min_lod", nullptr, &GL::Texture<dimensions>::setMinLod, "Minimum level-of-detail")
        .def_property("max_lod", nullptr, &GL::Texture<dimensions>::setMaxLod, "Maximum level-of-detail")
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .def_property("lod_bias", nullptr, &GL::Texture<dimensions>::setLodBias, "Level-of-detail bias")
        #endif
        .def_property("wrapping", nullptr,
            [](GL::Texture<dimensions>& self, const py::object& wrapping) {
                /** @todo accept two/three different values as well */
                if(py::isinstance<SamplerWrapping>(wrapping))
                    self.setWrapping(py::cast<SamplerWrapping>(wrapping));
                else if(py::isinstance<GL::SamplerWrapping>(wrapping))
                    self.setWrapping(py::cast<GL::SamplerWrapping>(wrapping));
                else {
                    PyErr_Format(PyExc_TypeError, "expected SamplerWrapping or gl.SamplerWrapping, got %A", wrapping.get_type().ptr());
                    throw py::error_already_set{};
                }
            }, "Wrapping")
        #ifndef MAGNUM_TARGET_WEBGL
        .def_property("border_color", nullptr,
            #ifdef MAGNUM_TARGET_GLES2
            &GL::Texture<dimensions>::setBorderColor,
            #else
            [](GL::Texture<dimensions>& self, const py::object& color) {
                if(py::isinstance<Vector3>(color))
                    self.setBorderColor(py::cast<Vector3>(color));
                else if(py::isinstance<Vector4>(color))
                    self.setBorderColor(py::cast<Vector4>(color));
                else if(py::isinstance<Vector4ui>(color))
                    self.setBorderColor(py::cast<Vector4ui>(color));
                else if(py::isinstance<Vector4i>(color))
                    self.setBorderColor(py::cast<Vector4i>(color));
                else {
                    PyErr_Format(PyExc_TypeError, "expected Color3, Color4, Vector4ui or Vector4i, got %A", color.get_type().ptr());
                    throw py::error_already_set{};
                }
            },
            #endif
            "Border color")
        #endif
        .def_property("max_anisotropy", nullptr, &GL::Texture<dimensions>::setMaxAnisotropy, "Max anisotropy")
        #ifndef MAGNUM_TARGET_WEBGL
        .def_property("srgb_decode", nullptr, &GL::Texture<dimensions>::setSrgbDecode, "sRGB decoding")
        #endif
        /** @todo component swizzle (it's compile-time on C++ side, ugh) */
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .def_property("compare_mode", nullptr, &GL::Texture<dimensions>::setCompareMode, "Depth texture comparison mode")
        .def_property("compare_function", nullptr, &GL::Texture<dimensions>::setCompareFunction, "Depth texture comparison function")
        #endif
        #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
        .def_property("depth_stencil_mode", nullptr, &GL::Texture<dimensions>::setDepthStencilMode, "Depth/stencil texture mode")
        #endif
        /* Using a lambda to avoid method chaining leaking to Python */
        .def("set_storage", [](GL::Texture<dimensions>& self, Int levels, GL::TextureFormat internalFormat, const typename PyDimensionTraits<dimensions, Int>::VectorType& size) {
            self.setStorage(levels, internalFormat, size);
        }, "Set storage", py::arg("levels"), py::arg("internal_format"), py::arg("size"))
        #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
        .def("image_size", [](GL::Texture<dimensions>& self, Int level) {
            return PyDimensionTraits<dimensions, Int>::from(self.imageSize(level));
        }, "Image size in given mip level", py::arg("level"))
        #endif
        /** @todo (compressed/buffer) (sub)image queries */
        /* Using a lambda to avoid method chaining leaking to Python */
        .def("set_image", [](GL::Texture<dimensions>& self, Int level, GL::TextureFormat internalFormat, const BasicImageView<dimensions>& image) {
            self.setImage(level, internalFormat, image);
        }, "Set image data", py::arg("level"), py::arg("internal_format"), py::arg("image"))
        /** @todo compressed/buffer setImage() */
        .def("set_sub_image", [](GL::Texture<dimensions>& self, Int level, const typename PyDimensionTraits<dimensions, Int>::VectorType& offset, const BasicImageView<dimensions>& image) {
            self.setSubImage(level, offset, image);
        }, "Set image subdata", py::arg("level"), py::arg("offset"), py::arg("image"))
        /** @todo compressed/buffer setSubImage() */
        .def("generate_mipmap", [](GL::Texture<dimensions>& self) {
            self.generateMipmap();
        }, "Generate mipmap")
        .def("invalidate_image", &GL::Texture<dimensions>::invalidateImage, "Invalidate texture image", py::arg("level"))
        .def("invalidate_sub_image", [](GL::Texture<dimensions>& self, Int level, const typename PyDimensionTraits<dimensions, Int>::VectorType& offset, const typename PyDimensionTraits<dimensions, Int>::VectorType& size) {
            self.invalidateSubImage(level, offset, size);
        }, "Invalidate texture subimage", py::arg("level"), py::arg("offset"), py::arg("size"));
}

}

void gl(py::module_& m) {
    /*
        Missing APIs:

        GL object labels
        limit queries
        wrap/release of underlying GL object IDs
    */

    m.doc() = "OpenGL wrapping layer";

    /* Version and related utilities */
    py::enum_<GL::Version>{m, "Version", "OpenGL version"}
        .value("NONE", GL::Version::None)
        #ifndef MAGNUM_TARGET_GLES
        .value("GL210", GL::Version::GL210)
        .value("GL300", GL::Version::GL300)
        .value("GL310", GL::Version::GL310)
        .value("GL320", GL::Version::GL320)
        .value("GL330", GL::Version::GL330)
        .value("GL400", GL::Version::GL400)
        .value("GL410", GL::Version::GL410)
        .value("GL420", GL::Version::GL420)
        .value("GL430", GL::Version::GL430)
        .value("GL440", GL::Version::GL440)
        .value("GL450", GL::Version::GL450)
        .value("GL460", GL::Version::GL460)
        #endif
        .value("GLES200", GL::Version::GLES200)
        .value("GLES300", GL::Version::GLES300)
        #ifndef MAGNUM_TARGET_WEBGL
        .value("GLES310", GL::Version::GLES310)
        .value("GLES320", GL::Version::GLES320)
        #endif
        ;
    m
        .def("version", static_cast<GL::Version(*)(Int, Int)>(GL::version), "Enum value from major and minor version number", py::arg("major"), py::arg("minor"))
        .def("version", static_cast<std::pair<Int, Int>(*)(GL::Version)>(GL::version), "Major and minor version number from enum value", py::arg("version"))
        .def("is_version_es", GL::isVersionES, "Whether given version is OpenGL ES or WebGL");

    /* Context */
    {
        py::class_<GL::Context, ContextHolder<GL::Context>> context{m, "Context", "Magnum OpenGL context"};

        py::enum_<GL::Context::Flag> contextFlag{context, "Flag", "Context flag"};
        contextFlag
            .value("DEBUG", GL::Context::Flag::Debug)
            #ifndef MAGNUM_TARGET_GLES
            .value("FORWARD_COMPATIBLE", GL::Context::Flag::ForwardCompatible)
            #endif
            .value("NO_ERROR", GL::Context::Flag::NoError)
            #ifndef MAGNUM_TARGET_GLES2
            .value("ROBUST_ACCESS", GL::Context::Flag::RobustAccess)
            #endif
            .value("NONE", GL::Context::Flag{});
        corrade::enumOperators(contextFlag);

        py::enum_<GL::Context::State> contextState{context, "State", "State to reset"};
        contextState
            .value("BUFFERS", GL::Context::State::Buffers)
            #ifndef MAGNUM_TARGET_GLES2
            .value("UNBIND_PIXEL_BUFFER", GL::Context::State::UnbindPixelBuffer)
            #endif
            .value("FRAMEBUFFERS", GL::Context::State::Framebuffers)
            .value("MESHES", GL::Context::State::Meshes)
            .value("MESH_VAO", GL::Context::State::MeshVao)
            .value("BIND_SCRATCH_VAO", GL::Context::State::BindScratchVao)
            .value("PIXEL_STORAGE", GL::Context::State::PixelStorage)
            .value("RENDERER", GL::Context::State::Renderer)
            .value("SHADERS", GL::Context::State::Shaders)
            .value("TEXTURES", GL::Context::State::Textures)
            #ifndef MAGNUM_TARGET_GLES2
            .value("TRANSFORM_FEEDBACK", GL::Context::State::TransformFeedback)
            #endif
            .value("ENTER_EXTERNAL", GL::Context::State::EnterExternal)
            .value("EXIT_EXTERNAL", GL::Context::State::ExitExternal);
        corrade::enumOperators(contextState);

        py::enum_<GL::Context::DetectedDriver> contextDetectedDriver{context, "DetectedDriver", "Detected driver"};
        contextDetectedDriver
            #ifndef MAGNUM_TARGET_WEBGL
            .value("AMD", GL::Context::DetectedDriver::Amd)
            #endif
            #ifdef MAGNUM_TARGET_GLES
            .value("ANGLE", GL::Context::DetectedDriver::Angle)
            #endif
            #ifndef MAGNUM_TARGET_WEBGL
            .value("INTEL_WINDOWS", GL::Context::DetectedDriver::IntelWindows)
            .value("MESA", GL::Context::DetectedDriver::Mesa)
            .value("NVIDIA", GL::Context::DetectedDriver::NVidia)
            .value("SVGA3D", GL::Context::DetectedDriver::Svga3D)
            #ifdef MAGNUM_TARGET_GLES
            .value("SWIFTSHADER", GL::Context::DetectedDriver::SwiftShader)
            #endif
            #endif
            #ifdef CORRADE_TARGET_ANDROID
            .value("ARM_MALI", GL::Context::DetectedDriver::ArmMali)
            #endif
            ;
        corrade::enumOperators(contextDetectedDriver);

        context
            .def_property_readonly_static("has_current", [](py::object) {
                return GL::Context::hasCurrent();
            }, "Whether there is any current context")
            .def_property_readonly_static("current", [](py::object) {
                if(!GL::Context::hasCurrent()) {
                    PyErr_SetString(PyExc_RuntimeError, "no current context");
                    throw py::error_already_set{};
                }

                py::object owner = py::none{};
                auto* glContextOwner = reinterpret_cast<std::pair<const void*, const std::type_info*>*>(py::get_shared_data("magnumGLContextOwner"));
                if(glContextOwner && glContextOwner->first) {
                    CORRADE_INTERNAL_ASSERT(glContextOwner->second);
                    owner = Corrade::pyObjectFromInstance(glContextOwner->first, *glContextOwner->second);
                }

                return ContextHolder<GL::Context>{&GL::Context::current(), owner};
            }, "Current context")
            /** @todo context switching (needs additions to the "who owns
                current context instance" variable -- a map?) */
            .def_property_readonly("version", &GL::Context::version, "OpenGL version")
            .def_property_readonly("vendor_string", &GL::Context::vendorString, "Vendor string")
            .def_property_readonly("renderer_string", &GL::Context::rendererString, "Renderer string")
            .def_property_readonly("version_string", &GL::Context::versionString, "Version string")
            .def_property_readonly("shading_language_version_string", &GL::Context::shadingLanguageVersionString, "Shading language version string")
            .def_property_readonly("shading_language_version_strings", &GL::Context::shadingLanguageVersionStrings, "Shading language version strings")
            .def_property_readonly("extension_strings", &GL::Context::extensionStrings, "Extension strings")
            #ifndef MAGNUM_TARGET_WEBGL
            .def_property_readonly("flags", &GL::Context::flags, "Context flags")
            #endif
            /** @todo supportedExtensions() (needs Extension exposed) */
            #ifndef MAGNUM_TARGET_GLES
            .def_property_readonly("is_core_profile", &GL::Context::isCoreProfile, "Detect if current OpenGL context is a core profile")
            #endif
            .def("is_version_supported", &GL::Context::isVersionSupported, "Get supported OpenGL version", py::arg("version"))
            /** @todo supportedVersion() (takes an initializer list, add an
                arrayview overload */
            /** @todo isExtensionSupported(), isExtensionDisabled() (needs
                Extension exposed) */
            .def("reset_state", [](GL::Context& self, GL::Context::State states) {
                self.resetState(states);
            }, "Reset internal state tracker", py::arg("states") = GL::Context::State{})
            .def_property_readonly("detected_driver", [](GL::Context& self) {
                return GL::Context::DetectedDriver(UnsignedShort(self.detectedDriver()));
            }, "Detected driver")

            .def_property_readonly("owner", [](GL::Context& self) {
                return pyObjectHolderFor<ContextHolder>(self).owner;
            }, "Magnum Application owning the context");
    }

    /* Shader (used by AbstractShaderProgram, so needs to be before) */
    {
        py::class_<GL::Shader> shader{m, "Shader", "Shader"};

        py::enum_<GL::Shader::Type>{shader, "Type", "Shader type"}
            .value("VERTEX", GL::Shader::Type::Vertex)
            #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
            .value("TESSELLATION_CONTROL", GL::Shader::Type::TessellationControl)
            .value("TESSELLATION_EVALUATION", GL::Shader::Type::TessellationEvaluation)
            .value("GEOMETRY", GL::Shader::Type::Geometry)
            .value("COMPUTE", GL::Shader::Type::Compute)
            #endif
            .value("FRAGMENT", GL::Shader::Type::Fragment);

        shader
            /** @todo limit queries */
            /* Constructors */
            .def(py::init<GL::Version, GL::Shader::Type>(), "Constructor", py::arg("version"), py::arg("type"))

            /* Interface */
            .def_property_readonly("id", &GL::Shader::id, "OpenGL shader ID")
            .def_property_readonly("type", &GL::Shader::type, "Shader type")
            .def_property_readonly("sources", &GL::Shader::sources, "Shader sources")
            /* Using lambdas to avoid method chaining leaking to Python */
            .def("add_source", [](GL::Shader& self, std::string source) {
                self.addSource(std::move(source));
            }, "Add shader source")
            .def("add_file", [](GL::Shader& self, const std::string& filename) {
                self.addFile(filename);
            }, "Add shader source file")
            .def("compile", [](GL::Shader& self) {
                /** @todo log redirection -- but we'd need assertions to not be
                    part of that so when it dies, the user can still see why */
                if(!self.compile()) {
                    PyErr_SetString(PyExc_RuntimeError, "compilation failed");
                    throw py::error_already_set{};
                }
            }, "Compile shader");
    }

    /* Mesh -- needed by AbstractShaderProgram.draw(), so defined earlier */
    py::class_<GL::Mesh, GL::PyMeshHolder<GL::Mesh>> mesh{m, "Mesh", "Mesh"};

    /* Abstract shader program */
    {
        /* The original class has protected functions and a pure virtual
           destructor to force people to subclass it. */
        struct PyAbstractShaderProgram: GL::AbstractShaderProgram {
            using GL::AbstractShaderProgram::AbstractShaderProgram;
        };
        py::class_<GL::AbstractShaderProgram, PyAbstractShaderProgram> abstractShaderProgram{m, "AbstractShaderProgram", "Base for shader program implementations"};

        #ifndef MAGNUM_TARGET_GLES2
        py::enum_<GL::AbstractShaderProgram::TransformFeedbackBufferMode>{abstractShaderProgram, "TransformFeedbackBufferMode", "Buffer mode for transform feedback"}
            .value("INTERLEAVED_ATTRIBUTES", GL::AbstractShaderProgram::TransformFeedbackBufferMode::InterleavedAttributes)
            .value("SEPARATE_ATTRIBUTES", GL::AbstractShaderProgram::TransformFeedbackBufferMode::SeparateAttributes);
        #endif

        abstractShaderProgram
            /** @todo limit queries */
            .def(py::init(), "Constructor")

            /* Public interface */
            .def_property_readonly("id", &GL::AbstractShaderProgram::id, "OpenGL program ID")
            .def("validate", &GL::AbstractShaderProgram::validate, "Validate program")
            .def("draw", static_cast<void(GL::AbstractShaderProgram::*)(GL::Mesh&)>(&GL::AbstractShaderProgram::draw), "Draw a mesh")
            #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
            .def("dispatch_compute", &GL::AbstractShaderProgram::dispatchCompute, "Dispatch compute")
            #endif

            /* Protected interface */
            #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
            .def_property("retrievable_binary", nullptr, &PublicizedAbstractShaderProgram::setRetrievableBinary, "Allow retrieving program binary")
            #endif
            #ifndef MAGNUM_TARGET_WEBGL
            .def_property("separable", nullptr, &PublicizedAbstractShaderProgram::setSeparable, "Allow the program to be bound to individual pipeline stages")
            #endif
            .def("attach_shader", &PublicizedAbstractShaderProgram::attachShader, "Attach a shader")
            /** @todo list-taking shader attach function */
            /* Somehow the overload static_casts don't work and it complains it
               can't bind a protected function, have to use lambdas */
            .def("bind_attribute_location", [](GL::AbstractShaderProgram& self, UnsignedInt location, const std::string& name) {
                static_cast<PublicizedAbstractShaderProgram&>(self).bindAttributeLocation(location, name);
            }, "Bind an attribute to given location", py::arg("location"), py::arg("name"))
            #ifndef MAGNUM_TARGET_GLES
            .def("bind_fragment_data_location_indexed", [](GL::AbstractShaderProgram& self, UnsignedInt location, UnsignedInt index, const std::string& name) {
                static_cast<PublicizedAbstractShaderProgram&>(self).bindFragmentDataLocationIndexed(location, index, name);
            }, "Bind fragment data to given location and first color input index", py::arg("location"), py::arg("index"), py::arg("name"))
            .def("bind_fragment_data_location", [](GL::AbstractShaderProgram& self, UnsignedInt location, const std::string& name) {
                static_cast<PublicizedAbstractShaderProgram&>(self).bindFragmentDataLocation(location, name);
            }, "Bind fragment data to given location and first color input index", py::arg("location"), py::arg("name"))
            #endif
            /** @todo setTransformFeedbackOutputs, list-taking link functions */
            /* Somehow the overload static_casts don't work and it complains it
               can't bind a protected function, have to use lambdas */
            .def("link", [](GL::AbstractShaderProgram& self) {
                /** @todo log redirection -- but we'd need assertions to not be
                    part of that so when it dies, the user can still see why */
                if(!static_cast<PublicizedAbstractShaderProgram&>(self).link()) {
                    PyErr_SetString(PyExc_RuntimeError, "linking failed");
                    throw py::error_already_set{};
                }
            }, "Link the shader")
            .def("uniform_location", [](GL::AbstractShaderProgram& self, const std::string& name) {
                /** @todo log redirection -- but we'd need assertions to not be
                    part of that so when it dies, the user can still see why */
                const Int location = static_cast<PublicizedAbstractShaderProgram&>(self).uniformLocation(name);
                if(location == -1) {
                    PyErr_Format(PyExc_ValueError, "location of uniform '%s' cannot be retrieved", name.data());
                    throw py::error_already_set{};
                }
                return location;
            }, "Get uniform location")
            #ifndef MAGNUM_TARGET_GLES2
            .def("uniform_block_index", [](GL::AbstractShaderProgram& self, const std::string& name) {
                /** @todo log redirection -- but we'd need assertions to not be
                    part of that so when it dies, the user can still see why */
                const UnsignedInt index = static_cast<PublicizedAbstractShaderProgram&>(self).uniformBlockIndex(name);
                if(index == 0xffffffffu) {
                    PyErr_Format(PyExc_ValueError, "index of uniform block '%s' cannot be retrieved", name.data());
                    throw py::error_already_set{};
                }
                return index;
            }, "Get uniform block index")
            #endif
            .def("set_uniform", setUniform<Float>, "Set uniform value")
            .def("set_uniform", setUniform<Int>, "Set uniform value")
            #ifndef MAGNUM_TARGET_GLES2
            /** @todo How to distinguish *this*? Python has just an int. */
            .def("set_uniform", setUniform<UnsignedInt>, "Set uniform value")
            #endif
            /** @todo double scalar uniform, how to distinguish? if I add it, it will get a priority over floats */
            .def("set_uniform", setUniform<Vector2>, "Set uniform value")
            .def("set_uniform", setUniform<Vector3>, "Set uniform value")
            .def("set_uniform", setUniform<Vector4>, "Set uniform value")
            .def("set_uniform", setUniform<Vector2i>, "Set uniform value")
            .def("set_uniform", setUniform<Vector3i>, "Set uniform value")
            .def("set_uniform", setUniform<Vector4i>, "Set uniform value")
            #ifndef MAGNUM_TARGET_GLES2
            .def("set_uniform", setUniform<Vector2ui>, "Set uniform value")
            .def("set_uniform", setUniform<Vector3ui>, "Set uniform value")
            .def("set_uniform", setUniform<Vector4ui>, "Set uniform value")
            #endif
            #ifndef MAGNUM_TARGET_GLES
            .def("set_uniform", setUniform<Vector2d>, "Set uniform value")
            .def("set_uniform", setUniform<Vector3d>, "Set uniform value")
            .def("set_uniform", setUniform<Vector4d>, "Set uniform value")
            #endif
            .def("set_uniform", setUniform<Matrix2x2>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix3x3>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix4x4>, "Set uniform value")
            #ifndef MAGNUM_TARGET_GLES2
            .def("set_uniform", setUniform<Matrix2x3>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix3x2>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix2x4>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix4x2>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix3x4>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix4x3>, "Set uniform value")
            #endif
            #ifndef MAGNUM_TARGET_GLES
            .def("set_uniform", setUniform<Matrix2x3d>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix3x2d>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix2x4d>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix4x2d>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix3x4d>, "Set uniform value")
            .def("set_uniform", setUniform<Matrix4x3d>, "Set uniform value")
            #endif
            #ifndef MAGNUM_TARGET_GLES2
            .def("set_uniform_block_binding", &PublicizedAbstractShaderProgram::setUniformBlockBinding, "Set uniform block binding")
            #endif
            ;
    }

    /* (Dynamic) attribute */
    py::class_<GL::DynamicAttribute> attribute{m, "Attribute", "Vertex attribute location and type"};

    py::enum_<GL::DynamicAttribute::Kind>{attribute, "Kind", "Attribute kind"}
        .value("GENERIC", GL::DynamicAttribute::Kind::Generic)
        .value("GENERIC_NORMALIZED", GL::DynamicAttribute::Kind::GenericNormalized)
        #ifndef MAGNUM_TARGET_GLES2
        .value("INTEGRAL", GL::DynamicAttribute::Kind::Integral)
        #ifndef MAGNUM_TARGET_GLES
        .value("LONG", GL::DynamicAttribute::Kind::Long)
        #endif
        #endif
        ;

    py::enum_<GL::DynamicAttribute::Components>{attribute, "Components", "Component count"}
        .value("ONE", GL::DynamicAttribute::Components::One)
        .value("TWO", GL::DynamicAttribute::Components::Two)
        .value("THREE", GL::DynamicAttribute::Components::Three)
        .value("FOUR", GL::DynamicAttribute::Components::Four)
        #ifndef MAGNUM_TARGET_GLES
        .value("BGRA", GL::DynamicAttribute::Components::BGRA)
        #endif
        ;

    py::enum_<GL::DynamicAttribute::DataType>{attribute, "DataType", "Data type"}
        .value("UNSIGNED_BYTE", GL::DynamicAttribute::DataType::UnsignedByte)
        .value("BYTE", GL::DynamicAttribute::DataType::Byte)
        .value("UNSIGNED_SHORT", GL::DynamicAttribute::DataType::UnsignedShort)
        .value("SHORT", GL::DynamicAttribute::DataType::Short)
        .value("UNSIGNED_INT", GL::DynamicAttribute::DataType::UnsignedInt)
        .value("INT", GL::DynamicAttribute::DataType::Int)
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("HALF_FLOAT", GL::DynamicAttribute::DataType::Half)
        #endif
        .value("FLOAT", GL::DynamicAttribute::DataType::Float)
        #ifndef MAGNUM_TARGET_GLES
        .value("DOUBLE", GL::DynamicAttribute::DataType::Double)
        .value("UNSIGNED_INT_10F_11F_11F_REV", GL::DynamicAttribute::DataType::UnsignedInt10f11f11fRev)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("UNSIGNED_INT_2_10_10_10_REV", GL::DynamicAttribute::DataType::UnsignedInt2101010Rev)
        .value("INT_2_10_10_10_REV", GL::DynamicAttribute::DataType::Int2101010Rev)
        #endif
        ;

    attribute
        .def(py::init<GL::DynamicAttribute::Kind, UnsignedInt, GL::DynamicAttribute::Components, GL::DynamicAttribute::DataType>(), "Constructor", py::arg("kind"), py::arg("location"), py::arg("components"), py::arg("data_type"))
        .def_property_readonly("kind", &GL::DynamicAttribute::kind, "Attribute kind")
        .def_property_readonly("location", &GL::DynamicAttribute::location, "Attribute location")
        .def_property_readonly("components", &GL::DynamicAttribute::components, "Component count")
        .def_property_readonly("data_type", &GL::DynamicAttribute::dataType, "Type of passed data");

    /* Buffer */
    py::enum_<GL::BufferUsage>{m, "BufferUsage", "Buffer usage"}
        .value("STREAM_DRAW", GL::BufferUsage::StreamDraw)
        #ifndef MAGNUM_TARGET_GLES2
        .value("STREAM_READ", GL::BufferUsage::StreamRead)
        .value("STREAM_COPY", GL::BufferUsage::StreamCopy)
        #endif
        .value("STATIC_DRAW", GL::BufferUsage::StaticDraw)
        #ifndef MAGNUM_TARGET_GLES2
        .value("STATIC_READ", GL::BufferUsage::StaticRead)
        .value("STATIC_COPY", GL::BufferUsage::StaticCopy)
        #endif
        .value("DYNAMIC_DRAW", GL::BufferUsage::DynamicDraw)
        #ifndef MAGNUM_TARGET_GLES2
        .value("DYNAMIC_READ", GL::BufferUsage::DynamicRead)
        .value("DYNAMIC_COPY", GL::BufferUsage::DynamicCopy)
        #endif
        ;

    py::class_<GL::Buffer> buffer{m, "Buffer", "Buffer"};

    py::enum_<GL::Buffer::TargetHint>{buffer, "TargetHint", "Buffer target"}
        .value("ARRAY", GL::Buffer::TargetHint::Array)
        #ifndef MAGNUM_TARGET_GLES2
        #ifndef MAGNUM_TARGET_WEBGL
        .value("ATOMIC_COUNTER", GL::Buffer::TargetHint::AtomicCounter)
        #endif
        .value("COPY_READ", GL::Buffer::TargetHint::CopyRead)
        .value("COPY_WRITE", GL::Buffer::TargetHint::CopyWrite)
        #ifndef MAGNUM_TARGET_WEBGL
        .value("DISPATCH_INDIRECT", GL::Buffer::TargetHint::DispatchIndirect)
        .value("DRAW_INDIRECT", GL::Buffer::TargetHint::DrawIndirect)
        #endif
        #endif
        .value("ELEMENT_ARRAY", GL::Buffer::TargetHint::ElementArray)
        #ifndef MAGNUM_TARGET_GLES2
        .value("PIXEL_PACK", GL::Buffer::TargetHint::PixelPack)
        .value("PIXEL_UNPACK", GL::Buffer::TargetHint::PixelUnpack)
        #ifndef MAGNUM_TARGET_WEBGL
        .value("SHADER_STORAGE", GL::Buffer::TargetHint::ShaderStorage)
        #endif
        #endif
        #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
        .value("TEXTURE", GL::Buffer::TargetHint::Texture)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("TRANSFORM_FEEDBACK", GL::Buffer::TargetHint::TransformFeedback)
        .value("UNIFORM", GL::Buffer::TargetHint::Uniform)
        #endif
        ;

    buffer
        /** @todo limit queries */
        .def(py::init<GL::Buffer::TargetHint>(), "Constructor", py::arg("target_hint") = GL::Buffer::TargetHint::Array)
        .def_property_readonly("id", &GL::Buffer::id, "OpenGL buffer ID")
        .def_property("target_hint", &GL::Buffer::targetHint, &GL::Buffer::setTargetHint, "Target hint")
        /* Using lambdas to avoid method chaining getting into signatures */
        .def("set_data", [](GL::Buffer& self, const Containers::ArrayView<const char>& data, GL::BufferUsage usage) {
            self.setData(data, usage);
        }, "Set buffer data", py::arg("data"), py::arg("usage") = GL::BufferUsage::StaticDraw)
        /** @todo more */;

    /* Renderbuffer */
    py::enum_<GL::RenderbufferFormat>{m, "RenderbufferFormat", "Internal renderbuffer format"}
        #ifndef MAGNUM_TARGET_GLES
        .value("RED", GL::RenderbufferFormat::Red)
        #endif
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("R8", GL::RenderbufferFormat::R8)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("RG", GL::RenderbufferFormat::RG)
        #endif
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("RG8", GL::RenderbufferFormat::RG8)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("RGBA", GL::RenderbufferFormat::RGBA)
        #endif
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("RGBA8", GL::RenderbufferFormat::RGBA8)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("R16", GL::RenderbufferFormat::R16)
        .value("RG16", GL::RenderbufferFormat::RG16)
        .value("RGB16", GL::RenderbufferFormat::RGB16)
        .value("RGBA16", GL::RenderbufferFormat::RGBA16)
        .value("R8UI", GL::RenderbufferFormat::R8UI)
        .value("RG8UI", GL::RenderbufferFormat::RG8UI)
        .value("RGBA8UI", GL::RenderbufferFormat::RGBA8UI)
        .value("R8I", GL::RenderbufferFormat::R8I)
        .value("RG8I", GL::RenderbufferFormat::RG8I)
        .value("RGBA8I", GL::RenderbufferFormat::RGBA8I)
        .value("R16UI", GL::RenderbufferFormat::R16UI)
        .value("RG16UI", GL::RenderbufferFormat::RG16UI)
        .value("RGBA16UI", GL::RenderbufferFormat::RGBA16UI)
        .value("R16I", GL::RenderbufferFormat::R16I)
        .value("RG16I", GL::RenderbufferFormat::RG16I)
        .value("RGBA16I", GL::RenderbufferFormat::RGBA16I)
        .value("R32UI", GL::RenderbufferFormat::R32UI)
        .value("RG32UI", GL::RenderbufferFormat::RG32UI)
        .value("RGBA32UI", GL::RenderbufferFormat::RGBA32UI)
        .value("R32I", GL::RenderbufferFormat::R32I)
        .value("RG32I", GL::RenderbufferFormat::RG32I)
        .value("RGBA32I", GL::RenderbufferFormat::RGBA32I)
        .value("R16F", GL::RenderbufferFormat::R16F)
        .value("RG16F", GL::RenderbufferFormat::RG16F)
        .value("RGBA16F", GL::RenderbufferFormat::RGBA16F)
        .value("R32F", GL::RenderbufferFormat::R32F)
        .value("RG32F", GL::RenderbufferFormat::RG32F)
        .value("RGBA32F", GL::RenderbufferFormat::RGBA32F)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("RGB10A2", GL::RenderbufferFormat::RGB10A2)
        .value("RGB10A2UI", GL::RenderbufferFormat::RGB10A2UI)
        #endif
        .value("RGB5A1", GL::RenderbufferFormat::RGB5A1)
        .value("RGBA4", GL::RenderbufferFormat::RGBA4)
        #ifndef MAGNUM_TARGET_GLES
        .value("R11FG11FB10F", GL::RenderbufferFormat::R11FG11FB10F)
        #endif
        .value("RGB565", GL::RenderbufferFormat::RGB565)
        .value("SRGB8_ALPHA8", GL::RenderbufferFormat::SRGB8Alpha8)
        #ifndef MAGNUM_TARGET_GLES
        .value("DEPTH_COMPONENT", GL::RenderbufferFormat::DepthComponent)
        #endif
        .value("DEPTH_COMPONENT16", GL::RenderbufferFormat::DepthComponent16)
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("DEPTH_COMPONENT24", GL::RenderbufferFormat::DepthComponent24)
        #endif
        #ifndef MAGNUM_TARGET_WEBGL
        .value("DEPTH_COMPONENT32", GL::RenderbufferFormat::DepthComponent32)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("DEPTH_COMPONENT32F", GL::RenderbufferFormat::DepthComponent32F)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("STENCIL_INDEX", GL::RenderbufferFormat::StencilIndex)
        #endif
        #ifndef MAGNUM_TARGET_WEBGL
        .value("STENCIL_INDEX1", GL::RenderbufferFormat::StencilIndex1)
        .value("STENCIL_INDEX4", GL::RenderbufferFormat::StencilIndex4)
        #endif
        .value("STENCIL_INDEX8", GL::RenderbufferFormat::StencilIndex8)
        #ifndef MAGNUM_TARGET_GLES
        .value("STENCIL_INDEX16", GL::RenderbufferFormat::StencilIndex16)
        #endif
        #if !defined(MAGNUM_TARGET_GLES) || (defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("DEPTH_STENCIL", GL::RenderbufferFormat::DepthStencil)
        #endif
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("DEPTH24_STENCIL8", GL::RenderbufferFormat::Depth24Stencil8)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("DEPTH32F_STENCIL8", GL::RenderbufferFormat::Depth32FStencil8)
        #endif
        ;

    py::class_<GL::Renderbuffer>{m, "Renderbuffer", "Renderbuffer"}
        /** @todo limit queries */

        .def(py::init(), "Constructor")
        .def_property_readonly("id", &GL::Renderbuffer::id, "OpenGL renderbuffer ID")
        .def("set_storage", &GL::Renderbuffer::setStorage, "Set renderbuffer storage")
        .def("set_storage_multisample", &GL::Renderbuffer::setStorageMultisample, "Set multisample renderbuffer storage");

    /* Framebuffers */
    py::enum_<GL::FramebufferClear> framebufferClear{m, "FramebufferClear", "Mask for framebuffer clearing"};
    framebufferClear
        .value("COLOR", GL::FramebufferClear::Color)
        .value("DEPTH", GL::FramebufferClear::Depth)
        .value("STENCIL", GL::FramebufferClear::Stencil);
    corrade::enumOperators(framebufferClear);

    py::enum_<GL::FramebufferBlit> framebufferBlit{m, "FramebufferBlit", "Mask for framebuffer blitting"};
    framebufferBlit
        .value("COLOR", GL::FramebufferBlit::Color)
        .value("DEPTH", GL::FramebufferBlit::Depth)
        .value("STENCIL", GL::FramebufferBlit::Stencil);
    corrade::enumOperators(framebufferBlit);

    py::enum_<GL::FramebufferBlitFilter>{m, "FramebufferBlitFilter", "Framebuffer blit filtering"}
        .value("NEAREST", GL::FramebufferBlitFilter::Nearest)
        .value("LINEAR", GL::FramebufferBlitFilter::Linear);

    py::class_<GL::AbstractFramebuffer, NonDefaultFramebufferHolder<GL::AbstractFramebuffer>> abstractFramebuffer{m,
        "AbstractFramebuffer", "Base for default and named framebuffers"};

    abstractFramebuffer
        /** @todo limit queries */

        /* Using lambdas to supply an enum and not enum set */
        .def_static("blit", [](GL::AbstractFramebuffer& source, GL::AbstractFramebuffer& destination, const Range2Di& sourceRectangle, const Range2Di& destinationRectangle, GL::FramebufferBlit mask, GL::FramebufferBlitFilter filter) {
            GL::AbstractFramebuffer::blit(source, destination, sourceRectangle, destinationRectangle, mask, filter);
        }, "Copy a block of pixels", py::arg("source"), py::arg("destination"), py::arg("source_rectangle"), py::arg("destination_rectangle"), py::arg("mask"), py::arg("filter"))
        .def_static("blit", [](GL::AbstractFramebuffer& source, GL::AbstractFramebuffer& destination, const Range2Di& rectangle, GL::FramebufferBlit mask) {
            GL::AbstractFramebuffer::blit(source, destination, rectangle, mask);
        }, "Copy a block of pixels", py::arg("source"), py::arg("destination"), py::arg("rectangle"), py::arg("mask"))
        .def("bind", &GL::AbstractFramebuffer::bind, "Bind framebuffer for drawing")
        .def_property("viewport", &GL::AbstractFramebuffer::viewport, &GL::AbstractFramebuffer::setViewport, "Viewport")
        /* Using lambdas to avoid method chaining getting into signatures */
        .def("clear", [](GL::AbstractFramebuffer& self, GL::FramebufferClear mask) {
            self.clear(mask);
        }, "Clear specified buffers in the framebuffer")
        .def("read", static_cast<void(GL::AbstractFramebuffer::*)(const Range2Di&, const MutableImageView2D&)>(&GL::AbstractFramebuffer::read), "Read a block of pixels from the framebuffer to an image view", py::arg("rectangle"), py::arg("image"))
        .def("read", static_cast<void(GL::AbstractFramebuffer::*)(const Range2Di&, Image2D&)>(&GL::AbstractFramebuffer::read), "Read a block of pixels from the framebuffer to an image", py::arg("rectangle"), py::arg("image"))
        /** @todo more */;

    py::class_<GL::DefaultFramebuffer, GL::AbstractFramebuffer, NonDefaultFramebufferHolder<GL::DefaultFramebuffer>> defaultFramebuffer{m,
        "DefaultFramebuffer", "Default framebuffer"};

    py::class_<GL::Framebuffer, GL::AbstractFramebuffer, GL::PyFramebufferHolder<GL::Framebuffer>> framebuffer{m,
        "Framebuffer", "Framebuffer"};

    py::class_<GL::Framebuffer::ColorAttachment>{framebuffer, "ColorAttachment", "Color attachment"}
        .def(py::init<UnsignedInt>(), "Constructor");

    py::class_<GL::Framebuffer::DrawAttachment>{framebuffer, "DrawAttachment", "Draw attachment"}
        .def(py::init<GL::Framebuffer::ColorAttachment>(), "Color attachment")
        .def_readonly_static("NONE", &GL::Framebuffer::DrawAttachment::None, "No attachment");

    py::implicitly_convertible<GL::Framebuffer::ColorAttachment, GL::Framebuffer::DrawAttachment>();

    py::class_<GL::Framebuffer::BufferAttachment>{framebuffer, "BufferAttachment", "Buffer attachment"}
        .def(py::init<GL::Framebuffer::ColorAttachment>(), "Color buffer")
        .def_readonly_static("DEPTH", &GL::Framebuffer::BufferAttachment::Depth, "Depth buffer")
        .def_readonly_static("STENCIL", &GL::Framebuffer::BufferAttachment::Stencil, "Stencil buffer")
        #if !defined(MAGNUM_TARGET_GLES2) || defined(MAGNUM_TARGET_WEBGL)
        .def_readonly_static("DEPTH_STENCIL", &GL::Framebuffer::BufferAttachment::DepthStencil, "Both depth and stencil buffer")
        #endif
        ;

    py::implicitly_convertible<GL::Framebuffer::ColorAttachment, GL::Framebuffer::BufferAttachment>();

    framebuffer
        .def(py::init<const Range2Di&>(), "Constructor")
        .def_property_readonly("id", &GL::Framebuffer::id, "OpenGL framebuffer ID")
        /* Using lambdas to avoid method chaining getting into signatures */
        .def("map_for_draw", [](GL::Framebuffer& self, GL::Framebuffer::DrawAttachment attachment) {
            self.mapForDraw(attachment);
        }, "Map shader output to an attachment")
        /** @todo list mapForDraw (neeeds a non-initlist variant on magnum side) */
        .def("map_for_read", [](GL::Framebuffer& self, GL::Framebuffer::ColorAttachment attachment) {
            self.mapForRead(attachment);
        }, "Map given color attachment for reading")
        .def("attach_renderbuffer", [](GL::Framebuffer& self, GL::Framebuffer::BufferAttachment attachment, GL::Renderbuffer& renderbuffer) {
            self.attachRenderbuffer(attachment, renderbuffer);

            /* Keep a reference to the renderbuffer to avoid it being deleted
               before the framebuffer */
            pyObjectHolderFor<GL::PyFramebufferHolder>(self).attachments.emplace_back(pyObjectFromInstance(renderbuffer));
        }, "Attach renderbuffer to given buffer")

        .def_property_readonly("attachments", [](GL::Framebuffer& self) {
            return pyObjectHolderFor<GL::PyFramebufferHolder>(self).attachments;
        }, "Renderbuffer and texture objects referenced by the framebuffer");

    /* An equivalent to this would be
        m.attr("default_framebuffer") = &GL::defaultFramebuffer;
       (have to use a & to make it choose return_value_policy::reference
       instead of return_value_policy::copy), but this is more explicit ---
       returning a raw pointer from functions makes pybind wrap it in an
       unique_ptr, which would cause double-free / memory corruption later */
    py::setattr(m, "default_framebuffer", py::cast(GL::defaultFramebuffer, py::return_value_policy::reference));

    /* Mesh */
    py::enum_<GL::MeshPrimitive>{m, "MeshPrimitive", "Mesh primitive type"}
        .value("POINTS", GL::MeshPrimitive::Points)
        .value("LINES", GL::MeshPrimitive::Lines)
        .value("LINE_LOOP", GL::MeshPrimitive::LineLoop)
        .value("LINE_STRIP", GL::MeshPrimitive::LineStrip)
        #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
        .value("LINES_ADJACENCY", GL::MeshPrimitive::LinesAdjacency)
        .value("LINE_STRIP_ADJACENCY", GL::MeshPrimitive::LineStripAdjacency)
        #endif
        .value("TRIANGLES", GL::MeshPrimitive::Triangles)
        .value("TRIANGLE_STRIP", GL::MeshPrimitive::TriangleStrip)
        .value("TRIANGLE_FAN", GL::MeshPrimitive::TriangleFan)
        #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
        .value("TRIANGLES_ADJACENCY", GL::MeshPrimitive::TrianglesAdjacency)
        .value("TRIANGLE_STRIP_ADJACENCY", GL::MeshPrimitive::TriangleStripAdjacency)
        .value("PATCHES", GL::MeshPrimitive::Patches)
        #endif
        ;

    /* Class definition above AbstractShaderProgram, since that needs it for
       the draw() signature */
    mesh.def(py::init<GL::MeshPrimitive>(), "Constructor", py::arg("primitive") = GL::MeshPrimitive::Triangles)
        .def(py::init<MeshPrimitive>(), "Constructor", py::arg("primitive"))
        .def_property_readonly("id", &GL::Mesh::id, "OpenGL vertex array ID")
        .def_property("primitive", &GL::Mesh::primitive,
            [](GL::Mesh& self, py::object primitive) {
                if(py::isinstance<MeshPrimitive>(primitive))
                    self.setPrimitive(py::cast<MeshPrimitive>(primitive));
                else if(py::isinstance<GL::MeshPrimitive>(primitive))
                    self.setPrimitive(py::cast<GL::MeshPrimitive>(primitive));
                else {
                    PyErr_Format(PyExc_TypeError, "expected MeshPrimitive or gl.MeshPrimitive, got %A", primitive.get_type().ptr());
                    throw py::error_already_set{};
                }
            }, "Primitive type")
        /* Have to use a lambda because it returns GL::Mesh which is not
           tracked (unlike PyMesh) */
        .def_property("count", &GL::Mesh::count, [](GL::Mesh& self, UnsignedInt count) {
            self.setCount(count);
        }, "Vertex/index count")

        /* Using lambdas to avoid method chaining getting into signatures */

        .def("add_vertex_buffer", [](GL::Mesh& self, GL::Buffer& buffer, GLintptr offset, GLsizei stride, const GL::DynamicAttribute& attribute) {
            self.addVertexBuffer(buffer, offset, stride, attribute);

            /* Keep a reference to the buffer to avoid it being deleted before
               the mesh */
            pyObjectHolderFor<GL::PyMeshHolder>(self).buffers.emplace_back(pyObjectFromInstance(buffer));
        }, "Add vertex buffer", py::arg("buffer"), py::arg("offset"), py::arg("stride"), py::arg("attribute"))
        /** @todo more */

        .def_property_readonly("buffers", [](GL::Mesh& self) {
            return pyObjectHolderFor<GL::PyMeshHolder>(self).buffers;
        }, "Buffer objects referenced by the mesh");

    /* Renderer */
    {
        py::class_<GL::Renderer> renderer{m, "Renderer", "Global renderer configuration"};

        py::enum_<GL::Renderer::Error>{renderer, "Error", "Error status"}
            .value("NO_ERROR", GL::Renderer::Error::NoError)
            .value("INVALID_ENUM", GL::Renderer::Error::InvalidEnum)
            .value("INVALID_VALUE", GL::Renderer::Error::InvalidValue)
            .value("INVALID_OPERATION", GL::Renderer::Error::InvalidOperation)
            .value("INVALID_FRAMEBUFFER_OPERATION", GL::Renderer::Error::InvalidFramebufferOperation)
            .value("OUT_OF_MEMORY", GL::Renderer::Error::OutOfMemory)
            #ifndef MAGNUM_TARGET_WEBGL
            .value("STACK_UNDERFLOW", GL::Renderer::Error::StackUnderflow)
            .value("STACK_OVERFLOW", GL::Renderer::Error::StackOverflow)
            #endif
            ;

        py::enum_<GL::Renderer::Feature>{renderer, "Feature", "Feature"}
            #ifndef MAGNUM_TARGET_WEBGL
            .value("BLEND_ADVANCED_COHERENT", GL::Renderer::Feature::BlendAdvancedCoherent)
            #endif
            .value("BLENDING", GL::Renderer::Feature::Blending)
            #if !(defined(MAGNUM_TARGET_GLES2) && defined(MAGNUM_TARGET_WEBGL))
            .value("CLIP_DISTANCE0", GL::Renderer::Feature::ClipDistance0)
            .value("CLIP_DISTANCE1", GL::Renderer::Feature::ClipDistance1)
            .value("CLIP_DISTANCE2", GL::Renderer::Feature::ClipDistance2)
            .value("CLIP_DISTANCE3", GL::Renderer::Feature::ClipDistance3)
            .value("CLIP_DISTANCE4", GL::Renderer::Feature::ClipDistance4)
            .value("CLIP_DISTANCE5", GL::Renderer::Feature::ClipDistance5)
            .value("CLIP_DISTANCE6", GL::Renderer::Feature::ClipDistance6)
            .value("CLIP_DISTANCE7", GL::Renderer::Feature::ClipDistance7)
            #endif
            #ifndef MAGNUM_TARGET_WEBGL
            .value("DEBUG_OUTPUT", GL::Renderer::Feature::DebugOutput)
            .value("DEBUG_OUTPUT_SYNCHRONOUS", GL::Renderer::Feature::DebugOutputSynchronous)
            #endif
            #ifndef MAGNUM_TARGET_GLES
            .value("DEPTH_CLAMP", GL::Renderer::Feature::DepthClamp)
            #endif
            .value("DEPTH_TEST", GL::Renderer::Feature::DepthTest)
            .value("DITHERING", GL::Renderer::Feature::Dithering)
            .value("FACE_CULLING", GL::Renderer::Feature::FaceCulling)
            #ifndef MAGNUM_TARGET_WEBGL
            .value("FRAMEBUFFER_SRGB", GL::Renderer::Feature::FramebufferSrgb)
            #endif
            #ifndef MAGNUM_TARGET_GLES
            .value("LOGIC_OPERATION", GL::Renderer::Feature::LogicOperation)
            .value("MULTISAMPLING", GL::Renderer::Feature::Multisampling)
            #endif
            .value("POLYGON_OFFSET_FILL", GL::Renderer::Feature::PolygonOffsetFill)
            #ifndef MAGNUM_TARGET_WEBGL
            .value("POLYGON_OFFSET_LINE", GL::Renderer::Feature::PolygonOffsetLine)
            .value("POLYGON_OFFSET_POINT", GL::Renderer::Feature::PolygonOffsetPoint)
            #endif
            #ifndef MAGNUM_TARGET_GLES
            .value("PROGRAM_POINT_SIZE", GL::Renderer::Feature::ProgramPointSize)
            #endif
            #ifndef MAGNUM_TARGET_GLES2
            .value("RASTERIZER_DISCARD", GL::Renderer::Feature::RasterizerDiscard)
            #endif
            #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
            .value("SAMPLE_SHADING", GL::Renderer::Feature::SampleShading)
            #endif
            #ifndef MAGNUM_TARGET_GLES
            .value("SEAMLESS_CUBE_MAP_TEXTURE", GL::Renderer::Feature::SeamlessCubeMapTexture)
            #endif
            .value("SCISSOR_TEST", GL::Renderer::Feature::ScissorTest)
            .value("STENCIL_TEST", GL::Renderer::Feature::StencilTest);

        py::enum_<GL::Renderer::BlendEquation>{renderer, "BlendEquation", "Blend Equation"}
            .value("ADD", GL::Renderer::BlendEquation::Add)
            .value("SUBTRACT", GL::Renderer::BlendEquation::Subtract)
            .value("REVERSE_SUBTRACT", GL::Renderer::BlendEquation::ReverseSubtract)
            .value("MIN", GL::Renderer::BlendEquation::Min)
            .value("MAX", GL::Renderer::BlendEquation::Max)
            .value("MULTIPLY", GL::Renderer::BlendEquation::Multiply)
            .value("SCREEN", GL::Renderer::BlendEquation::Screen)
            .value("OVERLAY", GL::Renderer::BlendEquation::Overlay)
            .value("DARKEN", GL::Renderer::BlendEquation::Darken)
            .value("LIGHTEN", GL::Renderer::BlendEquation::Lighten)
            .value("COLOR_DODGE", GL::Renderer::BlendEquation::ColorDodge)
            .value("COLOR_BURN", GL::Renderer::BlendEquation::ColorBurn)
            .value("HARD_LIGHT", GL::Renderer::BlendEquation::HardLight)
            .value("SOFT_LIGHT", GL::Renderer::BlendEquation::SoftLight)
            .value("DIFFERENCE", GL::Renderer::BlendEquation::Difference)
            .value("EXCLUSION", GL::Renderer::BlendEquation::Exclusion)
            .value("HSL_HUE", GL::Renderer::BlendEquation::HslHue)
            .value("HSL_SATURATION", GL::Renderer::BlendEquation::HslSaturation)
            .value("HSL_COLOR", GL::Renderer::BlendEquation::HslColor)
            .value("HSL_LUMINOSITY", GL::Renderer::BlendEquation::HslLuminosity);

        py::enum_<GL::Renderer::BlendFunction>{renderer, "BlendFunction", "Blend Function"}
            .value("ZERO", GL::Renderer::BlendFunction::Zero)
            .value("ONE", GL::Renderer::BlendFunction::One)
            .value("CONSTANT_COLOR", GL::Renderer::BlendFunction::ConstantColor)
            .value("ONE_MINUS_CONSTANT_COLOR", GL::Renderer::BlendFunction::OneMinusConstantColor)
            .value("CONSTANT_ALPHA", GL::Renderer::BlendFunction::ConstantAlpha)
            .value("ONE_MINUS_CONSTANT_ALPHA", GL::Renderer::BlendFunction::OneMinusConstantAlpha)
            .value("SOURCE_COLOR", GL::Renderer::BlendFunction::SourceColor)
            #ifndef MAGNUM_TARGET_GLES
            .value("SECOND_SOURCE_COLOR", GL::Renderer::BlendFunction::SecondSourceColor)
            #endif
            .value("ONE_MINUS_SOURCE_COLOR", GL::Renderer::BlendFunction::OneMinusSourceColor)
            #ifndef MAGNUM_TARGET_GLES
            .value("ONE_MINUS_SECOND_SOURCE_COLOR", GL::Renderer::BlendFunction::OneMinusSecondSourceColor)
            #endif
            .value("SOURCE_ALPHA", GL::Renderer::BlendFunction::SourceAlpha)
            .value("SOURCE_ALPHA_SATURATE", GL::Renderer::BlendFunction::SourceAlphaSaturate)
            #ifndef MAGNUM_TARGET_GLES
            .value("SECOND_SOURCE_ALPHA", GL::Renderer::BlendFunction::SecondSourceAlpha)
            #endif
            .value("ONE_MINUS_SOURCE_ALPHA", GL::Renderer::BlendFunction::OneMinusSourceAlpha)
            #ifndef MAGNUM_TARGET_GLES
            .value("ONE_MINUS_SECOND_SOURCE_ALPHA", GL::Renderer::BlendFunction::OneMinusSecondSourceAlpha)
            #endif
            .value("DESTINATION_COLOR", GL::Renderer::BlendFunction::DestinationColor)
            .value("ONE_MINUS_DESTINATION_COLOR", GL::Renderer::BlendFunction::OneMinusDestinationColor)
            .value("DESTINATION_ALPHA", GL::Renderer::BlendFunction::DestinationAlpha)
            .value("ONE_MINUS_DESTINATION_ALPHA", GL::Renderer::BlendFunction::OneMinusDestinationAlpha);

        renderer
            .def_static("enable", static_cast<void(*)(GL::Renderer::Feature)>(GL::Renderer::enable), "Enable a feature", py::arg("feature"))
            .def_static("disable", static_cast<void(*)(GL::Renderer::Feature)>(GL::Renderer::disable), "Disable a feature", py::arg("feature"))
            .def_static("set_feature", static_cast<void(*)(GL::Renderer::Feature, bool)>(GL::Renderer::setFeature), "Enable or disable a feature", py::arg("feature"), py::arg("enabled"))
            /** @todo indexed variants of enable/disable/set_feature (needs
                deprecation of the original ones and making draw_buffer first
                so it's consistent with the rest) */
            .def_static("set_blend_equation", static_cast<void(*)(GL::Renderer::BlendEquation)>(GL::Renderer::setBlendEquation), "Set blend equation", py::arg("equation"))
            #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
            .def_static("set_blend_equation", static_cast<void(*)(UnsignedInt, GL::Renderer::BlendEquation)>(GL::Renderer::setBlendEquation), "Set blend equation for given draw buffer", py::arg("draw_buffer"), py::arg("equation"))
            #endif
            .def_static("set_blend_equation", static_cast<void(*)(GL::Renderer::BlendEquation, GL::Renderer::BlendEquation)>(GL::Renderer::setBlendEquation), "Set blend equation separately for RGB and alpha components", py::arg("rgb"), py::arg("alpha"))
            #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
            .def_static("set_blend_equation", static_cast<void(*)(UnsignedInt, GL::Renderer::BlendEquation, GL::Renderer::BlendEquation)>(GL::Renderer::setBlendEquation), "Set blend equation for given draw buffer separately for RGB and alpha components", py::arg("draw_buffer"), py::arg("rgb"), py::arg("alpha"))
            #endif
            .def_static("set_blend_function", static_cast<void(*)(GL::Renderer::BlendFunction, GL::Renderer::BlendFunction)>(GL::Renderer::setBlendFunction), "Set blend function", py::arg("source"), py::arg("destination"))
            #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
            .def_static("set_blend_function", static_cast<void(*)(UnsignedInt, GL::Renderer::BlendFunction, GL::Renderer::BlendFunction)>(GL::Renderer::setBlendFunction), "Set blend function for given draw buffer", py::arg("draw_buffer"), py::arg("source"), py::arg("destination"))
            #endif
            .def_static("set_blend_function", static_cast<void(*)(GL::Renderer::BlendFunction, GL::Renderer::BlendFunction, GL::Renderer::BlendFunction, GL::Renderer::BlendFunction)>(GL::Renderer::setBlendFunction), "Set blend function separately for RGB and alpha components", py::arg("source_rgb"), py::arg("destination_rgb"), py::arg("source_alpha"), py::arg("destination_alpha"))
            #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
            .def_static("set_blend_function", static_cast<void(*)(UnsignedInt, GL::Renderer::BlendFunction, GL::Renderer::BlendFunction, GL::Renderer::BlendFunction, GL::Renderer::BlendFunction)>(GL::Renderer::setBlendFunction), "Set blend function separately for RGB and alpha components", py::arg("draw_buffer"), py::arg("source_rgb"), py::arg("destination_rgb"), py::arg("source_alpha"), py::arg("destination_alpha"))
            #endif

            /** @todo FFS why do I have to pass the class as first argument?! */
            .def_property_static("clear_color", nullptr, [](py::object, const Color4& color) {
                GL::Renderer::setClearColor(color);
            }, "Set clear color")
            .def_property_readonly_static("error", [](py::object) {
                return GL::Renderer::error();
            }, "Error status");
    }

    /* Textures */

    /** @todo enum constructors converting generic value to GL-specific? */
    py::enum_<GL::SamplerFilter>{m, "SamplerFilter", "Texture sampler filtering"}
        .value("NEAREST", GL::SamplerFilter::Nearest)
        .value("LINEAR", GL::SamplerFilter::Linear);
    py::enum_<GL::SamplerMipmap>{m, "SamplerMipmap", "Texture sampler mip level selection"}
        .value("BASE", GL::SamplerMipmap::Base)
        .value("NEAREST", GL::SamplerMipmap::Nearest)
        .value("LINEAR", GL::SamplerMipmap::Linear);
    py::enum_<GL::SamplerWrapping>{m, "SamplerWrapping", "Texture sampler wrapping"}
        .value("REPEAT", GL::SamplerWrapping::Repeat)
        .value("MIRRORED_REPEAT", GL::SamplerWrapping::MirroredRepeat)
        .value("CLAMP_TO_EDGE", GL::SamplerWrapping::ClampToEdge)
        #ifndef MAGNUM_TARGET_WEBGL
        .value("CLAMP_TO_BORDER", GL::SamplerWrapping::ClampToBorder)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("MIRROR_CLAMP_TO_EDGE", GL::SamplerWrapping::MirrorClampToEdge)
        #endif
        ;
    #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
    py::enum_<GL::SamplerCompareMode>{m, "SamplerCompareMode", "Depth texture comparison mode"}
        .value("NONE", GL::SamplerCompareMode::None)
        .value("COMPARE_REF_TO_TEXTURE", GL::SamplerCompareMode::CompareRefToTexture);
    py::enum_<GL::SamplerCompareFunction>{m, "SamplerCompareFunction", "Texture sampler depth comparison function"}
        .value("NEVER", GL::SamplerCompareFunction::Never)
        .value("ALWAYS", GL::SamplerCompareFunction::Always)
        .value("LESS", GL::SamplerCompareFunction::Less)
        .value("LESS_OR_EQUAL", GL::SamplerCompareFunction::LessOrEqual)
        .value("EQUAL", GL::SamplerCompareFunction::Equal)
        .value("NOT_EQUAL", GL::SamplerCompareFunction::NotEqual)
        .value("GREATER_OR_EQUAL", GL::SamplerCompareFunction::GreaterOrEqual)
        .value("GREATER", GL::SamplerCompareFunction::Greater);
    #endif
    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    py::enum_<GL::SamplerDepthStencilMode>{m, "SamplerDepthStencilMode", "Texture sampler depth/stencil mode"}
        .value("DEPTH_COMPONENT", GL::SamplerDepthStencilMode::DepthComponent)
        .value("STENCIL_INDEX", GL::SamplerDepthStencilMode::StencilIndex);
    #endif

    py::enum_<GL::TextureFormat>{m, "TextureFormat", "Internal texture format"}
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("RED", GL::TextureFormat::Red)
        .value("R8", GL::TextureFormat::R8)
        .value("RG", GL::TextureFormat::RG)
        #endif
        .value("RGB", GL::TextureFormat::RGB)
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("RGB8", GL::TextureFormat::RGB8)
        #endif
        .value("RGBA", GL::TextureFormat::RGBA)
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("RGBA8", GL::TextureFormat::RGBA8)
        #endif
        #ifndef MAGNUM_TARGET_WEBGL
        .value("SR8", GL::TextureFormat::SR8)
        #ifdef MAGNUM_TARGET_GLES
        /** @todo how to expose this one in the docs? */
        .value("SRG8", GL::TextureFormat::SRG8)
        #endif
        #endif
        #if !defined(MAGNUM_TARGET_GLES) || defined(MAGNUM_TARGET_GLES2)
        .value("SRGB", GL::TextureFormat::SRGB)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("SRGB8", GL::TextureFormat::SRGB8)
        #endif
        #if !defined(MAGNUM_TARGET_GLES) || defined(MAGNUM_TARGET_GLES2)
        .value("SRGB_ALPHA", GL::TextureFormat::SRGBAlpha)
        #endif
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("SRGB8_ALPHA8", GL::TextureFormat::SRGB8Alpha8)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("R8_SNORM", GL::TextureFormat::R8Snorm)
        .value("RG8_SNORM", GL::TextureFormat::RG8Snorm)
        .value("RGB8_SNORM", GL::TextureFormat::RGB8Snorm)
        .value("RGBA8_SNORM", GL::TextureFormat::RGBA8Snorm)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("R16", GL::TextureFormat::R16)
        .value("RG16", GL::TextureFormat::RG16)
        .value("RGB16", GL::TextureFormat::RGB16)
        .value("RGBA16", GL::TextureFormat::RGBA16)
        .value("R16_SNORM", GL::TextureFormat::R16Snorm)
        .value("RG16_SNORM", GL::TextureFormat::RG16Snorm)
        .value("RGB16_SNORM", GL::TextureFormat::RGB16Snorm)
        .value("RGBA16_SNORM", GL::TextureFormat::RGBA16Snorm)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("R8UI", GL::TextureFormat::R8UI)
        .value("RG8UI", GL::TextureFormat::RG8UI)
        .value("RGB8UI", GL::TextureFormat::RGB8UI)
        .value("RGBA8UI", GL::TextureFormat::RGBA8UI)
        .value("R8I", GL::TextureFormat::R8I)
        .value("RG8I", GL::TextureFormat::RG8I)
        .value("RGB8I", GL::TextureFormat::RGB8I)
        .value("RGBA8I", GL::TextureFormat::RGBA8I)
        .value("R16UI", GL::TextureFormat::R16UI)
        .value("RG16UI", GL::TextureFormat::RG16UI)
        .value("RGB16UI", GL::TextureFormat::RGB16UI)
        .value("RGBA16UI", GL::TextureFormat::RGBA16UI)
        .value("R16I", GL::TextureFormat::R16I)
        .value("RG16I", GL::TextureFormat::RG16I)
        .value("RGB16I", GL::TextureFormat::RGB16I)
        .value("RGBA16I", GL::TextureFormat::RGBA16I)
        .value("R32UI", GL::TextureFormat::R32UI)
        .value("RG32UI", GL::TextureFormat::RG32UI)
        .value("RGB32UI", GL::TextureFormat::RGB32UI)
        .value("RGBA32UI", GL::TextureFormat::RGBA32UI)
        .value("R32I", GL::TextureFormat::R32I)
        .value("RG32I", GL::TextureFormat::RG32I)
        .value("RGB32I", GL::TextureFormat::RGB32I)
        .value("RGBA32I", GL::TextureFormat::RGBA32I)
        .value("R16F", GL::TextureFormat::R16F)
        .value("RG16F", GL::TextureFormat::RG16F)
        .value("RGB16F", GL::TextureFormat::RGB16F)
        .value("RGBA16F", GL::TextureFormat::RGBA16F)
        .value("R32F", GL::TextureFormat::R32F)
        .value("RG32F", GL::TextureFormat::RG32F)
        .value("RGB32F", GL::TextureFormat::RGB32F)
        .value("RGBA32F", GL::TextureFormat::RGBA32F)
        #endif
        #ifdef MAGNUM_TARGET_GLES2
        /** @todo how to expose those in the docs? */
        .value("LUMINANCE", GL::TextureFormat::Luminance)
        .value("LUMINANCE_ALPHA", GL::TextureFormat::LuminanceAlpha)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("R3B3G2", GL::TextureFormat::R3G3B2)
        .value("RGB4", GL::TextureFormat::RGB4)
        .value("RGB5", GL::TextureFormat::RGB5)
        #endif
        .value("RGB565", GL::TextureFormat::RGB565)
        #if !defined(MAGNUM_TARGET_GLES) || (defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL))
        .value("RGB10", GL::TextureFormat::RGB10)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("RGB12", GL::TextureFormat::RGB12)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("R11FG11FB10F", GL::TextureFormat::R11FG11FB10F)
        .value("RGB9E5", GL::TextureFormat::RGB9E5)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("RGBA2", GL::TextureFormat::RGBA2)
        #endif
        .value("RGBA4", GL::TextureFormat::RGBA4)
        .value("RGB5A1", GL::TextureFormat::RGB5A1)
        #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
        .value("RGB10A2", GL::TextureFormat::RGB10A2)
        #endif
        #ifndef MAGNUM_TARGET_GLES2
        .value("RGB10A2UI", GL::TextureFormat::RGB10A2UI)
        #endif
        #ifndef MAGNUM_TARGET_GLES
        .value("RGBA12", GL::TextureFormat::RGBA12)
        #endif
        ;
        /** @todo compressed formats */

    PyNonDestructibleClass<GL::AbstractTexture>{m, "AbstractTexture", "Base for textures"}
        .def_static("unbind", static_cast<void(*)(Int)>(&GL::AbstractTexture::unbind), "Unbind any texture from given texture unit")
        /** @todo limits */
        .def_property_readonly("id", &GL::AbstractTexture::id, "OpenGL texture ID")
        /** @todo list-taking bind */
        .def("bind", static_cast<void(GL::AbstractTexture::*)(Int)>(&GL::AbstractTexture::bind), "Bind texture to given texture unit");

    #ifndef MAGNUM_TARGET_GLES
    py::class_<GL::Texture1D, GL::AbstractTexture> texture1D{m, "Texture1D", "One-dimensional texture"};
    texture(texture1D);
    #endif
    py::class_<GL::Texture2D, GL::AbstractTexture> texture2D{m, "Texture2D", "Two-dimensional texture"};
    texture(texture2D);
    #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
    py::class_<GL::Texture3D, GL::AbstractTexture> texture3D{m, "Texture3D", "Three-dimensional texture"};
    texture(texture3D);
    #endif
}

}

#ifndef MAGNUM_BUILD_STATIC
/* TODO: remove declaration when https://github.com/pybind/pybind11/pull/1863
   is released */
extern "C" PYBIND11_EXPORT PyObject* PyInit_gl();
PYBIND11_MODULE(gl, m) {
    magnum::gl(m);
}
#endif
