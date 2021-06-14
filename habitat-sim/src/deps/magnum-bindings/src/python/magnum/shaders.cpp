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
#include <pybind11/stl.h> /* for vector arguments */
#include <Corrade/Containers/ArrayViewStl.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Shaders/PhongGL.h>
#include <Magnum/Shaders/VertexColorGL.h>

#include "Corrade/PythonBindings.h"

#include "corrade/EnumOperators.h"
#include "magnum/bootstrap.h"

namespace magnum {

namespace {

template<class T> void anyShader(PyNonDestructibleClass<T, GL::AbstractShaderProgram>& c) {
    c.def("draw", [](GL::AbstractShaderProgram& self, GL::Mesh& mesh) {
        self.draw(mesh);
    }, "Draw a mesh");
}

template<UnsignedInt dimensions> void flat(PyNonDestructibleClass<Shaders::FlatGL<dimensions>, GL::AbstractShaderProgram>& c) {
    /* Attributes */
    c.attr("TEXTURE_COORDINATES") = GL::DynamicAttribute{typename Shaders::FlatGL<dimensions>::TextureCoordinates{}};
    c.attr("COLOR3") = GL::DynamicAttribute{typename Shaders::FlatGL<dimensions>::Color3{}};
    c.attr("COLOR4") = GL::DynamicAttribute{typename Shaders::FlatGL<dimensions>::Color4{}};

    /* Methods */
    c
        .def(py::init<typename Shaders::FlatGL<dimensions>::Flag>(), "Constructor",
            py::arg("flags") = typename Shaders::FlatGL<dimensions>::Flag{})

        /* Using lambdas to avoid method chaining getting into signatures */
        .def_property_readonly("flags", [](Shaders::FlatGL<dimensions>& self) {
            return typename Shaders::FlatGL<dimensions>::Flag(UnsignedShort(self.flags()));
        }, "Flags")
        .def_property("transformation_projection_matrix", nullptr, &Shaders::FlatGL<dimensions>::setTransformationProjectionMatrix,
            "Transformation and projection matrix")
        .def_property("color", nullptr, &Shaders::FlatGL<dimensions>::setColor, "Color")
        .def_property("alpha_mask", nullptr, [](Shaders::FlatGL<dimensions>& self, Float mask) {
            if(!(self.flags() & Shaders::FlatGL<dimensions>::Flag::AlphaMask)) {
                PyErr_SetString(PyExc_AttributeError, "the shader was not created with alpha mask enabled");
                throw py::error_already_set{};
            }

            self.setAlphaMask(mask);
        }, "Alpha mask")
        .def("bind_texture", [](Shaders::FlatGL<dimensions>& self, GL::Texture2D& texture) {
            if(!(self.flags() & Shaders::FlatGL<dimensions>::Flag::Textured)) {
                PyErr_SetString(PyExc_AttributeError, "the shader was not created with texturing enabled");
                throw py::error_already_set{};
            }

            self.bindTexture(texture);
        }, "Bind a color texture");

    anyShader(c);
}

template<UnsignedInt dimensions> void vertexColor(PyNonDestructibleClass<Shaders::VertexColorGL<dimensions>, GL::AbstractShaderProgram>& c) {
    /* Attributes */
    c.attr("COLOR3") = GL::DynamicAttribute{typename Shaders::VertexColorGL<dimensions>::Color3{}};
    c.attr("COLOR4") = GL::DynamicAttribute{typename Shaders::VertexColorGL<dimensions>::Color4{}};

    /* Methods */
    c
        .def(py::init(), "Constructor")

        /* Using lambdas to avoid method chaining getting into signatures */

        .def_property("transformation_projection_matrix", nullptr, &Shaders::VertexColorGL<dimensions>::setTransformationProjectionMatrix,
            "Transformation and projection matrix");

    anyShader(c);
}

}

void shaders(py::module_& m) {
    m.doc() = "Builtin shaders";

    #ifndef MAGNUM_BUILD_STATIC
    /* These are a part of the same module in the static build, no need to
       import (also can't import because there it's _magnum.*) */
    py::module_::import("magnum.gl");
    #endif

    /* 2D/3D flat shader */
    {
        PyNonDestructibleClass<Shaders::FlatGL2D, GL::AbstractShaderProgram> flatGL2D{m,
            "FlatGL2D", "2D flat OpenGL shader"};
        PyNonDestructibleClass<Shaders::FlatGL3D, GL::AbstractShaderProgram> flatGL3D{m,
            "FlatGL3D", "3D flat shader"};
        flatGL2D.attr("POSITION") = GL::DynamicAttribute{Shaders::FlatGL2D::Position{}};
        flatGL3D.attr("POSITION") = GL::DynamicAttribute{Shaders::FlatGL3D::Position{}};

        /* The flags are currently the same type for both 2D and 3D and pybind
           doesn't want to have a single type registered twice, so doing it
           this way instead */
        py::enum_<Shaders::FlatGL2D::Flag> flags{flatGL2D, "Flags", "Flags"};
        flags
            .value("TEXTURED", Shaders::FlatGL2D::Flag::Textured)
            .value("ALPHA_MASK", Shaders::FlatGL2D::Flag::AlphaMask)
            .value("VERTEX_COLOR", Shaders::FlatGL3D::Flag::AlphaMask)
            .value("NONE", Shaders::FlatGL3D::Flag{})
            /* TODO: OBJECT_ID, once multiple FB outputs and mapDraw is exposed */
            ;
        flatGL3D.attr("Flags") = flags;

        flat(flatGL2D);
        flat(flatGL3D);

        corrade::enumOperators(flags);

    }

    /* 2D/3D vertex color shader */
    {
        PyNonDestructibleClass<Shaders::VertexColorGL2D, GL::AbstractShaderProgram> vertexColorGL2D{m,
            "VertexColorGL2D", "2D vertex color shader"};
        PyNonDestructibleClass<Shaders::VertexColorGL3D, GL::AbstractShaderProgram> vertexColorGL3D{m,
            "VertexColorGL3D", "3D vertex color shader"};
        vertexColorGL2D.attr("POSITION") = GL::DynamicAttribute{Shaders::VertexColorGL2D::Position{}};
        vertexColorGL3D.attr("POSITION") = GL::DynamicAttribute{Shaders::VertexColorGL3D::Position{}};
        vertexColor(vertexColorGL2D);
        vertexColor(vertexColorGL3D);
    }

    /* Phong shader */
    {
        PyNonDestructibleClass<Shaders::PhongGL, GL::AbstractShaderProgram> phongGL{m,
            "PhongGL", "Phong shader"};
        phongGL.attr("POSITION") = GL::DynamicAttribute{Shaders::PhongGL::Position{}};
        phongGL.attr("NORMAL") = GL::DynamicAttribute{Shaders::PhongGL::Normal{}};
        phongGL.attr("TANGENT") = GL::DynamicAttribute{Shaders::PhongGL::Tangent{}};
        phongGL.attr("TANGENT4") = GL::DynamicAttribute{Shaders::PhongGL::Tangent4{}};
        phongGL.attr("BITANGENT") = GL::DynamicAttribute{Shaders::PhongGL::Bitangent{}};
        phongGL.attr("TEXTURE_COORDINATES") = GL::DynamicAttribute{Shaders::PhongGL::TextureCoordinates{}};
        phongGL.attr("COLOR3") = GL::DynamicAttribute{Shaders::PhongGL::Color3{}};
        phongGL.attr("COLOR4") = GL::DynamicAttribute{Shaders::PhongGL::Color4{}};
        /* TODO: OBJECT_ID attribute, once multiple FB outputs and mapDraw is
           exposed */
        phongGL.attr("TRANSFORMATION_MATRIX") = GL::DynamicAttribute{Shaders::PhongGL::TransformationMatrix{}};
        phongGL.attr("NORMAL_MATRIX") = GL::DynamicAttribute{Shaders::PhongGL::NormalMatrix{}};
        phongGL.attr("TEXTURE_OFFSET") = GL::DynamicAttribute{Shaders::PhongGL::TextureOffset{}};

        py::enum_<Shaders::PhongGL::Flag> flags{phongGL, "Flags", "Flags"};

        flags
            .value("AMBIENT_TEXTURE", Shaders::PhongGL::Flag::AmbientTexture)
            .value("DIFFUSE_TEXTURE", Shaders::PhongGL::Flag::DiffuseTexture)
            .value("SPECULAR_TEXTURE", Shaders::PhongGL::Flag::SpecularTexture)
            .value("NORMAL_TEXTURE", Shaders::PhongGL::Flag::NormalTexture)
            .value("ALPHA_MASK", Shaders::PhongGL::Flag::AlphaMask)
            .value("VERTEX_COLOR", Shaders::PhongGL::Flag::VertexColor)
            .value("BITANGENT", Shaders::PhongGL::Flag::Bitangent)
            .value("TEXTURE_TRANSFORMATION", Shaders::PhongGL::Flag::TextureTransformation)
            /* TODO: OBJECT_ID, once multiple FB outputs and mapDraw is exposed */
            .value("INSTANCED_TRANSFORMATION", Shaders::PhongGL::Flag::InstancedTransformation)
            .value("INSTANCED_TEXTURE_OFFSET", Shaders::PhongGL::Flag::InstancedTextureOffset)
            .value("NONE", Shaders::PhongGL::Flag{})
            ;
        corrade::enumOperators(flags);

        phongGL
            .def(py::init<Shaders::PhongGL::Flag, UnsignedInt>(), "Constructor",
                py::arg("flags") = Shaders::PhongGL::Flag{},
                py::arg("light_count") = 1)
            .def_property_readonly("flags", [](Shaders::PhongGL& self) {
                return Shaders::PhongGL::Flag(UnsignedInt(self.flags()));
            }, "Flags")
            .def_property_readonly("light_count", &Shaders::PhongGL::lightCount,
                "Light count")
            .def_property("ambient_color", nullptr,
                &Shaders::PhongGL::setAmbientColor, "Ambient color")
            .def_property("diffuse_color", nullptr,
                &Shaders::PhongGL::setDiffuseColor, "Diffuse color")
            .def_property("specular_color", nullptr,
                &Shaders::PhongGL::setSpecularColor, "Specular color")
            .def_property("shininess", nullptr,
                &Shaders::PhongGL::setShininess, "Shininess")
            .def_property("normal_texture_scale", nullptr, [](Shaders::PhongGL& self, Float scale) {
                if(!(self.flags() & Shaders::PhongGL::Flag::NormalTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with normal texture enabled");
                    throw py::error_already_set{};
                }

                self.setNormalTextureScale(scale);
            }, "Normal texture scale")
            .def_property("alpha_mask", nullptr, [](Shaders::PhongGL& self, Float mask) {
                if(!(self.flags() & Shaders::PhongGL::Flag::AlphaMask)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with alpha mask enabled");
                    throw py::error_already_set{};
                }

                self.setAlphaMask(mask);
            }, "Alpha mask")
            .def_property("transformation_matrix", nullptr,
                &Shaders::PhongGL::setTransformationMatrix, "Transformation matrix")
            .def_property("normal_matrix", nullptr,
                &Shaders::PhongGL::setNormalMatrix, "Normal matrix")
            .def_property("projection_matrix", nullptr,
                &Shaders::PhongGL::setProjectionMatrix, "Projection matrix")
            .def_property("texture_matrix", nullptr, [](Shaders::PhongGL& self, const Matrix3& matrix) {
                if(!(self.flags() & Shaders::PhongGL::Flag::TextureTransformation)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with texture transformation enabled");
                    throw py::error_already_set{};
                }

                self.setTextureMatrix(matrix);
            }, "Texture matrix")
            .def_property("light_positions", nullptr, [](Shaders::PhongGL& self, const std::vector<Vector4>& positions) {
                if(positions.size() != self.lightCount()) {
                    PyErr_Format(PyExc_ValueError, "expected %u items but got %u", self.lightCount(), UnsignedInt(positions.size()));
                    throw py::error_already_set{};
                }

                self.setLightPositions(positions);
            }, "Light positions")
            .def_property("light_colors", nullptr, [](Shaders::PhongGL& self, const std::vector<Color3>& colors) {
                if(colors.size() != self.lightCount()) {
                    PyErr_Format(PyExc_ValueError, "expected %u items but got %u", self.lightCount(), UnsignedInt(colors.size()));
                    throw py::error_already_set{};
                }

                self.setLightColors(colors);
            }, "Light colors")
            .def_property("light_specular_colors", nullptr, [](Shaders::PhongGL& self, const std::vector<Color3>& colors) {
                if(colors.size() != self.lightCount()) {
                    PyErr_Format(PyExc_ValueError, "expected %u items but got %u", self.lightCount(), UnsignedInt(colors.size()));
                    throw py::error_already_set{};
                }

                self.setLightSpecularColors(colors);
            }, "Light specular colors")
            .def_property("light_ranges", nullptr, [](Shaders::PhongGL& self, const std::vector<Float>& ranges) {
                if(ranges.size() != self.lightCount()) {
                    PyErr_Format(PyExc_ValueError, "expected %u items but got %u", self.lightCount(), UnsignedInt(ranges.size()));
                    throw py::error_already_set{};
                }

                self.setLightRanges(ranges);
            }, "Light ranges")

            .def("bind_ambient_texture", [](Shaders::PhongGL& self, GL::Texture2D& texture) {
                if(!(self.flags() & Shaders::PhongGL::Flag::AmbientTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with ambient texture enabled");
                    throw py::error_already_set{};
                }

                self.bindAmbientTexture(texture);
            }, "Bind an ambient texture")
            .def("bind_diffuse_texture", [](Shaders::PhongGL& self, GL::Texture2D& texture) {
                if(!(self.flags() & Shaders::PhongGL::Flag::DiffuseTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with diffuse texture enabled");
                    throw py::error_already_set{};
                }

                self.bindDiffuseTexture(texture);
            }, "Bind a diffuse texture")
            .def("bind_specular_texture", [](Shaders::PhongGL& self, GL::Texture2D& texture) {
                if(!(self.flags() & Shaders::PhongGL::Flag::SpecularTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with specular texture enabled");
                    throw py::error_already_set{};
                }

                self.bindSpecularTexture(texture);
            }, "Bind a specular texture")
            .def("bind_normal_texture", [](Shaders::PhongGL& self, GL::Texture2D& texture) {
                if(!(self.flags() & Shaders::PhongGL::Flag::NormalTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with normal texture enabled");
                    throw py::error_already_set{};
                }

                self.bindNormalTexture(texture);
            }, "Bind a normal texture")
            .def("bind_textures", [](Shaders::PhongGL& self, GL::Texture2D* ambient, GL::Texture2D* diffuse, GL::Texture2D* specular, GL::Texture2D* normal) {
                if(!(self.flags() & (Shaders::PhongGL::Flag::AmbientTexture|Shaders::PhongGL::Flag::DiffuseTexture|Shaders::PhongGL::Flag::SpecularTexture|Shaders::PhongGL::Flag::NormalTexture))) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with any textures enabled");
                    throw py::error_already_set{};
                }

                self.bindTextures(ambient, diffuse, specular, normal);
            }, "Bind textures", py::arg("ambient") = nullptr, py::arg("diffuse") = nullptr, py::arg("specular") = nullptr, py::arg("normal") = nullptr)
            ;

        anyShader(phongGL);
    }
}

}

#ifndef MAGNUM_BUILD_STATIC
/* TODO: remove declaration when https://github.com/pybind/pybind11/pull/1863
   is released */
extern "C" PYBIND11_EXPORT PyObject* PyInit_shaders();
PYBIND11_MODULE(shaders, m) {
    magnum::shaders(m);
}
#endif
