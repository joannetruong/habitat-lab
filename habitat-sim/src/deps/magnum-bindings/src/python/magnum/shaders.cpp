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
#include <Magnum/Shaders/Flat.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Shaders/VertexColor.h>

#include "Corrade/PythonBindings.h"

#include "corrade/EnumOperators.h"
#include "magnum/bootstrap.h"

namespace magnum {

namespace {

template<class T> void anyShader(PyNonDestructibleClass<T, GL::AbstractShaderProgram>& c) {
    c.def("draw", static_cast<void(GL::AbstractShaderProgram::*)(GL::Mesh&)>(&GL::AbstractShaderProgram::draw), "Draw a mesh");
}

template<UnsignedInt dimensions> void flat(PyNonDestructibleClass<Shaders::Flat<dimensions>, GL::AbstractShaderProgram>& c) {
    /* Attributes */
    c.attr("TEXTURE_COORDINATES") = GL::DynamicAttribute{typename Shaders::Flat<dimensions>::TextureCoordinates{}};
    c.attr("COLOR3") = GL::DynamicAttribute{typename Shaders::Flat<dimensions>::Color3{}};
    c.attr("COLOR4") = GL::DynamicAttribute{typename Shaders::Flat<dimensions>::Color4{}};

    /* Methods */
    c
        .def(py::init<typename Shaders::Flat<dimensions>::Flag>(), "Constructor",
            py::arg("flags") = typename Shaders::Flat<dimensions>::Flag{})

        /* Using lambdas to avoid method chaining getting into signatures */
        .def_property_readonly("flags", [](Shaders::Flat<dimensions>& self) {
            return typename Shaders::Flat<dimensions>::Flag(UnsignedByte(self.flags()));
        }, "Flags")
        .def_property("transformation_projection_matrix", nullptr, &Shaders::Flat<dimensions>::setTransformationProjectionMatrix,
            "Transformation and projection matrix")
        .def_property("color", nullptr, &Shaders::Flat<dimensions>::setColor, "Color")
        .def_property("alpha_mask", nullptr, [](Shaders::Flat<dimensions>& self, Float mask) {
            if(!(self.flags() & Shaders::Flat<dimensions>::Flag::AlphaMask)) {
                PyErr_SetString(PyExc_AttributeError, "the shader was not created with alpha mask enabled");
                throw py::error_already_set{};
            }

            self.setAlphaMask(mask);
        }, "Alpha mask")
        .def("bind_texture", [](Shaders::Flat<dimensions>& self, GL::Texture2D& texture) {
            if(!(self.flags() & Shaders::Flat<dimensions>::Flag::Textured)) {
                PyErr_SetString(PyExc_AttributeError, "the shader was not created with texturing enabled");
                throw py::error_already_set{};
            }

            self.bindTexture(texture);
        }, "Bind a color texture");

    anyShader(c);
}

template<UnsignedInt dimensions> void vertexColor(PyNonDestructibleClass<Shaders::VertexColor<dimensions>, GL::AbstractShaderProgram>& c) {
    /* Attributes */
    c.attr("COLOR3") = GL::DynamicAttribute{typename Shaders::VertexColor<dimensions>::Color3{}};
    c.attr("COLOR4") = GL::DynamicAttribute{typename Shaders::VertexColor<dimensions>::Color4{}};

    /* Methods */
    c
        .def(py::init(), "Constructor")

        /* Using lambdas to avoid method chaining getting into signatures */

        .def_property("transformation_projection_matrix", nullptr, &Shaders::VertexColor<dimensions>::setTransformationProjectionMatrix,
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
        PyNonDestructibleClass<Shaders::Flat2D, GL::AbstractShaderProgram> flat2D{m,
            "Flat2D", "2D flat shader"};
        PyNonDestructibleClass<Shaders::Flat3D, GL::AbstractShaderProgram> flat3D{m,
            "Flat3D", "3D flat shader"};
        flat2D.attr("POSITION") = GL::DynamicAttribute{Shaders::Flat2D::Position{}};
        flat3D.attr("POSITION") = GL::DynamicAttribute{Shaders::Flat3D::Position{}};

        /* The flags are currently the same type for both 2D and 3D and pybind
           doesn't want to have a single type registered twice, so doing it
           this way instead */
        py::enum_<Shaders::Flat2D::Flag> flags{flat2D, "Flags", "Flags"};
        flags
            .value("TEXTURED", Shaders::Flat2D::Flag::Textured)
            .value("ALPHA_MASK", Shaders::Flat2D::Flag::AlphaMask)
            .value("VERTEX_COLOR", Shaders::Flat3D::Flag::AlphaMask)
            .value("NONE", Shaders::Flat3D::Flag{})
            /* TODO: OBJECT_ID, once multiple FB outputs and mapDraw is exposed */
            ;
        flat3D.attr("Flags") = flags;

        flat(flat2D);
        flat(flat3D);

        corrade::enumOperators(flags);

    }

    /* 2D/3D vertex color shader */
    {
        PyNonDestructibleClass<Shaders::VertexColor2D, GL::AbstractShaderProgram> vertexColor2D{m,
            "VertexColor2D", "2D vertex color shader"};
        PyNonDestructibleClass<Shaders::VertexColor3D, GL::AbstractShaderProgram> vertexColor3D{m,
            "VertexColor3D", "3D vertex color shader"};
        vertexColor2D.attr("POSITION") = GL::DynamicAttribute{Shaders::VertexColor2D::Position{}};
        vertexColor3D.attr("POSITION") = GL::DynamicAttribute{Shaders::VertexColor3D::Position{}};
        vertexColor(vertexColor2D);
        vertexColor(vertexColor3D);
    }

    /* Phong shader */
    {
        PyNonDestructibleClass<Shaders::Phong, GL::AbstractShaderProgram> phong{m,
            "Phong", "Phong shader"};
        phong.attr("POSITION") = GL::DynamicAttribute{Shaders::Phong::Position{}};
        phong.attr("NORMAL") = GL::DynamicAttribute{Shaders::Phong::Normal{}};
        phong.attr("TANGENT") = GL::DynamicAttribute{Shaders::Phong::Tangent{}};
        phong.attr("TANGENT4") = GL::DynamicAttribute{Shaders::Phong::Tangent4{}};
        phong.attr("BITANGENT") = GL::DynamicAttribute{Shaders::Phong::Bitangent{}};
        phong.attr("TEXTURE_COORDINATES") = GL::DynamicAttribute{Shaders::Phong::TextureCoordinates{}};
        phong.attr("COLOR3") = GL::DynamicAttribute{Shaders::Phong::Color3{}};
        phong.attr("COLOR4") = GL::DynamicAttribute{Shaders::Phong::Color4{}};
        /* TODO: OBJECT_ID attribute, once multiple FB outputs and mapDraw is
           exposed */
        phong.attr("TRANSFORMATION_MATRIX") = GL::DynamicAttribute{Shaders::Phong::TransformationMatrix{}};
        phong.attr("NORMAL_MATRIX") = GL::DynamicAttribute{Shaders::Phong::NormalMatrix{}};
        phong.attr("TEXTURE_OFFSET") = GL::DynamicAttribute{Shaders::Phong::TextureOffset{}};

        py::enum_<Shaders::Phong::Flag> flags{phong, "Flags", "Flags"};

        flags
            .value("AMBIENT_TEXTURE", Shaders::Phong::Flag::AmbientTexture)
            .value("DIFFUSE_TEXTURE", Shaders::Phong::Flag::DiffuseTexture)
            .value("SPECULAR_TEXTURE", Shaders::Phong::Flag::SpecularTexture)
            .value("NORMAL_TEXTURE", Shaders::Phong::Flag::NormalTexture)
            .value("ALPHA_MASK", Shaders::Phong::Flag::AlphaMask)
            .value("VERTEX_COLOR", Shaders::Phong::Flag::VertexColor)
            .value("BITANGENT", Shaders::Phong::Flag::Bitangent)
            .value("TEXTURE_TRANSFORMATION", Shaders::Phong::Flag::TextureTransformation)
            /* TODO: OBJECT_ID, once multiple FB outputs and mapDraw is exposed */
            .value("INSTANCED_TRANSFORMATION", Shaders::Phong::Flag::InstancedTransformation)
            .value("INSTANCED_TEXTURE_OFFSET", Shaders::Phong::Flag::InstancedTextureOffset)
            .value("NONE", Shaders::Phong::Flag{})
            ;
        corrade::enumOperators(flags);

        phong
            .def(py::init<Shaders::Phong::Flag, UnsignedInt>(), "Constructor",
                py::arg("flags") = Shaders::Phong::Flag{},
                py::arg("light_count") = 1)
            .def_property_readonly("flags", [](Shaders::Phong& self) {
                return Shaders::Phong::Flag(UnsignedShort(self.flags()));
            }, "Flags")
            .def_property_readonly("light_count", &Shaders::Phong::lightCount,
                "Light count")
            .def_property("ambient_color", nullptr,
                &Shaders::Phong::setAmbientColor, "Ambient color")
            .def_property("diffuse_color", nullptr,
                &Shaders::Phong::setDiffuseColor, "Diffuse color")
            .def_property("specular_color", nullptr,
                &Shaders::Phong::setSpecularColor, "Specular color")
            .def_property("shininess", nullptr,
                &Shaders::Phong::setShininess, "Shininess")
            .def_property("normal_texture_scale", nullptr, [](Shaders::Phong& self, Float scale) {
                if(!(self.flags() & Shaders::Phong::Flag::NormalTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with normal texture enabled");
                    throw py::error_already_set{};
                }

                self.setNormalTextureScale(scale);
            }, "Normal texture scale")
            .def_property("alpha_mask", nullptr, [](Shaders::Phong& self, Float mask) {
                if(!(self.flags() & Shaders::Phong::Flag::AlphaMask)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with alpha mask enabled");
                    throw py::error_already_set{};
                }

                self.setAlphaMask(mask);
            }, "Alpha mask")
            .def_property("transformation_matrix", nullptr,
                &Shaders::Phong::setTransformationMatrix, "Transformation matrix")
            .def_property("normal_matrix", nullptr,
                &Shaders::Phong::setNormalMatrix, "Normal matrix")
            .def_property("projection_matrix", nullptr,
                &Shaders::Phong::setProjectionMatrix, "Projection matrix")
            .def_property("texture_matrix", nullptr, [](Shaders::Phong& self, const Matrix3& matrix) {
                if(!(self.flags() & Shaders::Phong::Flag::TextureTransformation)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with texture transformation enabled");
                    throw py::error_already_set{};
                }

                self.setTextureMatrix(matrix);
            }, "Texture matrix")
            .def_property("light_positions", nullptr, [](Shaders::Phong& self, const std::vector<Vector4>& positions) {
                if(positions.size() != self.lightCount()) {
                    PyErr_Format(PyExc_ValueError, "expected %u items but got %u", self.lightCount(), UnsignedInt(positions.size()));
                    throw py::error_already_set{};
                }

                self.setLightPositions(positions);
            }, "Light positions")
            .def_property("light_colors", nullptr, [](Shaders::Phong& self, const std::vector<Color3>& colors) {
                if(colors.size() != self.lightCount()) {
                    PyErr_Format(PyExc_ValueError, "expected %u items but got %u", self.lightCount(), UnsignedInt(colors.size()));
                    throw py::error_already_set{};
                }

                self.setLightColors(colors);
            }, "Light colors")
            .def_property("light_specular_colors", nullptr, [](Shaders::Phong& self, const std::vector<Color3>& colors) {
                if(colors.size() != self.lightCount()) {
                    PyErr_Format(PyExc_ValueError, "expected %u items but got %u", self.lightCount(), UnsignedInt(colors.size()));
                    throw py::error_already_set{};
                }

                self.setLightSpecularColors(colors);
            }, "Light specular colors")
            .def_property("light_ranges", nullptr, [](Shaders::Phong& self, const std::vector<Float>& ranges) {
                if(ranges.size() != self.lightCount()) {
                    PyErr_Format(PyExc_ValueError, "expected %u items but got %u", self.lightCount(), UnsignedInt(ranges.size()));
                    throw py::error_already_set{};
                }

                self.setLightRanges(ranges);
            }, "Light ranges")

            .def("bind_ambient_texture", [](Shaders::Phong& self, GL::Texture2D& texture) {
                if(!(self.flags() & Shaders::Phong::Flag::AmbientTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with ambient texture enabled");
                    throw py::error_already_set{};
                }

                self.bindAmbientTexture(texture);
            }, "Bind an ambient texture")
            .def("bind_diffuse_texture", [](Shaders::Phong& self, GL::Texture2D& texture) {
                if(!(self.flags() & Shaders::Phong::Flag::DiffuseTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with diffuse texture enabled");
                    throw py::error_already_set{};
                }

                self.bindDiffuseTexture(texture);
            }, "Bind a diffuse texture")
            .def("bind_specular_texture", [](Shaders::Phong& self, GL::Texture2D& texture) {
                if(!(self.flags() & Shaders::Phong::Flag::SpecularTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with specular texture enabled");
                    throw py::error_already_set{};
                }

                self.bindSpecularTexture(texture);
            }, "Bind a specular texture")
            .def("bind_normal_texture", [](Shaders::Phong& self, GL::Texture2D& texture) {
                if(!(self.flags() & Shaders::Phong::Flag::NormalTexture)) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with normal texture enabled");
                    throw py::error_already_set{};
                }

                self.bindNormalTexture(texture);
            }, "Bind a normal texture")
            .def("bind_textures", [](Shaders::Phong& self, GL::Texture2D* ambient, GL::Texture2D* diffuse, GL::Texture2D* specular, GL::Texture2D* normal) {
                if(!(self.flags() & (Shaders::Phong::Flag::AmbientTexture|Shaders::Phong::Flag::DiffuseTexture|Shaders::Phong::Flag::SpecularTexture|Shaders::Phong::Flag::NormalTexture))) {
                    PyErr_SetString(PyExc_AttributeError, "the shader was not created with any textures enabled");
                    throw py::error_already_set{};
                }

                self.bindTextures(ambient, diffuse, specular, normal);
            }, "Bind textures", py::arg("ambient") = nullptr, py::arg("diffuse") = nullptr, py::arg("specular") = nullptr, py::arg("normal") = nullptr)
            ;

        anyShader(phong);
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
