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
#include <Magnum/GL/Mesh.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Trade/MeshData.h>

#include "corrade/EnumOperators.h"
#include "magnum/bootstrap.h"

namespace magnum {

void meshtools(py::module_& m) {
    m.doc() = "Mesh tools";

    #ifndef MAGNUM_BUILD_STATIC
    /* These are a part of the same module in the static build, no need to
       import (also can't import because there it's _magnum.*) */
    py::module_::import("magnum.gl");
    py::module_::import("magnum.trade");
    #endif

    py::enum_<MeshTools::CompileFlag> compileFlag{m, "CompileFlag", "Mesh compilation flags"};
    compileFlag
        .value("NONE", MeshTools::CompileFlag{})
        .value("GENERATE_FLAT_NORMALS", MeshTools::CompileFlag::GenerateFlatNormals)
        .value("GENERATE_SMOOTH_NORMALS", MeshTools::CompileFlag::GenerateSmoothNormals);
    corrade::enumOperators(compileFlag);

    m
        .def("compile", [](const Trade::MeshData& meshData, MeshTools::CompileFlag flags) {
            return MeshTools::compile(meshData, flags);
        }, "Compile 3D mesh data", py::arg("mesh_data"), py::arg("flags") = MeshTools::CompileFlag{});
}

}

#ifndef MAGNUM_BUILD_STATIC
/* TODO: remove declaration when https://github.com/pybind/pybind11/pull/1863
   is released */
extern "C" PYBIND11_EXPORT PyObject* PyInit_meshtools();
PYBIND11_MODULE(meshtools, m) {
    magnum::meshtools(m);
}
#endif
