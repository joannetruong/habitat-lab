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

#include <Magnum/SceneGraph/TranslationRotationScalingTransformation2D.h>
#include <Magnum/SceneGraph/TranslationRotationScalingTransformation3D.h>

#include "scenegraph.h"

namespace magnum {

namespace {

template<class Transformation> void objectTrs(py::class_<SceneGraph::Object<Transformation>, SceneGraph::PyObject<SceneGraph::Object<Transformation>>, SceneGraph::AbstractObject<Transformation::Dimensions, typename Transformation::Type>, SceneGraph::PyObjectHolder<SceneGraph::Object<Transformation>>>& c) {
    c
        .def_property("translation",
            &SceneGraph::Object<Transformation>::translation,
            &SceneGraph::Object<Transformation>::setTranslation,
            "Object translation")
        .def_property("rotation",
            &SceneGraph::Object<Transformation>::rotation,
            &SceneGraph::Object<Transformation>::setRotation,
            "Object rotation")
        .def_property("scaling",
            &SceneGraph::Object<Transformation>::scaling,
            &SceneGraph::Object<Transformation>::setScaling,
            "Object scaling");
}

}

void scenegraphTrs(py::module_& m) {
    py::module_ matrix = m.def_submodule("trs");
    matrix.doc() = "Translation/rotation/scaling-based scene graph implementation";

    py::class_<SceneGraph::Scene<SceneGraph::TranslationRotationScalingTransformation2D>> scene2D_{matrix, "Scene2D", "Two-dimensional scene with TRS-based transformation implementation"};
    scene(scene2D_);

    py::class_<SceneGraph::Scene<SceneGraph::TranslationRotationScalingTransformation3D>> scene3D_{matrix, "Scene3D", "Three-dimensional scene with TRS-based transformation implementation"};
    scene(scene3D_);

    py::class_<SceneGraph::Object<SceneGraph::TranslationRotationScalingTransformation2D>, SceneGraph::PyObject<SceneGraph::Object<SceneGraph::TranslationRotationScalingTransformation2D>>, SceneGraph::AbstractObject2D, SceneGraph::PyObjectHolder<SceneGraph::Object<SceneGraph::TranslationRotationScalingTransformation2D>>> object2D_{matrix, "Object2D", "Two-dimensional object with TRS-based transformation implementation"};
    object(object2D_);
    object2D(object2D_);
    objectScale(object2D_);
    objectTrs(object2D_);

    py::class_<SceneGraph::Object<SceneGraph::TranslationRotationScalingTransformation3D>, SceneGraph::PyObject<SceneGraph::Object<SceneGraph::TranslationRotationScalingTransformation3D>>, SceneGraph::AbstractObject3D, SceneGraph::PyObjectHolder<SceneGraph::Object<SceneGraph::TranslationRotationScalingTransformation3D>>> object3D_{matrix, "Object3D", "Three-dimensional object with TRS-based transformation implementation"};
    object(object3D_);
    object3D(object3D_);
    objectScale(object3D_);
    objectTrs(object3D_);
}

}
