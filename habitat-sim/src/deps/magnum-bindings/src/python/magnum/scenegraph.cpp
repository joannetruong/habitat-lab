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
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/AbstractObject.h>

#include "magnum/scenegraph.h"

namespace magnum {

namespace {

template<UnsignedInt dimensions, class T> struct PyDrawable: SceneGraph::PyFeature<SceneGraph::Drawable<dimensions, T>> {
    explicit PyDrawable(SceneGraph::AbstractObject<dimensions, T>& object, SceneGraph::DrawableGroup<dimensions, T>* drawables): SceneGraph::PyFeature<SceneGraph::Drawable<dimensions, T>>{object, drawables} {}

    void draw(const MatrixTypeFor<dimensions, T>& transformationMatrix, SceneGraph::Camera<dimensions, T>& camera) override {
        PYBIND11_OVERLOAD_PURE_NAME(
            void,
            PyDrawable,
            "draw",
            draw,
            transformationMatrix,
            camera
        );
    }
};

template<UnsignedInt dimensions, class T> void abstractObject(py::class_<SceneGraph::AbstractObject<dimensions, T>, SceneGraph::PyObjectHolder<SceneGraph::AbstractObject<dimensions, T>>>& c) {
    c
        /* Matrix transformation APIs */
        .def("transformation_matrix", &SceneGraph::AbstractObject<dimensions, T>::transformationMatrix,
            "Transformation matrix")
        .def("absolute_transformation_matrix", &SceneGraph::AbstractObject<dimensions, T>::absoluteTransformationMatrix,
             "Transformation matrix relative to the root object");
}

template<class PyFeature, UnsignedInt dimensions, class Feature, class T> void featureGroup(py::class_<SceneGraph::FeatureGroup<dimensions, Feature, T>>& c) {
    c
        .def(py::init(), "Constructor")
        .def("__len__", &SceneGraph::FeatureGroup<dimensions, Feature, T>::size,
            "Count of features in the group")
        /* Get item. Fetching the already registered instance and returning
           that instead of wrapping the pointer again. Need to raise IndexError
           in order to allow iteration: https://docs.python.org/3/reference/datamodel.html#object.__getitem__ */
        .def("__getitem__", [](SceneGraph::FeatureGroup<dimensions, Feature, T>& self, std::size_t index) -> PyFeature& {
            if(index >= self.size())  {
                PyErr_SetNone(PyExc_IndexError);
                throw py::error_already_set{};
            }
            return static_cast<PyFeature&>(self[index]);
        }, "Feature at given index")
        .def("add", [](SceneGraph::FeatureGroup<dimensions, Feature, T>& self, PyFeature& feature) {
            self.add(feature);
        }, "Add a feature to the group")
        .def("remove", [](SceneGraph::FeatureGroup<dimensions, Feature, T>& self, PyFeature& feature) {
            self.add(feature);
        }, "Remove a feature from the group");
}

template<UnsignedInt dimensions, class T> void feature(py::class_<SceneGraph::AbstractFeature<dimensions, T>, SceneGraph::PyFeature<SceneGraph::AbstractFeature<dimensions, T>>, SceneGraph::PyFeatureHolder<SceneGraph::AbstractFeature<dimensions, T>>>& c) {
    c
        .def(py::init_alias<SceneGraph::AbstractObject<dimensions, T>&>(),
            "Constructor", py::arg("object"))
        .def_property_readonly("object", [](SceneGraph::AbstractFeature<dimensions, T>& self) -> SceneGraph::AbstractObject<dimensions, T>& {
            return self.object();
        }, "Object holding this feature");
}

template<UnsignedInt dimensions, class T> void drawable(py::class_<SceneGraph::Drawable<dimensions, T>, SceneGraph::AbstractFeature<dimensions, T>, PyDrawable<dimensions, T>, SceneGraph::PyFeatureHolder<SceneGraph::Drawable<dimensions, T>>>& c) {
    c
        .def(py::init_alias<SceneGraph::AbstractObject<dimensions, T>&, SceneGraph::DrawableGroup<dimensions, T>*>(),
            "Constructor", py::arg("object"), py::arg("drawables") = nullptr)
        .def_property_readonly("drawables", [](PyDrawable<dimensions, T>& self) {
            return self.drawables();
        }, "Group containing this drawable")
        .def("draw", [](PyDrawable<dimensions, T>& self, const MatrixTypeFor<dimensions, T>& transformationMatrix, SceneGraph::Camera<dimensions, T>& camera) {
            self.draw(transformationMatrix, camera);
        }, "Draw the object using given camera", py::arg("transformation_matrix"), py::arg("camera"));
}

template<UnsignedInt dimensions, class T> void camera(py::class_<SceneGraph::Camera<dimensions, T>, SceneGraph::AbstractFeature<dimensions, T>, SceneGraph::PyFeature<SceneGraph::Camera<dimensions, T>>, SceneGraph::PyFeatureHolder<SceneGraph::Camera<dimensions, T>>>& c) {
    c
        .def(py::init_alias<SceneGraph::AbstractObject<dimensions, T>&>(),
            "Constructor", py::arg("object"))
        .def_property("aspect_ratio_policy", &SceneGraph::Camera<dimensions, T>::aspectRatioPolicy,
            /* Using a lambda because the setter has method chaining */
            [](SceneGraph::Camera<dimensions, T>& self, SceneGraph::AspectRatioPolicy policy) {
                self.setAspectRatioPolicy(policy);
            }, "Aspect ratio policy")
        .def_property_readonly("camera_matrix", &SceneGraph::Camera<dimensions, T>::cameraMatrix,
            "Camera matrix")
        .def_property("projection_matrix", &SceneGraph::Camera<dimensions, T>::projectionMatrix,
            /* Using a lambda because the setter has method chaining */
            [](SceneGraph::Camera<dimensions, T>& self, const MatrixTypeFor<dimensions, T>& matrix) {
                self.setProjectionMatrix(matrix);
            }, "Projection matrix")
        .def("projection_size", &SceneGraph::Camera<dimensions, T>::projectionSize,
            "Size of (near) XY plane in current projection")
        .def_property("viewport", &SceneGraph::Camera<dimensions, T>::viewport,
            &SceneGraph::Camera<dimensions, T>::setViewport,
            "Viewport size")
        .def("draw", static_cast<void(SceneGraph::Camera<dimensions, T>::*)(SceneGraph::DrawableGroup<dimensions, T>&)>(&SceneGraph::Camera<dimensions, T>::draw),
            "Draw");
}

}

void scenegraph(py::module_& m) {
    m.doc() = "Scene graph library";

    /* Abstract objects. Returned from feature.object, so need to be registered
       as well. */
    {
        py::class_<SceneGraph::AbstractObject2D, SceneGraph::PyObjectHolder<SceneGraph::AbstractObject2D>> abstractObject2D{m, "AbstractObject2D", "Base object for two-dimensional scenes"};
        py::class_<SceneGraph::AbstractObject3D, SceneGraph::PyObjectHolder<SceneGraph::AbstractObject3D>> abstractObject3D{m, "AbstractObject3D", "Base object for three-dimensional scenes"};
        abstractObject(abstractObject2D);
        abstractObject(abstractObject3D);
    }

    /* Drawables, camera */
    {
        py::enum_<SceneGraph::AspectRatioPolicy>{m, "AspectRatioPolicy", "Camera aspect ratio policy"}
            .value("NOT_PRESERVED", SceneGraph::AspectRatioPolicy::NotPreserved)
            .value("EXTEND", SceneGraph::AspectRatioPolicy::Extend)
            .value("CLIP", SceneGraph::AspectRatioPolicy::Clip);

        py::class_<SceneGraph::DrawableGroup2D> drawableGroup2D{m, "DrawableGroup2D", "Group of drawables for two-dimensional float scenes"};
        py::class_<SceneGraph::DrawableGroup3D> drawableGroup3D{m, "DrawableGroup3D", "Group of drawables for three-dimensional float scenes"};

        py::class_<SceneGraph::AbstractFeature2D, SceneGraph::PyFeature<SceneGraph::AbstractFeature2D>, SceneGraph::PyFeatureHolder<SceneGraph::AbstractFeature2D>> feature2D{m, "AbstractFeature2D", "Base for two-dimensional float features"};
        py::class_<SceneGraph::AbstractFeature3D, SceneGraph::PyFeature<SceneGraph::AbstractFeature3D>, SceneGraph::PyFeatureHolder<SceneGraph::AbstractFeature3D>> feature3D{m, "AbstractFeature3D", "Base for three-dimensional float features"};
        feature(feature2D);
        feature(feature3D);

        py::class_<SceneGraph::Drawable2D, SceneGraph::AbstractFeature2D, PyDrawable<2, Float>, SceneGraph::PyFeatureHolder<SceneGraph::Drawable2D>> drawable2D{m, "Drawable2D", "Drawable for two-dimensional float scenes"};
        py::class_<SceneGraph::Drawable3D, SceneGraph::AbstractFeature3D, PyDrawable<3, Float>, SceneGraph::PyFeatureHolder<SceneGraph::Drawable3D>> drawable3D{m, "Drawable3D", "Drawable for three-dimensional float scenes"};

        py::class_<SceneGraph::Camera2D, SceneGraph::AbstractFeature2D, SceneGraph::PyFeature<SceneGraph::Camera2D>, SceneGraph::PyFeatureHolder<SceneGraph::Camera2D>> camera2D{m, "Camera2D", "Camera for two-dimensional float scenes"};
        py::class_<SceneGraph::Camera3D, SceneGraph::AbstractFeature3D, SceneGraph::PyFeature<SceneGraph::Camera3D>, SceneGraph::PyFeatureHolder<SceneGraph::Camera3D>> camera3D{m, "Camera3D", "Camera for three-dimensional float scenes"};

        featureGroup<PyDrawable<2, Float>>(drawableGroup2D);
        featureGroup<PyDrawable<3, Float>>(drawableGroup3D);
        drawable(drawable2D);
        drawable(drawable3D);

        camera(camera2D);
        camera(camera3D);
    }

    /* Concrete transformation implementations */
    magnum::scenegraphMatrix(m);
    magnum::scenegraphTrs(m);
}

}

#ifndef MAGNUM_BUILD_STATIC
/* TODO: remove declaration when https://github.com/pybind/pybind11/pull/1863
   is released */
extern "C" PYBIND11_EXPORT PyObject* PyInit_scenegraph();
PYBIND11_MODULE(scenegraph, m) {
    magnum::scenegraph(m);
}
#endif
