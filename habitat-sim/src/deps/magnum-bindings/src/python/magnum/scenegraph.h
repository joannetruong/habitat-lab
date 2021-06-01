#ifndef magnum_scenegraph_h
#define magnum_scenegraph_h
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
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Scene.h>

#include "Magnum/SceneGraph/PythonBindings.h"

#include "magnum/bootstrap.h"

namespace magnum {

template<class Transformation> void scene(py::class_<SceneGraph::Scene<Transformation>>& c) {
    c.def(py::init(), "Constructor");
}

template<UnsignedInt dimensions, class T, class Transformation> void object(py::class_<SceneGraph::Object<Transformation>, SceneGraph::PyObject<SceneGraph::Object<Transformation>>, SceneGraph::AbstractObject<dimensions, T>, SceneGraph::PyObjectHolder<SceneGraph::Object<Transformation>>>& c) {
    c
        .def(py::init_alias<SceneGraph::PyObject<SceneGraph::Object<Transformation>>*>(),
             "Constructor", py::arg("parent") = nullptr)
        .def(py::init_alias<SceneGraph::Scene<Transformation>*>(),
            "Constructor", py::arg("parent") = nullptr)

        /* Properties */
        .def_property_readonly("scene", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self) {
            return static_cast<SceneGraph::Scene<Transformation>*>(self.scene());
        }, "Scene or None if the object is not a part of any scene")
        .def_property("parent", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self) {
            return static_cast<SceneGraph::PyObject<SceneGraph::Object<Transformation>>*>(self.parent());
        }, [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, py::object parentobj) {
            SceneGraph::Object<Transformation>* parent;
            if(py::isinstance<SceneGraph::PyObject<SceneGraph::Object<Transformation>>>(parentobj))
                parent = py::cast<SceneGraph::PyObject<SceneGraph::Object<Transformation>>*>(parentobj);
            else if(py::isinstance<SceneGraph::Scene<Transformation>>(parentobj))
                parent = py::cast<SceneGraph::Scene<Transformation>*>(parentobj);
            else if(py::isinstance<py::none>(parentobj))
                parent = nullptr;
            else {
                PyErr_Format(PyExc_TypeError, "expected Scene, Object or None, got %A", parentobj.get_type().ptr());
                throw py::error_already_set{};
            }

            /* Decrease refcount if a parent is removed, increase it if a
               parent gets added */
            if(self.parent() && !parent) py::cast(&self).dec_ref();
            else if(!self.parent() && parent) py::cast(&self).inc_ref();

            self.setParent(parent);
        }, "Parent object or None if this is the root object")

        /* Transformation APIs common to all implementations */
        .def_property("transformation",
            &SceneGraph::PyObject<SceneGraph::Object<Transformation>>::transformation,
            &SceneGraph::PyObject<SceneGraph::Object<Transformation>>::setTransformation,
            "Object transformation")
        .def("absolute_transformation", &SceneGraph::PyObject<SceneGraph::Object<Transformation>>::absoluteTransformation,
             "Transformation relative to the root object")
        .def("reset_transformation", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self) {
            self.resetTransformation();
        }, "Reset the transformation")
        .def("translate", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const VectorTypeFor<dimensions, typename Transformation::DataType::Type>& vector) {
            self.translate(vector);
        }, "Translate the object")
        .def("translate_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const VectorTypeFor<dimensions, typename Transformation::DataType::Type>& vector) {
            self.translateLocal(vector);
        }, "Translate the object as a local transformation");
}

template<class Transformation> void objectTransform(py::class_<SceneGraph::Object<Transformation>, SceneGraph::PyObject<SceneGraph::Object<Transformation>>, SceneGraph::AbstractObject<Transformation::Dimensions, typename Transformation::Type>,  SceneGraph::PyObjectHolder<SceneGraph::Object<Transformation>>>& c) {
    c
        .def("transform", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const typename Transformation::DataType& transformation) {
            self.transform(transformation);
        }, "Transform the object")
        .def("transform_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const typename Transformation::DataType& transformation) {
            self.transformLocal(transformation);
        }, "Transform the object as a local transformation");
}

template<UnsignedInt dimensions, class T, class Transformation> void object2D(py::class_<SceneGraph::Object<Transformation>, SceneGraph::PyObject<SceneGraph::Object<Transformation>>, SceneGraph::AbstractObject<dimensions, T>,  SceneGraph::PyObjectHolder<SceneGraph::Object<Transformation>>>& c) {
    c
        .def("rotate", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle) {
            self.rotate(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle));
        }, "Rotate the object")
        .def("rotate_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle) {
            self.rotateLocal(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle));
        }, "Rotate the object as a local transformation");
}

template<UnsignedInt dimensions, class T, class Transformation> void object3D(py::class_<SceneGraph::Object<Transformation>, SceneGraph::PyObject<SceneGraph::Object<Transformation>>, SceneGraph::AbstractObject<dimensions, T>, SceneGraph::PyObjectHolder<SceneGraph::Object<Transformation>>>& c) {
    c
        .def("rotate", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle, const Math::Vector3<typename Transformation::DataType::Type>& normalizedAxis) {
            self.rotate(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle), normalizedAxis);
        }, "Rotate the object as a local transformation", py::arg("angle"), py::arg("normalized_axis"))
        .def("rotate_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle, const Math::Vector3<typename Transformation::DataType::Type>& normalizedAxis) {
            self.rotateLocal(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle), normalizedAxis);
        }, "Rotate the object as a local transformation", py::arg("angle"), py::arg("normalized_axis"))
        .def("rotate_x", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle) {
            self.rotateX(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle));
        }, "Rotate the object around X axis")
        .def("rotate_x_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle) {
            self.rotateXLocal(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle));
        }, "Rotate the object around X axis as a local transformation")
        .def("rotate_y", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle) {
            self.rotateY(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle));
        }, "Rotate the object around Y axis")
        .def("rotate_y_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle) {
            self.rotateYLocal(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle));
        }, "Rotate the object around Y axis as a local transformation")
        .def("rotate_z", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle) {
            self.rotateZ(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle));
        }, "Rotate the object around Z axis")
        .def("rotate_z_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const Radd angle) {
            self.rotateZLocal(static_cast<Math::Rad<typename Transformation::DataType::Type>>(angle));
        }, "Rotate the object around Z axis as a local transformation");
}

template<UnsignedInt dimensions, class T, class Transformation> void objectScale(py::class_<SceneGraph::Object<Transformation>, SceneGraph::PyObject<SceneGraph::Object<Transformation>>, SceneGraph::AbstractObject<dimensions, T>, SceneGraph::PyObjectHolder<SceneGraph::Object<Transformation>>>& c) {
    c
        .def("scale", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const VectorTypeFor<dimensions, typename Transformation::DataType::Type>& vector) {
            self.scale(vector);
        }, "Scale the object")
        .def("scale_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const VectorTypeFor<dimensions, typename Transformation::DataType::Type>& vector) {
            self.scaleLocal(vector);
        }, "Scale the object as a local transformation");
}

template<class Transformation> void objectReflect(py::class_<SceneGraph::Object<Transformation>, SceneGraph::PyObject<SceneGraph::Object<Transformation>>, SceneGraph::AbstractObject<Transformation::Dimensions, typename Transformation::Type>, SceneGraph::PyObjectHolder<SceneGraph::Object<Transformation>>>& c) {
    c
        .def("reflect", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const VectorTypeFor<Transformation::Dimensions, typename Transformation::Type>& vector) {
            self.reflect(vector);
        }, "Reflect the object")
        .def("reflect_local", [](SceneGraph::PyObject<SceneGraph::Object<Transformation>>& self, const VectorTypeFor<Transformation::Dimensions, typename Transformation::Type>& vector) {
            self.reflectLocal(vector);
        }, "Reflect the object as a local transformation");
}

void scenegraphMatrix(py::module_& m);
void scenegraphTrs(py::module_& m);

}

#endif
