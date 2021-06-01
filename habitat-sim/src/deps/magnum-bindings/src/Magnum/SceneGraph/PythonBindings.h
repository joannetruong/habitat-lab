#ifndef Magnum_SceneGraph_PythonBindings_h
#define Magnum_SceneGraph_PythonBindings_h
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

#include <memory> /* :( */
#include <pybind11/pybind11.h>
#include <Corrade/Utility/Assert.h>

namespace Magnum { namespace SceneGraph {

/* This is a variant of https://github.com/pybind/pybind11/issues/1389. If the
   object has a parent, its refcount gets increased in order to avoid it being
   deleted by Python too soon. The refcount gets decreased when the parent is
   removed again or the parent gets deleted. I thought this would be doable
   inside py::init() as

    .def(py::init([](SceneGraph::Scene<Transformation>* parent) {
        auto self = new PyObject<Transformation>{parent};
        if(parent) py::cast(self).inc_ref();
        return self;
    }))

   but FOR SOME REASON py::cast(self) inside py::init() returns a different
   underlying PyObject pointer, so it only leads to crashes. */

template<class T> struct PyObjectHolder: std::unique_ptr<T> {
    explicit PyObjectHolder(T* object): std::unique_ptr<T>{object} {
        CORRADE_INTERNAL_ASSERT(object);
        if(object->parent()) pybind11::cast(object).inc_ref();
    }
};

template<class T> struct PyFeatureHolder: std::unique_ptr<T> {
    explicit PyFeatureHolder(T* object): std::unique_ptr<T>{object} {
        CORRADE_INTERNAL_ASSERT(object);
        pybind11::cast(object).inc_ref();
    }
};

/* Hey this needs docs. */

/* This template parameter can't be just Object, as that makes MSVC confused
   when CRTP'ing something else than a class named Object -- it fails inside
   the Object{} call in the constructor, saying `error C2614:
   'Magnum::SceneGraph::PyObject<SomeOtherName>': illegal member initialization:
   'Object' is not a base or member`. */
template<class Object_> class PyObject: public Object_ {
    public:
        template<class ...Args> explicit PyObject(Args&&... args): Object_{std::forward<Args>(args)...} {}

        PyObject(const PyObject<Object_>&) = delete;
        PyObject(PyObject<Object_>&&) = delete;

        PyObject<Object_>& operator=(const PyObject<Object_>&) = delete;
        PyObject<Object_>& operator=(PyObject<Object_>&&) = delete;

    private:
        void doErase() override {
            /* When deleting a parent, disconnect this from the parent instead
               of deleting it. Deletion is then handled by Python itself. */
            CORRADE_INTERNAL_ASSERT(Object_::parent());
            Object_::setParent(nullptr);
            pybind11::cast(this).dec_ref();
        }
};

template<class Feature> class PyFeature: public Feature {
    public:
        template<class ...Args> explicit PyFeature(Args&&... args): Feature{std::forward<Args>(args)...} {}

        PyFeature(const PyFeature<Feature>&) = delete;
        PyFeature(PyFeature<Feature>&&) = delete;

        PyFeature<Feature>& operator=(const PyFeature<Feature>&) = delete;
        PyFeature<Feature>& operator=(PyFeature<Feature>&&) = delete;

    private:
        void doErase() override {
            /* When deleting the holder object, disconnect this from that
               object instead of deleting it. This makes it rather useless, but
               better than having dangling memory or double deletion. This is
               of course not allowed by the C++ API due to private inheritance
               so we have to reinterpret self as the list instead. UGLY. */
            auto& listItem = reinterpret_cast<Containers::LinkedListItem<SceneGraph::AbstractFeature<Feature::Dimensions, typename Feature::Type>, SceneGraph::AbstractObject<Feature::Dimensions, typename Feature::Type>>&>(*this);

            CORRADE_INTERNAL_ASSERT(listItem.list());
            listItem.list()->features().cut(this);
            pybind11::cast(this).dec_ref();
        }
};

}}

PYBIND11_DECLARE_HOLDER_TYPE(T, Magnum::SceneGraph::PyObjectHolder<T>)
PYBIND11_DECLARE_HOLDER_TYPE(T, Magnum::SceneGraph::PyFeatureHolder<T>)

#endif
