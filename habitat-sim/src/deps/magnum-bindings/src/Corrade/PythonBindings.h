#ifndef Corrade_PythonBindings_h
#define Corrade_PythonBindings_h
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
#include <Corrade/Containers/ArrayView.h>

namespace Corrade {

inline pybind11::handle pyHandleFromInstance(const void* obj, const std::type_info& type) {
    /** @todo don't tell me there's no API for this either, ugh */
    return pybind11::detail::get_object_handle(obj, pybind11::detail::get_type_info(type));
}

template<class T> inline pybind11::handle pyHandleFromInstance(T& obj) {
    return pyHandleFromInstance(&obj, typeid(T));
}

inline pybind11::object pyObjectFromInstance(const void* obj, const std::type_info& type) {
    /* reinterpret_borrow?! seriously?! */
    return pybind11::reinterpret_borrow<pybind11::object>(pyHandleFromInstance(obj, type));
}

template<class T> inline pybind11::object pyObjectFromInstance(T& obj) {
    return pyObjectFromInstance(&obj, typeid(T));
}

template<class T> inline T& pyInstanceFromHandle(pybind11::handle handle) {
    /** @todo and this?! ugh! there's handle.cast() but that calls
        caster.load(handle, true) which we DO NOT WANT */
    /* Stolen from pybind11::class_::def_buffer(). Not sure what exactly it
       does, but I assume caster is implicitly convertible to Class& and thus
       can magically access the actual Class from the PyObject. */
    pybind11::detail::make_caster<T> caster;
    CORRADE_INTERNAL_ASSERT(caster.load(handle, /*convert=*/false));
    return caster;
}

/* py::cast() doesn't work on holder types because it takes const T&. Fuck
   that. The casting "just works" for function return types, so instead reuse
   the stuff that's done inside py::class_::def(). */
template<template<class> class Holder, class T> pybind11::object pyCastButNotShitty(Holder<T>&& holder) {
    static_assert(std::is_base_of<std::unique_ptr<T>, Holder<T>>::value,
        "holder should be a subclass of std::unique_ptr");
    /* Extracted out of cpp_function::initialize(), the cast_out alias. Not
       *exactly* sure about the return value policy or parent. Stealing the
       reference because cpp_function::dispatcher() seems to do that too (and
       using reinterpret_borrow makes tests fail with too high refcount) */
    return pybind11::reinterpret_steal<pybind11::object>(pybind11::detail::make_caster<Holder<T>>::cast(std::move(holder), pybind11::return_value_policy::move, {}));
}

/* Extracted the simplest case from py::type_caster_generic::load_impl() */
template<class T> T& pyObjectHolderFor(pybind11::handle obj, pybind11::detail::type_info* typeinfo) {
    /* So we don't need to bother with
       copyable_holder_caster::check_holder_compat() */
    static_assert(!std::is_copy_constructible<T>::value,
        "holder should be a move-only type");
    /* Assume the type is not subclassed on Python side */
    CORRADE_INTERNAL_ASSERT(Py_TYPE(obj.ptr()) == typeinfo->type);
    const pybind11::detail::value_and_holder vh = reinterpret_cast<pybind11::detail::instance*>(obj.ptr())->get_value_and_holder();
    /* And its already created and everything (i.e., we're not calling it in
       its own constructor) */
    CORRADE_INTERNAL_ASSERT(vh.holder_constructed());
    CORRADE_INTERNAL_ASSERT(vh.instance_registered());
    return vh.holder<T>();
}

template<template<class> class T, class U> T<U>& pyObjectHolderFor(U& obj) {
    /* Not using pyHandleFromInstance in order to avoid calling get_type_info
       more than once. I bet it involves some std::unordered_map access and
       that's like the slowest stuff ever. */
    pybind11::detail::type_info* typeinfo = pybind11::detail::get_type_info(typeid(U));
    return pyObjectHolderFor<T<U>>(pybind11::detail::get_object_handle(&obj, typeinfo), typeinfo);
}

/* This is a variant of https://github.com/pybind/pybind11/issues/1178,
   implemented on the client side instead of patching pybind itself */
template<class, bool> struct PyNonDestructibleBaseDeleter;
template<class T> struct PyNonDestructibleBaseDeleter<T, false> {
    void operator()(T*) { CORRADE_INTERNAL_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */ }
};
template<class T> struct PyNonDestructibleBaseDeleter<T, true> {
    void operator()(T* ptr) { delete ptr; }
};
template<class T, class... Args> using PyNonDestructibleClass = pybind11::class_<T, Args..., std::unique_ptr<T, PyNonDestructibleBaseDeleter<T, std::is_destructible<T>::value>>>;

}

#endif
