#ifndef corrade_pluginmanager_h
#define corrade_pluginmanager_h
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
#include <Corrade/PluginManager/Manager.h>

#include "Corrade/PythonBindings.h"

#include "corrade/bootstrap.h"

namespace Corrade { namespace PluginManager {

/* Stores additional stuff needed for proper refcounting of array views. Due
   to obvious reasons we can't subclass plugins so this is the only possible
   way. */
template<class T> struct PyPluginHolder: std::unique_ptr<T> {
    explicit PyPluginHolder(T*) {
        /* Pybind needs this signature, but it should never be called */
        CORRADE_INTERNAL_ASSERT_UNREACHABLE();
    }

    explicit PyPluginHolder(T* object, pybind11::object manager) noexcept: std::unique_ptr<T>{object}, manager{std::move(manager)} {}

    PyPluginHolder(PyPluginHolder<T>&&) noexcept = default;
    PyPluginHolder(const PyPluginHolder<T>&) = delete;
    PyPluginHolder<T>& operator=(PyPluginHolder<T>&&) noexcept = default;
    PyPluginHolder<T>& operator=(const PyPluginHolder<T>&) = default;

    ~PyPluginHolder() {
        /* On destruction, first `manager` and then the plugin would be
           destroyed, which would mean it asserts due to the manager being
           destructed while plugins are still around. To flip the order, we
           need to reset the pointer first */
        std::unique_ptr<T>::reset();
    }

    pybind11::object manager;
};

}}

PYBIND11_DECLARE_HOLDER_TYPE(T, Corrade::PluginManager::PyPluginHolder<T>)

namespace corrade {

template<class T> void plugin(py::class_<T, PluginManager::PyPluginHolder<T>>& c) {
    c
        .def_property_readonly("manager", [](const T& self) {
            return pyObjectHolderFor<PluginManager::PyPluginHolder>(self).manager;
        }, "Manager owning this plugin instance");
}

template<class T> void manager(py::class_<PluginManager::Manager<T>, PluginManager::AbstractManager>& c) {
    c
        .def(py::init<const std::string&>(), py::arg("plugin_directory") = std::string{}, "Constructor")
        .def("instantiate", [](PluginManager::Manager<T>& self, const std::string& plugin) {
            /* This causes a double lookup, but well... better than dying */
            if(!(self.loadState(plugin) & PluginManager::LoadState::Loaded)) {
                PyErr_Format(PyExc_RuntimeError, "plugin %s is not loaded", plugin.data());
                throw py::error_already_set{};
            }

            auto loaded = self.instantiate(plugin);
            if(!loaded) {
                PyErr_Format(PyExc_RuntimeError, "can't instantiate plugin %s", plugin.data());
                throw py::error_already_set{};
            }

            return PluginManager::PyPluginHolder<T>{loaded.release(), py::cast(self)};
        })
        .def("load_and_instantiate", [](PluginManager::Manager<T>& self, const std::string& plugin) {
            auto loaded = self.loadAndInstantiate(plugin);
            if(!loaded) {
                PyErr_Format(PyExc_RuntimeError, "can't load and instantiate plugin %s", plugin.data());
                throw py::error_already_set{};
            }

            return PluginManager::PyPluginHolder<T>{loaded.release(), py::cast(self)};
        });
}

}

#endif
