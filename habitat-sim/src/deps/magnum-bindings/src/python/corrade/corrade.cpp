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
#include <Corrade/configure.h>

#include "corrade/bootstrap.h"

#ifdef CORRADE_BUILD_STATIC
#include "corrade/staticconfigure.h"
#endif

namespace py = pybind11;

/* TODO: remove declaration when https://github.com/pybind/pybind11/pull/1863
   is released */
extern "C" PYBIND11_EXPORT PyObject* PyInit__corrade();
PYBIND11_MODULE(_corrade, m) {
    m.doc() = "Root Corrade module";
    m.attr("BUILD_STATIC") =
        #ifdef CORRADE_BUILD_STATIC
        true
        #else
        false
        #endif
        ;
    m.attr("BUILD_MULTITHREADED") =
        #ifdef CORRADE_BUILD_MULTITHREADED
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_APPLE") =
        #ifdef CORRADE_TARGET_APPLE
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_IOS") =
        #ifdef CORRADE_TARGET_IOS
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_IOS_SIMULATOR") =
        #ifdef CORRADE_TARGET_IOS_SIMULATOR
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_UNIX") =
        #ifdef CORRADE_TARGET_UNIX
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_WINDOWS") =
        #ifdef CORRADE_TARGET_WINDOWS
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_WINDOWS_RT") =
        #ifdef CORRADE_TARGET_WINDOWS_RT
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_EMSCRIPTEN") =
        #ifdef CORRADE_TARGET_EMSCRIPTEN
        true
        #else
        false
        #endif
        ;
    m.attr("TARGET_ANDROID") =
        #ifdef CORRADE_TARGET_ANDROID
        true
        #else
        false
        #endif
        ;
    /* Not exposing CORRADE_PLUGINMANAGER_NO_DYNAMIC_PLUGIN_SUPPORT as this
       is a plugin itself and so if this works, plugin manager should too */

    /* In case Corrade is a bunch of static libraries, put everything into a
       single shared lib to make it easier to install (which is the point of
       static builds) and avoid issues with multiply-defined global symbols.

       These need to be defined in the order they depend on. */
    #ifdef CORRADE_BUILD_STATIC
    py::module_ containers = m.def_submodule("containers");
    corrade::containers(containers);

    #ifdef Corrade_PluginManager_FOUND
    py::module_ pluginmanager = m.def_submodule("pluginmanager");
    corrade::pluginmanager(pluginmanager);
    #endif
    #endif
}
