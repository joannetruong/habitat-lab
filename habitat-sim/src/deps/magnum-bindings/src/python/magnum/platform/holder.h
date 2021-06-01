#ifndef magnum_platform_holder_h
#define magnum_platform_holder_h
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

namespace magnum { namespace platform { namespace {

/* Takes care of updating magnum::glContextOwner so it doesn't need to be
   duplicated in every application implementation */
template<class T> struct ApplicationHolder: std::unique_ptr<T> {
    explicit ApplicationHolder(T* object): std::unique_ptr<T>{object} {
        /* There's no real possibility to export a symbol from magnum.gl and
           access it from here (because there's no real possibility for a
           module to ensure another module is loaded before it in order to make
           the symbols resolve correctly; Corrade's PluginManager does that by
           having dependency info *external* to the module, that's the only
           way), so we're sharing the data using a bunch of very ugly
           allocations instead. Fortunately construction/destruction of an
           application happens *very seldom*, and gl.Context.current()
           hopefully also not that often. Yes, the parameter is a std::string.
           JOY. */
        auto* glContextOwner = static_cast<std::pair<const void*, const std::type_info*>*>(py::get_shared_data("magnumGLContextOwner"));
        if(!glContextOwner)
            py::set_shared_data("magnumGLContextOwner", glContextOwner = new std::pair<const void*, const std::type_info*>{});

        CORRADE_ASSERT(!glContextOwner->first, "Sorry, just one magnum.*.Application instance can exist at a time", );
        *glContextOwner = {object, &typeid(T)};
    }

    ApplicationHolder(ApplicationHolder<T>&&) noexcept = default;
    ApplicationHolder(const ApplicationHolder<T>&) = delete;

    ~ApplicationHolder() {
        auto* glContextOwner = static_cast<std::pair<const void*, const std::type_info*>*>(py::get_shared_data("magnumGLContextOwner"));
        CORRADE_INTERNAL_ASSERT(glContextOwner && glContextOwner->first == this->get());
        delete glContextOwner;
        py::set_shared_data("magnumGLContextOwner", nullptr);
    }

    ApplicationHolder<T>& operator=(ApplicationHolder<T>&&) noexcept = default;
    ApplicationHolder<T>& operator=(const ApplicationHolder<T>&) = default;
};

}}}

PYBIND11_DECLARE_HOLDER_TYPE(T, magnum::platform::ApplicationHolder<T>)

#endif
