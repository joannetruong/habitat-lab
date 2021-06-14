#ifndef Corrade_Containers_OptionalPythonBindings_h
#define Corrade_Containers_OptionalPythonBindings_h
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
#include <Corrade/Containers/Optional.h>

namespace pybind11 { namespace detail {

/* pybind11/stl.h has optional_caster for this, but that relies on a value_type
   typedef that Optional doesn't have, so adapting a copy of it, also without
   std::is_lvalue_reference<T>::value, which is not a thing here */
template<class T> struct type_caster<Corrade::Containers::Optional<T>> {
    using value_conv = make_caster<T>;

    template<class T_> static handle cast(T_&& src, const return_value_policy policy, const handle parent) {
        if(!src) return none{}.inc_ref();
        return value_conv::cast(*std::forward<T_>(src), return_value_policy_override<T>::policy(policy), parent);
    }

    bool load(const handle src, bool convert) {
        if(!src) return false;

        /* default-constructed value is already empty */
        if(src.is_none()) return true;

        value_conv inner_caster;
        if(!inner_caster.load(src, convert)) return false;

        value.emplace(cast_op<T&&>(std::move(inner_caster)));
        return true;
    }

    PYBIND11_TYPE_CASTER(Corrade::Containers::Optional<T>, _("Optional[") + value_conv::name + _("]"));
};

}}

#endif
