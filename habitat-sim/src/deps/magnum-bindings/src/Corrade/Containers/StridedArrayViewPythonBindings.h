#ifndef Corrade_Containers_StridedArrayViewPythonBindings_h
#define Corrade_Containers_StridedArrayViewPythonBindings_h
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
#include <Corrade/Containers/StridedArrayView.h>

namespace Corrade { namespace Containers {

namespace Implementation {

/* For maintainability please keep in the same order as
   https://docs.python.org/3/library/struct.html#format-characters */
template<class T> constexpr const char* formatString() {
    static_assert(sizeof(T) == 0, "format string unknown for this type, supply it explicitly");
    return {};
}
/* Representing bytes as unsigned. Not using 'c' because then it behaves
   differently from bytes/bytearray, where you can do `a[0] = ord('A')`. */
template<> constexpr const char* formatString<char>() { return "B"; }
template<> constexpr const char* formatString<std::int8_t>() { return "b"; }
template<> constexpr const char* formatString<std::uint8_t>() { return "B"; }
template<> constexpr const char* formatString<std::int16_t>() { return "h"; }
template<> constexpr const char* formatString<std::uint16_t>() { return "H"; }
template<> constexpr const char* formatString<std::int32_t>() { return "i"; }
template<> constexpr const char* formatString<std::uint32_t>() { return "I"; }
/* *not* l / L, that's 4 bytes in Python */
template<> constexpr const char* formatString<std::int64_t>() { return "q"; }
template<> constexpr const char* formatString<std::uint64_t>() { return "Q"; }
/** @todo how to represent std::size_t? conflicts with uint32_t/uint64_t above */
/** @todo half? take from Magnum? */
template<> constexpr const char* formatString<float>() { return "f"; }
template<> constexpr const char* formatString<double>() { return "d"; }

template<class T, class U> struct PyStridedArrayViewSetItem;
template<class U> struct PyStridedArrayViewSetItem<const char, U> {
    /* __setitem__ is not even exposed for immutable views so this is fine */
    constexpr static std::nullptr_t set = nullptr;
};
template<class U> struct PyStridedArrayViewSetItem<char, U> {
    static void set(char* item, pybind11::handle object) {
        *reinterpret_cast<U*>(item) = pybind11::cast<U>(object);
    }
};

template<unsigned, class> struct PyStridedElement;

}

template<unsigned dimensions, class T> class PyStridedArrayView: public StridedArrayView<dimensions, T> {
    public:
        /* Null function pointers should be okay as it shouldn't ever get to
           them -- IndexError gets fired first. Not really sure about the
           format, choosing bytes for safety. */
        /*implicit*/ PyStridedArrayView(): format{"B"}, getitem{} {}

        template<class U> explicit PyStridedArrayView(const StridedArrayView<dimensions, U>& view): PyStridedArrayView{view, Implementation::formatString<typename std::decay<U>::type>(), sizeof(U)} {}

        template<class U> explicit PyStridedArrayView(const StridedArrayView<dimensions, U>& view, const char* format, std::size_t itemsize): PyStridedArrayView<dimensions, T>{
            arrayCast<T>(view),
            format,
            itemsize,
            [](const char* item) {
                return pybind11::cast(*reinterpret_cast<const U*>(item));
            },
            Implementation::PyStridedArrayViewSetItem<T, U>::set
        } {}

        explicit PyStridedArrayView(const StridedArrayView<dimensions, T>& view, const char* format, std::size_t itemsize, pybind11::object(*getitem)(const char*), void(*setitem)(char*, pybind11::handle)): StridedArrayView<dimensions, T>{view}, format{format}, itemsize{itemsize}, getitem{getitem}, setitem{setitem} {}

        /* All APIs that are exposed by bindings and return a StridedArrayView
           have to return the wrapper now */

        typedef typename std::conditional<dimensions == 1, T&, PyStridedArrayView<dimensions - 1, T>>::type ElementType;

        ElementType operator[](std::size_t i) const {
            return Implementation::PyStridedElement<dimensions, T>::wrap(StridedArrayView<dimensions, T>::operator[](i), format, itemsize, getitem, setitem);
        }

        PyStridedArrayView<dimensions, T> slice(std::size_t begin, std::size_t end) const {
            return PyStridedArrayView<dimensions, T>{StridedArrayView<dimensions, T>::slice(begin, end), format, itemsize, getitem, setitem};
        }
        PyStridedArrayView<dimensions, T> slice(const typename StridedArrayView<dimensions, T>::Size& begin, const typename StridedArrayView<dimensions, T>::Size& end) const {
            return PyStridedArrayView<dimensions, T>{StridedArrayView<dimensions, T>::slice(begin, end), format, itemsize, getitem, setitem};
        }

        /* slice() with templated dimensions not used */
        /* slice(&T::member) not used */
        /* prefix(), suffix(), except() not used */

        PyStridedArrayView<dimensions, T> every(std::size_t skip) const {
            return PyStridedArrayView<dimensions, T>{StridedArrayView<dimensions, T>::every(skip), format, itemsize, getitem, setitem};
        }

        PyStridedArrayView<dimensions, T> every(const typename StridedArrayView<dimensions, T>::Stride& skip) const {
            return PyStridedArrayView<dimensions, T>{StridedArrayView<dimensions, T>::every(skip), format, itemsize, getitem, setitem};
        }

        template<unsigned dimensionA, unsigned dimensionB> PyStridedArrayView<dimensions, T> transposed() const {
            return PyStridedArrayView<dimensions, T>{StridedArrayView<dimensions, T>::template transposed<dimensionA, dimensionB>(), format, itemsize, getitem, setitem};
        }

        template<unsigned dimension> PyStridedArrayView<dimensions, T> flipped() const {
            return PyStridedArrayView<dimensions, T>{StridedArrayView<dimensions, T>::template flipped<dimension>(), format, itemsize, getitem, setitem};
        }

        template<unsigned dimension> PyStridedArrayView<dimensions, T> broadcasted(std::size_t size) const {
            return PyStridedArrayView<dimensions, T>{StridedArrayView<dimensions, T>::template broadcasted<dimension>(size), format, itemsize, getitem, setitem};
        }

        /* has to be public as it's accessed by the bindings directly */
        const char* format;
        std::size_t itemsize;
        pybind11::object(*getitem)(const char*);
        void(*setitem)(char*, pybind11::handle);
};

namespace Implementation {

template<unsigned dimensions, class T> struct PyStridedElement {
    static PyStridedArrayView<dimensions - 1, T> wrap(const StridedArrayView<dimensions - 1, T>& element, const char* format, std::size_t itemsize, pybind11::object(*getitem)(const char*), void(*setitem)(char*, pybind11::handle)) {
        return PyStridedArrayView<dimensions - 1, T>{element, format, itemsize, getitem, setitem};
    }
};

template<class T> struct PyStridedElement<1, T> {
    static T& wrap(T& element, const char*, std::size_t, pybind11::object(*)(const char*), void(*)(char*, pybind11::handle)) {
        return element;
    }
};

}

}}

#endif
