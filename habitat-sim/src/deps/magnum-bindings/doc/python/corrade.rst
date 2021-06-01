..
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
..

.. roles used for all other docs

.. role:: cpp(code)
    :language: c++
.. role:: py(code)
    :language: py
.. role:: sh(code)
    :language: sh

.. doctest setup
    >>> from corrade import *

.. py:module:: corrade
    :data BUILD_STATIC: Static library build
    :data BUILD_MULTITHREADED: Multi-threaded build
    :data TARGET_UNIX: Unix target
    :data TARGET_APPLE: Apple target
    :data TARGET_IOS: iOS target
    :data TARGET_IOS_SIMULATOR: iOS simulator target
    :data TARGET_WINDOWS: Windows target
    :data TARGET_WINDOWS_RT: Windows RT target
    :data TARGET_EMSCRIPTEN: Emscripten target
    :data TARGET_ANDROID: Android target
