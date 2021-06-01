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

.. py:module:: magnum.platform

    Concrete implementations of :py:`Application` and :py:`WindowlessApplication`
    classes are available in submodules, and it's up to you which one you
    choose. For example:

    .. code:: py

        from magnum import platform
        import magnum.platform.sdl2

        class MyApp(platform.sdl2.Application):
            ...

    The same goes for :py:`WindowlessApplication` implementations, for example:

    .. code:: py

        from magnum import platform
        import magnum.platform.egl

        class MyApp(platform.egl.WindowlessApplication):
            ...

    Alternatively, if you want to narrow down the mention of a particular
    toolkit to just one line, you can also do:

    .. code:: py

        from magnum.platform.sdl2 import Application

        class MyApp(Application):
