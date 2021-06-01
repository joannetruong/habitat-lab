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

Downloading and building
########################

.. role:: sh(code)
    :language: sh

:summary: Installation guide for the Python bindings.
:ref-prefix:
    corrade
    magnum

Building of Python bindings is a similar process to
:dox:`building Magnum itself <building>` with an additional step involving
Python setuptools. Minimal set of tools and libraries required for building is:

-   C++ compiler with good C++11 support. Compilers which are tested to have
    everything needed are **GCC** >= 4.8.1, **Clang** >= 3.3 and **MSVC**
    >= 2015. On Windows you can also use **MinGW-w64**.
-   **CMake** >= 3.1
-   **Corrade** and **Magnum** installed
    :dox:`as described in their docs <building>`
-   **Python** >= 3.5 and :gh:`pybind11 <pybind/pybind11>`

`Prepared packages`_
====================

`ArchLinux packages`_
---------------------

In ``package/archlinux`` there is a development package, similar to the ones
in Magnum itself. They allow you to build and install the package directly from
the source tree.

.. code:: sh

    git clone git://github.com/mosra/magnum-bindings && cd magnum-bindings
    cd package/archlinux
    makepkg -fp PKGBUILD

The PKGBUILD also contains a :sh:`check()` function which will run all unit
tests before packaging. That might sometimes fail or take too long, pass
``--nocheck`` to ``makepkg`` to skip that.

Once built, install the package using ``pacman``:

.. code:: sh

    sudo pacman -U magnum-bindings-*.pkg.tar.zst

`Homebrew formulas for macOS`_
------------------------------

macOS `Homebrew <https://brew.sh>`_ formulas building the latest Git revision
are in the ``package/homebrew`` directory. Either use the ``*.rb`` files
directly or use the tap at https://github.com/mosra/homebrew-magnum. Right now,
there's no stable release of Python bindings yet, so you need to install the
latest Git revision of all Magnum projects instead:

.. code:: sh

    brew install --HEAD mosra/magnum/corrade
    brew install --HEAD mosra/magnum/magnum
    brew install --HEAD mosra/magnum/magnum-bindings

When installing from the ``*.rb`` files you need to install the
:dox:`Corrade <building-corrade-packages-brew>` and
:dox:`Magnum <building-packages-brew>` Homebrew packages first. If you want to
pass additional flags to CMake or ``setup.py`` or enable / disable additional
features, edit the ``*.rb`` file.

There are also Homebrew packages for
:dox:`Magnum Plugins <building-plugins-packages-brew>`,
:dox:`Magnum Integration <building-integration-packages-brew>`,
:dox:`Magnum Extras <building-extras-packages-brew>` and
:dox:`Magnum Examples <building-examples-packages-brew>`.

`Manual build`_
===============

The source is available on GitHub at https://github.com/mosra/magnum-bindings.
Clone the repository with your favorite IDE or Git GUI, download currrent
snapshot as a compressed archive or use the command line:

.. code:: sh

    git clone git://github.com/mosra/magnum-bindings.git

Assuming a Unix-based OS, the first step is to build the native libraries. The
bindings will be generated for all Corrade and Magnum libraries that are found,
ignoring the ones which aren't. If Corrade, Magnum and pybind11 are not in a
default location known to CMake, add their path to ``CMAKE_PREFIX_PATH``.

.. code:: sh

    mkdir build && cd build
    cmake .. \
        -DWITH_PYTHON=ON
    make

Note that pybind11 compilation is quite time- and memory-hungry, so you might
not want to run the build on all cores on memory-constrained systems. In the
build directory, CMake will create the desired Python package layout, meaning
the bindings can be used directly if you ``cd`` into ``build/src/python/magnum``.
For installing into a system-wide location, CMake generates a ``setup.py``
containing location of all built libraries for use with Python setuptools:

.. code:: sh

    cd build/src/python/magnum
    python setup.py install # or python3, sudo might be needed

`Static build`_
---------------

In case Corrade or Magnum is built with :dox:`CORRADE_BUILD_STATIC` /
:dox:`MAGNUM_BUILD_STATIC`, the corresponding bindings are compiled into a
single dynamic module instead of one module per Corrade/Magnum library.

`Running unit tests`_
---------------------

Essential functionality of the bindings is tested using Python's builtin
``unittest`` module. The tests currently assume a CMake build directory with
all binaries already built located in a ``build/`` directory in project root,
running them is then a matter of:

.. code:: sh

    cd src/python/magnum
    python -m unittest

.. block-default:: Disabling GL tests

    If the tests detect that one of
    :ref:`platform.WindowlessApplication <platform.egl.WindowlessApplication>`\ s
    is present, GL tests (suffixed with ``_gl``) will be run as well. In order
    to disable them (for example when running on a headless CI), set the
    :sh:`$MAGNUM_SKIP_GL_TESTS` environment variable to ``ON``:

    .. code:: sh

        MAGNUM_SKIP_GL_TESTS=ON python -m unittest

For code coverage, `coverage.py <https://coverage.readthedocs.io/>`_ is used.
Get it via ``pip`` or as a system package.

.. code:: sh

    pip install coverage # sudo might be needed

Running the unit tests with coverage enabled is then a matter of executing the
following commands, the resulting HTML overview is located in
``htmlcov/index.html``:

.. code:: sh

    cd src/python/magnum
    coverage run -m unittest
    coverage html

`Continuous Integration`_
=========================

In ``package/ci/`` there is a ``travis.yml`` file that compiles and tests the
bindings on Linux GCC 4.8 + CMake 3.1 and on macOS, online at
https://travis-ci.com/mosra/magnum-bindings. For Windows there is an
``appveyor.yml`` testing on Windows with MSVC 2017 and 2019, online at
https://ci.appveyor.com/project/mosra/magnum-bindings. Code coverage for both
the C++ bindings code and Python side is reported to
https://codecov.io/gh/mosra/magnum-bindings.
