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

Credits
#######

:ref-prefix:
    corrade
    magnum
:summary: Third-party dependencies and their corresponding license information;
    people and organizations that contributed to Magnum Python Bindings.

`Third-party components`_
=========================

.. TODO: ffs doxygen SORT YOUR SHIT OUT, why can't I link to
    credits-third-party?!
.. role:: doxygen-you-fool(link)
    :class: m-doc-external

While Magnum Python Bindings themselves don't depend on much, a lot of
third-party components is used transitively from the Magnum C++ implementation.
Please see the `main page <std:doc:index#license>` for license of Magnum Python
Bindings themselves and the
:doxygen-you-fool:`Third-party components <https://doc.magnum.graphics/magnum/credits-third-party.html>`
page of C++ docs for a detailed overview of all used components. The list below
uses the same color-coding scheme for easier overview:

`Magnum Python Bindings <std:doc:index>`
    Bindings generated with :gh:`pybind11 <pybind/pybind11>`, released under a
    :label-success:`BSD-style license`
    (`license text <https://github.com/pybind/pybind11/blob/master/LICENSE>`_,
    `choosealicense.com <https://choosealicense.com/licenses/bsd-3-clause/>`_).
    It requires attribution for public use.

`Contributors`_
===============

Listing only people with code contributions or other significant work, because
otherwise there's too many :) There's also a
:dox:`similar list for Corrade <corrade-credits-contributors>` and
:dox:`Magnum <credits-contributors>` themselves. Big thanks to everyone
involved!

.. class:: m-text-center m-text m-dim

    Are the below lists missing your name or something's wrong?
    `Let us know! <https://magnum.graphics/contact/>`_

-   **Aaron Gokaslan** (:gh:`Skylion007`) --- minor performance fixes
-   **Cameron Egbert** (:gh:`cegbertOculus`) --- initial Windows port
-   **John Laxson** (:gh:`jlaxson`) --- Homebrew package improvements
-   **Vladimir Gamalyan** (:gh:`vladimirgamalyan`) --- expanding
    :ref:`gl.Renderer` bindings
