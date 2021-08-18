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

Changelog
#########

:ref-prefix:
    corrade
    magnum
:some_directive: TODO: why can't the title below "just work" and I have to
    add some bogus content before?

`Changes since 2020.06`_
========================

-   Exposed missing :ref:`Vector4` constructor from a :ref:`Vector3` and a
    W component and :ref:`Vector3` from :ref:`Vector2` and a Z component
-   Renamed :py:`Matrix3.from()` / :py:`Matrix4.from()` to :ref:`Matrix3.from_()`
    / :ref:`Matrix4.from_()` because :py:`from` is a Python keyword and it
    would be silly to have to write :py:`getattr(Matrix4, 'from')` just to use
    these APIs
-   Exposed :ref:`gl.Renderer.set_blend_function()`,
    :ref:`gl.Renderer.set_blend_equation()` and related enums (see :gh:`mosra/magnum-bindings#9`)
-   Exposed :ref:`gl.Renderer.Feature.CLIP_DISTANCEn <gl.Renderer.Feature.CLIP_DISTANCE0>`
    values that are new since 2020.06
-   Exposed new instancing, normal-mapping-related and lighting features in
    :ref:`shaders.PhongGL`
-   Renamed all helper ``Python.h`` headers to ``PythonBindings.h`` to avoid
    issues with shitty IDE indexers such as Eclipse, confusing these with
    Python's ``<Python.h>``
-   Applied minor performance fixes suggested by Clang Tidy (see :gh:`mosra/magnum-bindings#10`)
-   Linux and macOS builds were migrated from Travis to Circle CI
-   On CMake 3.16 and newer, `FindMagnumBindings.cmake` can provide additional
    details if some component is not found
-   The Homebrew package now uses `std_cmake_args` instead of hardcoded build
    type and install prefix, which resolves certain build issues (see
    :gh:`mosra/homebrew-magnum#6`)
-   Added a caster for :dox:`Containers::Optional`, allowing it to be used
    directly in function signatures and showing up on the Python side as either
    :py:`None` or the actual value
-   Various documentation fixes (see :gh:`mosra/magnum-bindings#11`)

`2020.06`_
==========

Released 2020-06-27, tagged as
:gh:`v2020.06 <mosra/magnum-bindings/releases/tag/v2020.06>`.

-   Exposed :ref:`Matrix4.cofactor()`, :ref:`Matrix4.comatrix()`,
    :ref:`Matrix4.adjugate()` (and equivalents in other matrix sizes), and
    :ref:`Matrix4.normal_matrix()`
-   Exposed :ref:`gl.AbstractFramebuffer.blit()` functions and related enums
-   Exposed more keys in :ref:`platform.sdl2.Application` and
    :ref:`platform.glfw.Application`
-   Exposed :ref:`gl.AbstractTexture.unbind()`
-   Exposed :ref:`trade.AbstractImporter.image2d_level_count()` and related
    APIs for 1D and 3D
-   Exposed :ref:`trade.MeshData` and related APIs, the previous
    ``trade.MeshData3D`` APIs are removed
-   Exposed new APIs and tangent support in the :ref:`primitives` library
-   :ref:`platform.sdl2.Application` and :ref:`platform.glfw.Application` now
    provide a clear error instead of "pure virtual method call" in case
    ``draw_event()`` is not implemented
-   Library version is now exposed through ``MAGNUMBINDINGS_VERSION_YEAR``,
    ``MAGNUMBINDINGS_VERSION_MONTH``, ``MAGNUMBINDINGS_VERSION_COMMIT``,
    ``MAGNUMBINDINGS_VERSION_HASH`` and ``MAGNUMBINDINGS_VERSION_STRING``
    preprocessor defines in a new ``Magnum/versionBindings.h`` header. This
    header is not included by any other header to avoid trigerring a full
    rebuild when Git commit changes. If Git is not found, only the first two
    defines are present.

`2019.10`_
==========

Released 2019-10-24, tagged as
:gh:`v2019.10 <mosra/magnum-bindings/releases/tag/v2019.10>`.

Initial version. See :gh:`mosra/magnum#228`, :gh:`mosra/magnum-bindings#1`,
:gh:`mosra/magnum-bindings#2` and :gh:`mosra/magnum-bindings#6` for more
information.
