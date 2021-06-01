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

Python API conventions
######################

:summary: Basic rules and good practices for both Python binding developers and
    users.
:ref-prefix:
    corrade
    magnum

`API naming`_
=============

-   mapping is made as clear as possible, the user should not need assistance
    to know what's the corresponding name in Python
-   modules ``lowercase``, words *not* separated with underscores (which means
    C++ namespaces have to be named clearly and tersely to make this possible)
-   class names ``CamelCase``
-   function names ``snake_case``
-   constants and enums ``UPPERCASE``, again underscores omitted if it doesn't
    hurt readability

`Preprocessor definitions`_
---------------------------

Exposed to Python as plain boolean constants, and only those that actually are
useful in a Python setting.

.. class:: m-table

=================================== ==================================
C++                                 Python
=================================== ==================================
:dox:`CORRADE_BUILD_MULTITHREADED`  :ref:`corrade.BUILD_MULTITHREADED`
:dox:`MAGNUM_TARGET_GLES`           :ref:`magnum.TARGET_GLES`
=================================== ==================================

`Namespaces / modules`_
-----------------------

.. class:: m-table

=================================== ============================
C++                                 Python
=================================== ============================
:dox:`Magnum::Math`                 :ref:`magnum.math`
:dox:`Magnum::SceneGraph`           :ref:`magnum.scenegraph`
=================================== ============================

`Classes`_
----------

.. class:: m-table

=================================== ============================
C++                                 Python
=================================== ============================
:dox:`Vector2i`                     :ref:`Vector2i`
:dox:`GL::Buffer`                   :ref:`gl.Buffer`
=================================== ============================

`Functions`_
------------

.. class:: m-table

=============================================================== ===========
C++                                                             Python
=============================================================== ===========
:dox:`Math::angle()`                                            :ref:`math.angle()`
:dox:`Vector2::xAxis() <Math::Vector2::xAxis()>`                :ref:`Vector2.x_axis()`
:dox:`v.isZero() <Math::Vector::isZero()>`                      :ref:`v.is_zero() <Vector3.is_zero()>`
:dox:`m.transformVector(a) <Math::Matrix4::transformVector()>`  :ref:`m.transform_vector(a) <Matrix4.transform_vector()>`
=============================================================== ===========

`Enums`_
--------

.. class:: m-table

============================================== ============================
C++                                            Python
============================================== ============================
:dox:`PixelFormat::RGB8Unorm`                  :ref:`PixelFormat.RGB8_UNORM`
:dox:`MeshPrimitive::TriangleStrip`            :ref:`MeshPrimitive.TRIANGLE_STRIP`
============================================== ============================

`Constants`_
------------

Apart from :dox:`Math::Constants`, which are exposed directly as members of the
:ref:`magnum.math` submodule to mimic Python's :ref:`math`, most of the
constants used throughout the C++ API are related to templates. Those are,
where applicable, converted to Python builtins such as :py:`len()`.

.. class:: m-table

============================================== ============================
C++                                            Python
============================================== ============================
:dox:`Constants::pi() <Math::Constants::pi()>` :ref:`math.pi <magnum.math.pi>`
:dox:`Math::Vector::Size`                      :py:`len(vec)`
============================================== ============================

`Initialization tags`_
----------------------

Since overloading based on argument types is not a common thing to do in Python
(and it adds extra overhead in pybind11), all initialization tags are converted
to static constructors instead:

.. container:: m-row

    .. container:: m-col-m-6

        .. code-figure::

            .. code:: c++

                Matrix4 a{Math::IdentityInit, 5.0f};
                GL::Buffer b{NoCreate};

            C++

    .. container:: m-col-m-6

        .. code-figure::

            .. code:: py

                a = Matrix4.identity_init(5.0)
                b = gl.Buffer.no_create()

            Python

There's no equivalent for the :dox:`Math::NoInit <Math::NoInitT>` tag, as
such optimization doesn't make much sense when instances are copied back
and forth between C++ and Python. Similarly, the :dox:`NoCreate <NoCreateT>`
tag makes sense only in C++ which differentiates between stack-allocated and
heap-allocated instances. In Python it's enough to simply set an instance to
:py:`None` to achieve the same effect.

`Name import conventions`_
==========================

Similarly to C++, where it's encouraged to do something like

.. code:: c++

    namespace YourProject {
        using namespace Magnum;
    }

and then use Magnum C++ APIs unprefixed from inside that namespace, the
recommended Python workflow is similar. Note that importing the root module
*does not* import submodules, so you are expected to import those on an
as-needed basis as well.

.. code:: py

    from magnum import *
    from magnum import gl, platform

In particular, both the C++ and the Python API is designed in a way to prevent
too generic or confusing names in the root namespace / module and also keeping
it relatively clean and small, without too many symbols. On the other hand, the
subnamespaces *do* have generic names. The :dox:`GL::version()` /
:ref:`gl.version()` API is one example --- it's tucked in a subnamespace so the
generic name isn't a problem, but you wouldn't find anything of similar
genericity in the root namespace / module.

An exception to this rule is exposed preprocessor definitions --- these are
*not* pulled in when doing :py:`from magnum import *` as this would likely
cause conflicts (in particular, :ref:`BUILD_STATIC` is defined by Corrade as
well). Instead, you have to access them like this:

.. code:: py

    import magnum

    if magnum.TARGET_GLES2:
        format = gl.TextureFormat.RGBA8
    else:
        format = gl.TextureFormat.R8

`Handling of alternate implementations`_
----------------------------------------

C++ APIs that have alternative implementations (such as
:dox:`Platform::Sdl2Application` vs. :dox:`Platform::GlfwApplication`, or
:dox:`SceneGraph::MatrixTransformation3D` vs.
:dox:`SceneGraph::TranslationRotationScalingTransformation3D`) either provide
:cpp:`typedef`\ s based on what header you include or require you to
:cpp:`typedef` them yourselves:

.. code:: c++

    class MyApplication: Platform::Application {}; // depends on what you include

    typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;

In Python, the alternate implementations are tucked in submodules (such as
:ref:`platform.sdl2` vs. :ref:`platform.glfw`, or :ref:`scenegraph.matrix` vs.
:ref:`scenegraph.trs`), each submodule providing the same names (such as
:ref:`Application <platform.sdl2.Application>` or
:ref:`Object3D <scenegraph.matrix.Object3D>`)
and the designed way to use them is via :py:`from ... import`:

.. code:: py

    from magnum.platform.sdl2 import Application
    from magnum.scenegraph.trs import Scene3D, Object3D

`Basic guarantees`_
===================

-   All types printable using :dox:`Utility::Debug` implement :py:`__repr__()`
    on the Python side, producing the exact same output.
