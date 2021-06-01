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

.. py:module:: magnum.scenegraph

    The Python API for :dox:`SceneGraph` provides, similarly to C++, multiple
    different transformation implementations. Recommended usage is importing
    desired implementation akin to :cpp:`typedef`\ ing the types in C++:

    .. code-figure::

        .. code:: c++

            #include <Magnum/SceneGraph/Object.h>
            #include <Magnum/SceneGraph/Scene.h>
            #include <Magnum/SceneGraph/MatrixTransformation3D.h>

            typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;
            typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;

        C++

    .. code-figure::

        .. code:: py

            from magnum import scenegraph
            from magnum.scenegraph.matrix import Scene3D, Object3D

        Python

    `Scene vs Object`_
    ==================

    In C++, the Scene is a subclass of Object. However, because the Scene
    object is not transformable nor it's possible to attach features to it,
    most of the inherited API is unusable. This could be considered a wart of
    the C++ API, so the Python bindings expose Scene and Object as two
    unrelated types and all APIs that can take either a Scene or an Object
    have corresponding overloads.

    `Reference counting`_
    =====================

    Compared to C++, the following is done with all Object instances created
    on Python side:

    -   the object is additionally referenced by its parent (if there's any)
        so objects created in local scope stay alive even after exiting the
        scope
    -   deleting its parent (either due to it going out of scope or using
        :py:`del` in Python) will cause it to have no parent instead of being
        cascade deleted (unless it's not referenced anymore, in which case it's deleted as well)
    -   in order to actually destroy an object, it has to have no parent

    For features it's slightly different:

    -   the feature is additionally referenced by the holder object so features
        created in local scope stay alive even after exiting the scope
    -   deleting the holder object (either due to it going out of scope
        or using :py:`del` in Python) will cause it to be without a holder
        object (unless it's not referenced anymore, in which case it's deleted
        as well) --- this makes any further operations on it impossible and
        likely dangerous
    -   in order to actually destroy a feature, it has to have no holder object
