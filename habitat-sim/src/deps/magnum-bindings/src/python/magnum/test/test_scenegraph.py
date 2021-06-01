#
#   This file is part of Magnum.
#
#   Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
#               2020, 2021 Vladimír Vondruš <mosra@centrum.cz>
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#

import sys
import unittest

from magnum import *
from magnum import scenegraph
from magnum.scenegraph.matrix import Object3D, Scene3D

class Object(unittest.TestCase):
    def test_hierarchy(self):
        scene = Scene3D()
        scene_refcount = sys.getrefcount(scene)

        a = Object3D()
        a_refcount = sys.getrefcount(a)
        self.assertIs(a.scene, None)

        b = Object3D(parent=scene)
        b_refcount = sys.getrefcount(b)
        self.assertIs(b.scene, scene)
        self.assertIs(b.parent, scene)

        # B should be referenced by the scene, but not cyclically
        self.assertEqual(sys.getrefcount(b), scene_refcount + 1)
        self.assertEqual(sys.getrefcount(scene), scene_refcount)

        c = Object3D(parent=b)
        c_refcount = sys.getrefcount(c)
        self.assertIs(c.scene, scene)
        self.assertIs(c.parent, b)

        # C should be referenced by B
        self.assertEqual(sys.getrefcount(b), scene_refcount + 1)
        self.assertEqual(sys.getrefcount(c), scene_refcount + 1)
        self.assertEqual(sys.getrefcount(scene), scene_refcount)

        # Delete B. Because B has a parent as well, it's not deleted yet
        del b
        self.assertIsNotNone(c.parent)
        self.assertEqual(sys.getrefcount(c.parent), b_refcount - 1)
        self.assertEqual(sys.getrefcount(c), scene_refcount + 1)

        # Delete a scene. That also makes B deleted and C is then orphaned
        del scene
        self.assertIsNone(c.parent)
        self.assertEqual(sys.getrefcount(c), c_refcount - 1)

    def test_hierarchy_set_parent(self):
        # Same as test_hierarchy, but setting the parent later

        scene = Scene3D()
        scene_refcount = sys.getrefcount(scene)

        a = Object3D()
        a_refcount = sys.getrefcount(a)
        self.assertIs(a.scene, None)

        b = Object3D()
        b.parent = scene
        b_refcount = sys.getrefcount(b)
        self.assertIs(b.scene, scene)
        self.assertIs(b.parent, scene)

        # B should be referenced by the scene, but not cyclically
        self.assertEqual(sys.getrefcount(b), scene_refcount + 1)
        self.assertEqual(sys.getrefcount(scene), scene_refcount)

        c = Object3D()
        c.parent = b
        c_refcount = sys.getrefcount(c)
        self.assertIs(c.scene, scene)
        self.assertIs(c.parent, b)

        # C should be referenced by B
        self.assertEqual(sys.getrefcount(b), scene_refcount + 1)
        self.assertEqual(sys.getrefcount(c), scene_refcount + 1)
        self.assertEqual(sys.getrefcount(scene), scene_refcount)

        # Delete B. Because B has a parent as well, it's not deleted yet
        del b
        self.assertIsNotNone(c.parent)
        self.assertEqual(sys.getrefcount(c.parent), b_refcount - 1)
        self.assertEqual(sys.getrefcount(c), scene_refcount + 1)

        # Delete a scene. That also makes B deleted and C is then orphaned
        del scene
        self.assertIsNone(c.parent)
        self.assertEqual(sys.getrefcount(c), c_refcount - 1)

    def test_set_parent_invalid(self):
        a = Object3D()
        with self.assertRaisesRegex(TypeError, "expected Scene, Object or None, got <class 'str'>"):
            a.parent = "noo"

    def test_drawable(self):
        object = Object3D()
        object_refcount = sys.getrefcount(object)

        a = scenegraph.Drawable3D(object)
        a_refcount = sys.getrefcount(a)
        self.assertIs(a.object, object)

        b = scenegraph.Drawable3D(object)
        b_refcount = sys.getrefcount(b)
        self.assertIs(b.object, object)

        # Drawables should be referenced by the object, but not cyclically
        self.assertEqual(sys.getrefcount(object), object_refcount)
        self.assertEqual(sys.getrefcount(a), object_refcount + 1)
        self.assertEqual(sys.getrefcount(b), object_refcount + 1)

        # Delete the object. The drawable should be still alive, but
        # disconnected from the object (and thus useless).
        del object
        self.assertIsNone(a.object)
        self.assertIsNone(b.object)
        self.assertEqual(sys.getrefcount(a), a_refcount - 1)
        self.assertEqual(sys.getrefcount(b), b_refcount - 1)

    def test_drawable_group(self):
        object = Object3D()
        drawables = scenegraph.DrawableGroup3D()

        deleted = 0
        class MyDrawable(scenegraph.Drawable3D):
            def __del__(self):
                nonlocal deleted
                deleted += 1

        a = MyDrawable(object, drawables)
        b = MyDrawable(object, drawables)

        # The drawable group should have these listed
        self.assertEqual([i for i in drawables], [a, b])

        # Deleting each of them should do nothing, since they're still
        # referenced by the object
        del a, b
        self.assertEqual(deleted, 0)
        self.assertEqual(len(drawables), 2)

        # Deleting the holder object will, tho
        del object
        self.assertEqual(deleted, 2)
        self.assertEqual(len(drawables), 0)

    def test_camera(self):
        object = Object3D()
        object.translate(Vector3.z_axis(5.0))
        object_refcount = sys.getrefcount(object)

        a = scenegraph.Camera3D(object)
        a.viewport = (400, 300)
        a.projection_matrix = Matrix4.perspective_projection(
            fov=Deg(45.0), near=0.01, far=100.0, aspect_ratio=1.0)
        a.aspect_ratio_policy = scenegraph.AspectRatioPolicy.EXTEND
        a_refcount = sys.getrefcount(a)
        self.assertEqual(a.viewport, Vector2i(400, 300))
        self.assertEqual(a.projection_matrix, Matrix4.perspective_projection(
            fov=Deg(57.82240), near=0.01, far=100.0, aspect_ratio=1.33333333))
        self.assertEqual(a.camera_matrix, Matrix4.translation(-Vector3.z_axis(5.0)))
        self.assertEqual(a.aspect_ratio_policy, scenegraph.AspectRatioPolicy.EXTEND)
        self.assertIs(a.object, object)

        # Camera should be referenced by the object, but not cyclically
        self.assertEqual(sys.getrefcount(object), object_refcount)
        self.assertEqual(sys.getrefcount(a), object_refcount + 1)

        # Delete the object. The camera should be still alive, but disconnected
        # from the object (and thus useless).
        del object
        self.assertIsNone(a.object)
        self.assertEqual(sys.getrefcount(a), a_refcount - 1)

    def test_camera_draw(self):
        scene = Scene3D()
        drawables = scenegraph.DrawableGroup3D()

        camera_object = Object3D(scene)
        camera_object.translate((0.0, 1.0, 5.0))

        camera = scenegraph.Camera3D(camera_object)

        rendered = None, None
        deleted = "no :)"
        class MyDrawable(scenegraph.Drawable3D):
            def draw(self, transformation_matrix: Matrix4, camera: scenegraph.Camera3D):
                nonlocal rendered
                rendered = (transformation_matrix, camera)

            def __del__(self):
                nonlocal deleted
                deleted = "yes :("

        class MySilentDrawable(scenegraph.Drawable3D):
            def draw(self, transformation_matrix: Matrix4, camera: scenegraph.Camera3D):
                pass

        object = Object3D(scene)
        object.translate(Vector3.x_axis(5.0))
        a = MyDrawable(object, drawables)
        b = MySilentDrawable(object, drawables)

        # The drawable group should have these listed
        self.assertEqual([i for i in drawables], [a, b])

        # Deleting the object, the camera holder and drawable does nothing
        del camera_object, object, a, b

        camera.draw(drawables)
        self.assertEqual(rendered[0],
            Matrix4.translation(Vector3.x_axis(5.0))@
            Matrix4.translation((0.0, -1.0, -5.0)))
        self.assertIs(rendered[1], camera)

        # Deleting the scene will delete A and the drawable as well
        del scene
        self.assertEqual(deleted, "yes :(")
        self.assertIsNone(camera.object)
        self.assertIs(len(drawables), 0)

class Feature(unittest.TestCase):
    def test(self):
        class MyFeature(scenegraph.AbstractFeature3D):
            def __init__(self, object: Object3D):
                scenegraph.AbstractFeature3D.__init__(self, object)

        object = Object3D()
        feature = MyFeature(object)
        self.assertIs(feature.object, object)
