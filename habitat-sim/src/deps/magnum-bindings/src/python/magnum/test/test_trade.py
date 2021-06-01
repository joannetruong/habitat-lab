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

import os
import sys
import unittest

from corrade import pluginmanager
from magnum import *
from magnum import trade

class ImageData(unittest.TestCase):
    def test(self):
        # The only way to get an image instance is through a manager
        importer = trade.ImporterManager().load_and_instantiate('StbImageImporter')
        importer.open_file(os.path.join(os.path.dirname(__file__), "rgb.png"))
        image = importer.image2d(0)
        self.assertFalse(image.is_compressed)
        self.assertEqual(image.storage.alignment, 1) # libPNG has 4 tho
        self.assertEqual(image.format, PixelFormat.RGB8_UNORM)
        self.assertEqual(image.pixel_size, 3)
        self.assertEqual(image.size, Vector2i(3, 2))
        # TODO: ugh, report as bytes, not chars
        self.assertEqual(ord(image.pixels[1, 2, 2]), 181)
        self.assertEqual(ord(image.data[9 + 6 + 2]), 181) # libPNG has 12 +

    def test_compressed(self):
        # The only way to get an image instance is through a manager
        importer = trade.ImporterManager().load_and_instantiate('DdsImporter')
        importer.open_file(os.path.join(os.path.dirname(__file__), "rgba_dxt1.dds"))
        image = importer.image2d(0)
        self.assertEqual(len(image.data), 8)
        self.assertTrue(image.is_compressed)
        # TODO: compressed properties

        # No compressed-image-related APIs exposed ATM, so just verifying the
        # uncompressed ones fail properly
        with self.assertRaisesRegex(AttributeError, "image is compressed"):
            image.storage
        with self.assertRaisesRegex(AttributeError, "image is compressed"):
            image.format
        with self.assertRaisesRegex(AttributeError, "image is compressed"):
            image.pixel_size
        with self.assertRaisesRegex(AttributeError, "image is compressed"):
            image.pixels

    def test_convert_view(self):
        # The only way to get an image instance is through a manager
        importer = trade.ImporterManager().load_and_instantiate('StbImageImporter')
        importer.open_file(os.path.join(os.path.dirname(__file__), "rgb.png"))
        image = importer.image2d(0)

        view = ImageView2D(image)
        mutable_view = MutableImageView2D(image)

    def test_convert_view_compressed(self):
        # The only way to get an image instance is through a manager
        importer = trade.ImporterManager().load_and_instantiate('DdsImporter')
        importer.open_file(os.path.join(os.path.dirname(__file__), "rgba_dxt1.dds"))
        image = importer.image2d(0)

        # No compressed-image-related APIs exposed ATM, so just verifying the
        # uncompressed ones fail properly
        with self.assertRaisesRegex(RuntimeError, "image is compressed"):
            view = ImageView2D(image)
        with self.assertRaisesRegex(RuntimeError, "image is compressed"):
            mutable_view = MutableImageView2D(image)

class MeshData(unittest.TestCase):
    def test(self):
        # The only way to get a mesh instance is through a manager
        importer = trade.ImporterManager().load_and_instantiate('TinyGltfImporter')
        importer.open_file(os.path.join(os.path.dirname(__file__), 'mesh.glb'))

        mesh = importer.mesh(0)
        self.assertEqual(mesh.primitive, MeshPrimitive.TRIANGLES)
        # TODO: test more, once it's exposed

class Importer(unittest.TestCase):
    def test(self):
        manager = trade.ImporterManager()
        self.assertIn('StbImageImporter', manager.alias_list)
        self.assertEqual(manager.load_state('StbImageImporter'), pluginmanager.LoadState.NOT_LOADED)

        self.assertTrue(manager.load('StbImageImporter') & pluginmanager.LoadState.LOADED)
        self.assertEqual(manager.unload('StbImageImporter'), pluginmanager.LoadState.NOT_LOADED)

        with self.assertRaisesRegex(RuntimeError, "can't load plugin"):
            manager.load('NonexistentImporter')
        with self.assertRaisesRegex(RuntimeError, "can't unload plugin"):
            manager.unload('NonexistentImporter')

    def test_no_file_opened(self):
        importer = trade.ImporterManager().load_and_instantiate('StbImageImporter')
        self.assertFalse(importer.is_opened)

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.mesh_count
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.mesh_level_count(0)

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.mesh_for_name('')

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.mesh_name(0)

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.mesh(0)

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image1d_count
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image2d_count
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image3d_count

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image1d_level_count(0)
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image2d_level_count(0)
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image3d_level_count(0)

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image1d_for_name('')
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image2d_for_name('')
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image3d_for_name('')

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image1d_name(0)
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image2d_name(0)
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image3d_name(0)

        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image1d(0)
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image2d(0)
        with self.assertRaisesRegex(RuntimeError, "no file opened"):
            importer.image3d(0)

    def test_index_oob(self):
        importer = trade.ImporterManager().load_and_instantiate('StbImageImporter')
        importer.open_file(os.path.join(os.path.dirname(__file__), 'rgb.png'))

        with self.assertRaises(IndexError):
            importer.mesh_level_count(0)
        with self.assertRaises(IndexError):
            importer.mesh_name(0)
        with self.assertRaises(IndexError):
            importer.mesh(0)

        with self.assertRaises(IndexError):
            importer.image1d_level_count(0)
        with self.assertRaises(IndexError):
            importer.image2d_level_count(1)
        with self.assertRaises(IndexError):
            importer.image3d_level_count(0)

        with self.assertRaises(IndexError):
            importer.image1d_name(0)
        with self.assertRaises(IndexError):
            importer.image2d_name(1)
        with self.assertRaises(IndexError):
            importer.image3d_name(0)

        with self.assertRaises(IndexError):
            importer.image1d(0)
        with self.assertRaises(IndexError):
            importer.image2d(0, 1)
        with self.assertRaises(IndexError):
            importer.image2d(1)
        with self.assertRaises(IndexError):
            importer.image3d(0)

    def test_open_failed(self):
        importer = trade.ImporterManager().load_and_instantiate('StbImageImporter')

        with self.assertRaisesRegex(RuntimeError, "opening nonexistent.png failed"):
            importer.open_file('nonexistent.png')
        with self.assertRaisesRegex(RuntimeError, "opening data failed"):
            importer.open_data(b'')

    def test_mesh(self):
        # importer refcounting tested in image2d
        importer = trade.ImporterManager().load_and_instantiate('TinyGltfImporter')
        importer.open_file(os.path.join(os.path.dirname(__file__), 'mesh.glb'))
        self.assertEqual(importer.mesh_count, 3)
        self.assertEqual(importer.mesh_level_count(0), 1)
        self.assertEqual(importer.mesh_name(0), 'Non-indexed mesh')
        self.assertEqual(importer.mesh_for_name('Non-indexed mesh'), 0)

        mesh = importer.mesh(0)
        self.assertEqual(mesh.primitive, MeshPrimitive.TRIANGLES)

    def test_mesh_index_oob(self):
        # importer refcounting tested in image2d
        importer = trade.ImporterManager().load_and_instantiate('TinyGltfImporter')
        importer.open_file(os.path.join(os.path.dirname(__file__), 'mesh.glb'))

        with self.assertRaises(IndexError):
            importer.mesh(0, 1)

    def test_image2d(self):
        manager = trade.ImporterManager()
        manager_refcount = sys.getrefcount(manager)

        # Importer references the manager to ensure it doesn't get GC'd before
        # the plugin instances
        importer = manager.load_and_instantiate('StbImageImporter')
        self.assertIs(importer.manager, manager)
        self.assertEqual(sys.getrefcount(manager), manager_refcount + 1)

        importer.open_file(os.path.join(os.path.dirname(__file__), 'rgb.png'))
        self.assertEqual(importer.image2d_count, 1)
        self.assertEqual(importer.image2d_level_count(0), 1)
        self.assertEqual(importer.image2d_name(0), '')
        self.assertEqual(importer.image2d_for_name(''), -1)

        image = importer.image2d(0)
        self.assertEqual(image.size, Vector2i(3, 2))

        # Deleting the importer should decrease manager refcount again
        del importer
        self.assertEqual(sys.getrefcount(manager), manager_refcount)

    def test_image2d_data(self):
        importer = trade.ImporterManager().load_and_instantiate('StbImageImporter')

        with open(os.path.join(os.path.dirname(__file__), "rgb.png"), 'rb') as f:
            importer.open_data(f.read())

        image = importer.image2d(0)
        self.assertEqual(image.size, Vector2i(3, 2))

    def test_image2d_failed(self):
        importer = trade.ImporterManager().load_and_instantiate('StbImageImporter')
        importer.open_data(b'bla')

        with self.assertRaisesRegex(RuntimeError, "import failed"):
            image = importer.image2d(0)
