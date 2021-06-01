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
import shutil

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

extension_paths = {
    # Filled in by cmake. This works for both static and dynamic builds -- in
    # case a library is built statically, only the underscored name will be
    # present.
    '_corrade': '$<TARGET_FILE:corrade>',
    'corrade.containers': '${corrade_containers_file}',
    'corrade.pluginmanager': '${corrade_pluginmanager_file}',
    '_magnum': '$<TARGET_FILE:magnum>',
    'magnum.gl': '${magnum_gl_file}',
    'magnum.meshtools': '${magnum_meshtools_file}',
    'magnum.primitives': '${magnum_primitives_file}',
    'magnum.scenegraph': '${magnum_scenegraph_file}',
    'magnum.shaders': '${magnum_shaders_file}',
    'magnum.platform.egl': '${magnum_platform_egl_file}',
    'magnum.platform.glx': '${magnum_platform_glx_file}',
    'magnum.platform.glfw': '${magnum_platform_glfw_file}',
    'magnum.platform.sdl2': '${magnum_platform_sdl2_file}',
    'magnum.trade': '${magnum_trade_file}',
}

packages = ['corrade', 'magnum']

# On dynamic builds, platform is a package with an __init__.py and submodules.
# On static builds, it's a submodule of magnum with everything present
# statically.
if '${MAGNUM_BUILD_STATIC}' != 'ON':
    packages += ['magnum.platform']

class TheExtensionIsAlreadyBuiltWhyThisHasToBeSoDamnComplicated(build_ext):
    def run(self):
        for ext in self.extensions:
            shutil.copyfile(extension_paths[ext.name], self.get_ext_fullpath(ext.name))

setup(
    name='magnum',
    packages=packages,
    ext_modules=[Extension(name, sources=[]) for name, path in extension_paths.items() if path],
    package_dir={
        # Explicitly supply package_dir so importing this file from another
        # setup.py (i.e., when this is a bundled dependency of another project)
        # does the right thing. Can't use '': because that doesn't work when
        # executing setup.py directly, can't use '.': because that doesn't work
        # when executing setup.py from another setup, so need to list all
        # packages explicitly.
        i: os.path.join(os.path.dirname(__file__), i.replace('.', '/')) for i in packages
    },
    cmdclass={
        'build_ext': TheExtensionIsAlreadyBuiltWhyThisHasToBeSoDamnComplicated
    },
    zip_safe=True
)

# kate: hl python
