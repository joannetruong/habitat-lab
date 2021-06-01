import os
import sys

# TODO make this less brittle
sys.path = [os.path.join(os.path.dirname(__file__), '../../build/src/python/')] + sys.path

import corrade
import corrade.containers
import corrade.pluginmanager

import magnum
import magnum.gl
import magnum.meshtools
import magnum.platform
import magnum.platform.egl
import magnum.platform.glx
import magnum.platform.glfw
import magnum.platform.sdl2
import magnum.primitives
import magnum.shaders
import magnum.scenegraph
import magnum.trade

# So the doc see everything
# TODO: use just +=, m.css should reorder this on its own
corrade.__all__ = ['containers', 'pluginmanager', 'BUILD_STATIC', 'BUILD_MULTITHREADED', 'TARGET_UNIX', 'TARGET_APPLE', 'TARGET_IOS', 'TARGET_IOS_SIMULATOR', 'TARGET_WINDOWS', 'TARGET_WINDOWS_RT', 'TARGET_EMSCRIPTEN', 'TARGET_ANDROID']
magnum.__all__ = ['math', 'gl', 'meshtools', 'platform', 'primitives', 'shaders', 'scenegraph', 'trade', 'BUILD_STATIC', 'TARGET_GL', 'TARGET_GLES', 'TARGET_GLES2', 'TARGET_WEBGL', 'TARGET_VK'] + magnum.__all__

# hide values of the preprocessor defines to avoid confusion by assigning a
# class without __repr__ to them
# TODO: more systematic solution directly in m.css
class DoNotPrintValue: pass
corrade.BUILD_STATIC = DoNotPrintValue()
corrade.BUILD_MULTITHREADED = DoNotPrintValue()
corrade.TARGET_UNIX = DoNotPrintValue()
corrade.TARGET_APPLE = DoNotPrintValue()
corrade.TARGET_IOS = DoNotPrintValue()
corrade.TARGET_IOS_SIMULATOR = DoNotPrintValue()
corrade.TARGET_WINDOWS = DoNotPrintValue()
corrade.TARGET_WINDOWS_RT = DoNotPrintValue()
corrade.TARGET_EMSCRIPTEN = DoNotPrintValue()
corrade.TARGET_ANDROID = DoNotPrintValue()
magnum.BUILD_STATIC = DoNotPrintValue()
magnum.TARGET_GL = DoNotPrintValue()
magnum.TARGET_GLES = DoNotPrintValue()
magnum.TARGET_GLES2 = DoNotPrintValue()
magnum.TARGET_WEBGL = DoNotPrintValue()
magnum.TARGET_VK = DoNotPrintValue()

# TODO ugh... can this be expressed directly in pybind?
corrade.__annotations__ = {
    'BUILD_STATIC': bool,
    'BUILD_MULTITHREADED': bool,
    'TARGET_UNIX': bool,
    'TARGET_APPLE': bool,
    'TARGET_IOS': bool,
    'TARGET_IOS_SIMULATOR': bool,
    'TARGET_WINDOWS': bool,
    'TARGET_WINDOWS_RT': bool,
    'TARGET_EMSCRIPTEN': bool,
    'TARGET_ANDROID': bool
}
magnum.__annotations__ = {
    'BUILD_STATIC': bool,
    'TARGET_GL': bool,
    'TARGET_GLES': bool,
    'TARGET_GLES2': bool,
    'TARGET_WEBGL': bool,
    'TARGET_VK': bool
}
magnum.gl.__annotations__ = {
    'default_framebuffer': magnum.gl.DefaultFramebuffer
}
magnum.shaders.Flat2D.__annotations__ = {
    'POSITION': magnum.gl.Attribute,
    'TEXTURE_COORDINATES': magnum.gl.Attribute,
    'COLOR3': magnum.gl.Attribute,
    'COLOR4': magnum.gl.Attribute
}
magnum.shaders.Flat3D.__annotations__ = {
    'POSITION': magnum.gl.Attribute,
    'TEXTURE_COORDINATES': magnum.gl.Attribute,
    'COLOR3': magnum.gl.Attribute,
    'COLOR4': magnum.gl.Attribute
}
magnum.shaders.VertexColor2D.__annotations__ = {
    'POSITION': magnum.gl.Attribute,
    'COLOR3': magnum.gl.Attribute,
    'COLOR4': magnum.gl.Attribute
}
magnum.shaders.VertexColor3D.__annotations__ = {
    'POSITION': magnum.gl.Attribute,
    'COLOR3': magnum.gl.Attribute,
    'COLOR4': magnum.gl.Attribute
}
magnum.shaders.Phong.__annotations__ = {
    'POSITION': magnum.gl.Attribute,
    'NORMAL': magnum.gl.Attribute,
    'TANGENT': magnum.gl.Attribute,
    'TEXTURE_COORDINATES': magnum.gl.Attribute,
    'COLOR3': magnum.gl.Attribute,
    'COLOR4': magnum.gl.Attribute
}

PROJECT_TITLE = 'Magnum'
PROJECT_SUBTITLE = 'Python docs'
MAIN_PROJECT_URL = 'https://magnum.graphics'
INPUT_MODULES = [corrade, magnum]
INPUT_PAGES = [
    'pages/index.rst',
    'pages/building.rst',
    'pages/api-conventions.rst',
    'pages/changelog.rst',
    'pages/credits.rst',

    '../../../magnum-examples/doc/python/examples.rst'
]
INPUT_DOCS = [
    'corrade.rst',
    'corrade.containers.rst',
    'corrade.pluginmanager.rst',

    'magnum.rst',
    'magnum.gl.rst',
    'magnum.math.rst',
    'magnum.platform.rst',
    'magnum.scenegraph.rst',
    'magnum.shaders.rst',
    'magnum.trade.rst',
]

LINKS_NAVBAR2 = [
    ('C++ API', '../../../../magnum/build/doc-mcss/html/index.html', [])
]

PLUGINS = [
    'm.code',
    'm.components',
    'm.dox',
    'm.gh',
    'm.htmlsanity',
    'm.images',
    'm.link',
    'm.math',
    'm.sphinx'
]

STYLESHEETS = [
    'https://fonts.googleapis.com/css?family=Source+Sans+Pro:400,400i,600,600i%7CSource+Code+Pro:400,400i,600&subset=latin-ext',
    '../css/m-dark+documentation.compiled.css'
]

FAVICON = '../favicon.ico'

M_DOX_TAGFILES = [
    ('../../../corrade/build/doc-mcss/corrade.tag', '../../../../corrade/build/doc-mcss/html/', ['Corrade::'],  ['m-doc-external']),
    ('../../../magnum/build/doc-mcss/magnum.tag', '../../../../magnum/build/doc-mcss/html/', ['Magnum::'],  ['m-doc-external'])
]
M_SPHINX_INVENTORY_OUTPUT = 'objects.inv'
M_SPHINX_INVENTORIES = [
    ('python.inv', 'https://docs.python.org/3/', [], ['m-doc-external']),
    ('numpy.inv', 'https://docs.scipy.org/doc/numpy/', [], ["m-doc-external"])
]
M_HTMLSANITY_SMART_QUOTES = True
M_MATH_CACHE = 'm.math.cache'

PYBIND11_COMPATIBILITY = True

OUTPUT = '../../build/doc/python/'

PAGE_HEADER = """
.. container:: m-note m-success

    Welcome to Python-flavored Magnum! Please note that, while already being
    rather stable, this functionality is still considered *experimental* and
    some APIs might get changed without preserving full backwards compatibility.
"""

FINE_PRINT = """
| Magnum Python docs. Part of the `Magnum project <https://magnum.graphics/>`_,
  copyright © `Vladimír Vondruš <http://mosra.cz/>`_ and contributors, 2010–2021.
| Generated by `m.css Python doc generator <https://mcss.mosra.cz/documentation/python/>`_.
  Contact the team via `GitHub <https://github.com/mosra/magnum>`_,
  `Gitter <https://gitter.im/mosra/magnum>`_,
  `e-mail <mailto:info@magnum.graphics>`_ or
  `Twitter <https://twitter.com/czmosra>`_"""
