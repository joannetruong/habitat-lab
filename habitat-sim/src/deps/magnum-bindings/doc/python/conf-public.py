# Inherit everything from the local config
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from conf import *

assert LINKS_NAVBAR2[0][0] == 'C++ API'
LINKS_NAVBAR2[0] = (LINKS_NAVBAR2[0][0], 'https://magnum.graphics/doc/magnum/', [])

OUTPUT = '../../build/doc-public/python/'

assert len(M_DOX_TAGFILES) == 2
M_DOX_TAGFILES = [
    # TODO: the path should be relative to this file
    (os.path.join(os.path.dirname(__file__), '../../../corrade/build/doc-mcss/corrade.tag'), 'https://doc.magnum.graphics/corrade/', ['Corrade::'],  ['m-doc-external']),
    (os.path.join(os.path.dirname(__file__), '../../../magnum/build/doc-mcss/magnum.tag'), 'https://doc.magnum.graphics/magnum/', ['Magnum::'],  ['m-doc-external'])
]

STYLESHEETS = [
    'https://fonts.googleapis.com/css?family=Source+Sans+Pro:400,400i,600,600i%7CSource+Code+Pro:400,400i,600&subset=latin-ext',
    'https://static.magnum.graphics/m-dark.compiled.css',
    'https://static.magnum.graphics/m-dark.documentation.compiled.css'
]

FAVICON = 'https://doc.magnum.graphics/favicon.ico'

SEARCH_DOWNLOAD_BINARY = 'searchdata-v1.bin'
SEARCH_BASE_URL = 'https://doc.magnum.graphics/python/'
SEARCH_EXTERNAL_URL = 'https://google.com/search?q=site:doc.magnum.graphics+Magnum+Python+{query}'

def URL_FORMATTER(type, path):
    # Put static files into the root, everything else into subdirs
    if type.name == 'STATIC':
        prefix = os.path.basename(path[0])
        return prefix, '/python/' + prefix

    # And special casing for index, of course
    if type.name == 'PAGE' and len(path) == 1 and path[0] == 'index':
        return 'index.html', '/python/'

    prefix = '/'.join(path) + '/'
    return prefix + 'index.html', '/python/' + prefix
