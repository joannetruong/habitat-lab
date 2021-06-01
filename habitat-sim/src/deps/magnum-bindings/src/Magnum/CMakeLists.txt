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

# Generate version header. If Git is found and this is a Git working copy,
# extract values from there, otherwise use just MAGNUMBINDINGS_VERSION_YEAR /
# MONTH that are set in project root CMakeLists.
find_package(Git)
if(Git_FOUND)
    # Match only tags starting with `v`, always use the long format so we have
    # a commit hash also on a tagged version
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --match "v*" --long
        OUTPUT_VARIABLE MAGNUMBINDINGS_VERSION_STRING
        RESULT_VARIABLE _MAGNUMBINDINGS_VERSION_RESULT
        # Otherwise this gets executed in the build dir, which might be inside
        # a totally different Git working copy
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(MAGNUMBINDINGS_VERSION_STRING MATCHES "^v([0-9][0-9][0-9][0-9])\\.0?([0-9][0-9]?)-([0-9]+)-g([a-f0-9]+)$")
        set(MAGNUMBINDINGS_VERSION_YEAR ${CMAKE_MATCH_1})
        set(MAGNUMBINDINGS_VERSION_MONTH ${CMAKE_MATCH_2})
        set(MAGNUMBINDINGS_VERSION_COMMIT ${CMAKE_MATCH_3})
        # Otherwise if commit is 0, it would #undef the variable
        set(MAGNUMBINDINGS_VERSION_COMMIT " ${MAGNUMBINDINGS_VERSION_COMMIT}")
        set(MAGNUMBINDINGS_VERSION_HASH ${CMAKE_MATCH_4})
    elseif(_MAGNUMBINDINGS_VERSION_RESULT EQUAL 0)
        message(WARNING "Can't match Git version from ${MAGNUMBINDINGS_VERSION_STRING}")
    endif()
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/versionBindings.h.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/versionBindings.h)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/versionBindings.h DESTINATION ${MAGNUM_INCLUDE_INSTALL_DIR})

if(WITH_PYTHON)
    add_custom_target(MagnumPython SOURCES PythonBindings.h)
    set_target_properties(MagnumPython PROPERTIES FOLDER "Magnum/Python")
    install(FILES PythonBindings.h DESTINATION ${MAGNUM_INCLUDE_INSTALL_DIR})
endif()

find_package(Magnum COMPONENTS GL SceneGraph)

if(Magnum_GL_FOUND)
    add_subdirectory(GL)
endif()

if(Magnum_SceneGraph_FOUND)
    add_subdirectory(SceneGraph)
endif()

if(BUILD_TESTS)
    add_subdirectory(Test)
endif()
