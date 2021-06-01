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

# Thanks, CMake, for making the recommended way of installing pacakges so
# useless that it's impossible to call find_package() consistently independent
# of whether it's installed system-wide or added through add_subdirectory().
# This also means we can't use pybind11_VERSION below but have to extract it
# from some private cache variables instead. THIS IS ABSOLUTELY AWFUL, FFS.
if(NOT COMMAND pybind11_add_module)
    find_package(pybind11 CONFIG REQUIRED)
elseif(NOT pybind11_VERSION)
    set(pybind11_VERSION ${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH})
endif()

# In pybind11 2.2.4 and below, pybind11_add_module() added the include
# directories as non-system. That, combined with Corrade's warning level, added
# an insane amount of warnings to the build. Since 2.3 it was possible to
# override that by passing SYSTEM to pybind11_add_module(), HOWEVER since 2.6
# doing so causes an ANNOYING warning because they made that a default. That
# all in a span of barely two years. Can't things just stay stable for a little
# moment?!
if(pybind11_VERSION VERSION_LESS 2.6)
    set(pybind11_add_module_SYSTEM SYSTEM)
endif()

# UGH FFS
get_property(CMAKE_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(CMAKE_GENERATOR_IS_MULTI_CONFIG)
    set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
else()
    set(output_dir ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_subdirectory(corrade)
add_subdirectory(magnum)

# This would be bad enough already, but $<TARGET_EXISTS> doesn't exist until
# CMake 3.12, so I need to do two passes, first replacing variables using
# configure_file() and then replacing generator expressions with file(GENERATE)
foreach(target
    corrade_containers
    corrade_pluginmanager
    magnum_gl
    magnum_meshtools
    magnum_primitives
    magnum_scenegraph
    magnum_shaders
    magnum_platform_egl
    magnum_platform_glx
    magnum_platform_glfw
    magnum_platform_sdl2
    magnum_trade)
    if(TARGET ${target})
        set(${target}_file $<TARGET_FILE:${target}>)
    endif()
endforeach()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py.in)
file(GENERATE OUTPUT ${output_dir}/setup.py
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py.in)
