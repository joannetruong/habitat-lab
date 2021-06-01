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

if(NOT MAGNUM_BUILD_STATIC)
    if(Magnum_GlfwApplication_FOUND)
        pybind11_add_module(magnum_platform_glfw ${pybind11_add_module_SYSTEM} glfw.cpp)
        target_link_libraries(magnum_platform_glfw PRIVATE Magnum::GlfwApplication)
        target_include_directories(magnum_platform_glfw PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/python)
        set_target_properties(magnum_platform_glfw PROPERTIES
            FOLDER "python/platform"
            OUTPUT_NAME "glfw"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum/platform)
    endif()

    if(Magnum_Sdl2Application_FOUND)
        pybind11_add_module(magnum_platform_sdl2 ${pybind11_add_module_SYSTEM} sdl2.cpp)
        target_link_libraries(magnum_platform_sdl2 PRIVATE Magnum::Sdl2Application)
        target_include_directories(magnum_platform_sdl2 PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/python)
        set_target_properties(magnum_platform_sdl2 PROPERTIES
            FOLDER "python/platform"
            OUTPUT_NAME "sdl2"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum/platform)
    endif()

    if(Magnum_WindowlessEglApplication_FOUND)
        pybind11_add_module(magnum_platform_egl ${pybind11_add_module_SYSTEM} egl.cpp)
        target_link_libraries(magnum_platform_egl PRIVATE Magnum::WindowlessEglApplication)
        target_include_directories(magnum_platform_egl PRIVATE ${PROJECT_SOURCE_DIR}/src/python)
        set_target_properties(magnum_platform_egl PROPERTIES
            FOLDER "python/platform"
            OUTPUT_NAME "egl"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum/platform)
    endif()

    if(Magnum_WindowlessGlxApplication_FOUND)
        pybind11_add_module(magnum_platform_glx ${pybind11_add_module_SYSTEM} glx.cpp)
        target_link_libraries(magnum_platform_glx PRIVATE Magnum::WindowlessGlxApplication)
        target_include_directories(magnum_platform_glx PRIVATE ${PROJECT_SOURCE_DIR}/src/python)
        set_target_properties(magnum_platform_glx PROPERTIES
            FOLDER "python/platform"
            OUTPUT_NAME "glx"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum/platform)
    endif()

    if(Magnum_WindowlessWglApplication_FOUND)
        pybind11_add_module(magnum_platform_wgl SYSTEM wgl.cpp)
        target_link_libraries(magnum_platform_wgl PRIVATE Magnum::WindowlessWglApplication)
        target_include_directories(magnum_platform_wgl PRIVATE ${PROJECT_SOURCE_DIR}/src/python)
        set_target_properties(magnum_platform_wgl PROPERTIES
            FOLDER "python/platform"
            OUTPUT_NAME "wgl"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum/platform)
    endif()
endif()

# Not named __init__.py so this directory is not treated as a module by python
# when running python -m unittest in the parent directory (because numpy uses
# python's builtin platform module to check for PyPy and it's all downhill from
# there)
file(GENERATE OUTPUT ${output_dir}/magnum/platform/__init__.py
    INPUT ${CMAKE_CURRENT_SOURCE_DIR}/_init.py)
