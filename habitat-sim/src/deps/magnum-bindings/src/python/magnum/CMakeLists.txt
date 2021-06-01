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

# *Not* REQUIRED
find_package(Magnum COMPONENTS
    GL
    MeshTools
    Primitives
    SceneGraph
    Shaders
    Trade

    GlfwApplication
    Sdl2Application)
# Find platform-specific apps only on the platforms where it matters, so we
# don't get confusing -- Could NOT find WindowlessWglApplication on Linux and
# such
if(CORRADE_TARGET_UNIX OR MAGNUM_TARGET_GLES)
    find_package(Magnum COMPONENTS WindowlessEglApplication)
endif()
if(CORRADE_TARGET_UNIX)
    find_package(Magnum COMPONENTS WindowlessGlxApplication)
endif()
if(CORRADE_TARGET_WINDOWS)
    find_package(Magnum COMPONENTS WindowlessWglApplication)
endif()

set(magnum_SRCS
    magnum.cpp
    math.cpp
    math.matrixfloat.cpp
    math.matrixdouble.cpp
    math.range.cpp
    math.vectorfloat.cpp
    math.vectorintegral.cpp)

# Extra libraries to link to. Populated only in case of MAGNUM_BUILD_STATIC.
set(magnum_LIBS )

set(magnum_gl_SRCS
    gl.cpp)

set(magnum_meshtools_SRCS
    meshtools.cpp)

set(magnum_primitives_SRCS
    primitives.cpp)

set(magnum_scenegraph_SRCS
    scenegraph.cpp
    scenegraph.matrix.cpp
    scenegraph.trs.cpp)

set(magnum_shaders_SRCS
    shaders.cpp)

set(magnum_trade_SRCS
    trade.cpp)

# If Magnum is not built as static, compile the sub-libraries as separate
# modules
if(NOT MAGNUM_BUILD_STATIC)
    if(Magnum_GL_FOUND)
        pybind11_add_module(magnum_gl ${pybind11_add_module_SYSTEM} ${magnum_gl_SRCS})
        target_include_directories(magnum_gl PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/python)
        target_link_libraries(magnum_gl PRIVATE Magnum::GL)
        set_target_properties(magnum_gl PROPERTIES
            FOLDER "python"
            OUTPUT_NAME "gl"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum)
    endif()

    if(Magnum_MeshTools_FOUND)
        pybind11_add_module(magnum_meshtools ${pybind11_add_module_SYSTEM} ${magnum_meshtools_SRCS})
        target_include_directories(magnum_meshtools PRIVATE ${PROJECT_SOURCE_DIR}/src/python)
        target_link_libraries(magnum_meshtools PRIVATE Magnum::MeshTools)
        set_target_properties(magnum_meshtools PROPERTIES
            FOLDER "python"
            OUTPUT_NAME "meshtools"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum)
    endif()

    if(Magnum_Primitives_FOUND)
        pybind11_add_module(magnum_primitives ${pybind11_add_module_SYSTEM} ${magnum_primitives_SRCS})
        target_include_directories(magnum_primitives PRIVATE ${PROJECT_SOURCE_DIR}/src/python)
        target_link_libraries(magnum_primitives PRIVATE Magnum::Primitives)
        set_target_properties(magnum_primitives PROPERTIES
            FOLDER "python"
            OUTPUT_NAME "primitives"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum)
    endif()

    if(Magnum_SceneGraph_FOUND)
        pybind11_add_module(magnum_scenegraph ${pybind11_add_module_SYSTEM} ${magnum_scenegraph_SRCS})
        target_include_directories(magnum_scenegraph PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/python)
        target_link_libraries(magnum_scenegraph PRIVATE Magnum::SceneGraph)
        set_target_properties(magnum_scenegraph PROPERTIES
            FOLDER "python"
            OUTPUT_NAME "scenegraph"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum)
    endif()

    if(Magnum_Shaders_FOUND)
        pybind11_add_module(magnum_shaders ${pybind11_add_module_SYSTEM} ${magnum_shaders_SRCS})
        target_include_directories(magnum_shaders PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/python)
        target_link_libraries(magnum_shaders PRIVATE Magnum::Shaders)
        set_target_properties(magnum_shaders PROPERTIES
            FOLDER "python"
            OUTPUT_NAME "shaders"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum)
    endif()

    if(Magnum_Trade_FOUND)
        pybind11_add_module(magnum_trade ${pybind11_add_module_SYSTEM} ${magnum_trade_SRCS})
        target_include_directories(magnum_trade PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/python)
        target_link_libraries(magnum_trade PRIVATE Magnum::Trade)
        set_target_properties(magnum_trade PROPERTIES
            FOLDER "python"
            OUTPUT_NAME "trade"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/magnum)
    endif()

# Otherwise put it all into the core library so it's easier to install (which
# is the point of static builds). It also nicely avoids problems with
# multiply-defined global data.
else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/staticconfigure.h.cmake
                   ${CMAKE_CURRENT_BINARY_DIR}/staticconfigure.h)

    if(Magnum_GL_FOUND)
        list(APPEND magnum_SRCS ${magnum_gl_SRCS})
        list(APPEND magnum_LIBS Magnum::GL)
    endif()

    if(Magnum_MeshTools_FOUND)
        list(APPEND magnum_SRCS ${magnum_meshtools_SRCS})
        list(APPEND magnum_LIBS Magnum::MeshTools)
    endif()

    if(Magnum_Primitives_FOUND)
        list(APPEND magnum_SRCS ${magnum_primitives_SRCS})
        list(APPEND magnum_LIBS Magnum::Primitives)
    endif()

    if(Magnum_SceneGraph_FOUND)
        list(APPEND magnum_SRCS ${magnum_scenegraph_SRCS})
        list(APPEND magnum_LIBS Magnum::SceneGraph)
    endif()

    if(Magnum_Shaders_FOUND)
        list(APPEND magnum_SRCS ${magnum_shaders_SRCS})
        list(APPEND magnum_LIBS Magnum::Shaders)
    endif()

    if(Magnum_Trade_FOUND)
        list(APPEND magnum_SRCS ${magnum_trade_SRCS})
        list(APPEND magnum_LIBS Magnum::Trade)
    endif()

    if(Magnum_GlfwApplication_FOUND)
        list(APPEND magnum_SRCS platform/glfw.cpp)
        list(APPEND magnum_LIBS Magnum::GlfwApplication)
    endif()

    if(Magnum_Sdl2Application_FOUND)
        list(APPEND magnum_SRCS platform/sdl2.cpp)
        list(APPEND magnum_LIBS Magnum::Sdl2Application)
    endif()

    if(Magnum_WindowlessEglApplication_FOUND)
        list(APPEND magnum_SRCS platform/egl.cpp)
        list(APPEND magnum_LIBS Magnum::WindowlessEglApplication)
    endif()

    if(Magnum_WindowlessGlxApplication_FOUND)
        list(APPEND magnum_SRCS platform/glx.cpp)
        list(APPEND magnum_LIBS Magnum::WindowlessGlxApplication)
    endif()

    if(Magnum_WindowlessWglApplication_FOUND)
        list(APPEND magnum_SRCS platform/wgl.cpp)
        list(APPEND magnum_LIBS Magnum::WindowlessWglApplication)
    endif()
endif()

pybind11_add_module(magnum ${pybind11_add_module_SYSTEM} ${magnum_SRCS})
target_include_directories(magnum PRIVATE
    ${PROJECT_SOURCE_DIR}/src # SceneGraph/Python.h for static build
    ${PROJECT_SOURCE_DIR}/src/python
    ${PROJECT_BINARY_DIR}/src/python) # for static build
target_link_libraries(magnum PRIVATE Magnum::Magnum ${magnum_LIBS})
set_target_properties(magnum PROPERTIES
    FOLDER "python"
    OUTPUT_NAME "_magnum"
    LIBRARY_OUTPUT_DIRECTORY ${output_dir})

file(GENERATE OUTPUT ${output_dir}/magnum/__init__.py
    INPUT ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py)

add_subdirectory(platform)
