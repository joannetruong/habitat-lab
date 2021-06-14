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
find_package(Corrade COMPONENTS
    PluginManager)

set(corrade_SRCS
    corrade.cpp)

# Extra libraries to link to. Populated only in case of CORRADE_BUILD_STATIC.
set(corrade_LIBS )

set(corrade_containers_SRCS
    containers.cpp)

set(corrade_pluginmanager_SRCS
    pluginmanager.cpp)

# If Corrade is not built as static, compile the sub-libraries as separate
# modules
if(NOT CORRADE_BUILD_STATIC)
    pybind11_add_module(corrade_containers ${pybind11_add_module_SYSTEM} ${corrade_containers_SRCS})
    target_include_directories(corrade_containers PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/python)
    target_link_libraries(corrade_containers PRIVATE
        Corrade::Containers
        Corrade::Utility)
    set_target_properties(corrade_containers PROPERTIES
        FOLDER "python/corrade"
        OUTPUT_NAME "containers"
        LIBRARY_OUTPUT_DIRECTORY ${output_dir}/corrade)

    if(Corrade_PluginManager_FOUND)
        pybind11_add_module(corrade_pluginmanager ${pybind11_add_module_SYSTEM} ${corrade_pluginmanager_SRCS})
        target_include_directories(corrade_pluginmanager PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/python)
        target_link_libraries(corrade_pluginmanager PRIVATE
            Corrade::PluginManager)
        set_target_properties(corrade_pluginmanager PROPERTIES
            FOLDER "python/corrade"
            OUTPUT_NAME "pluginmanager"
            LIBRARY_OUTPUT_DIRECTORY ${output_dir}/corrade)
    endif()

# Otherwise put it all into one library so it's easier to install (which is the
# point of static builds). It also nicely avoids problems with multiply-defined
# global data.
else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/staticconfigure.h.cmake
                   ${CMAKE_CURRENT_BINARY_DIR}/staticconfigure.h)

    list(APPEND corrade_SRCS ${corrade_containers_SRCS})
    list(APPEND corrade_LIBS Corrade::Containers)

    if(Corrade_PluginManager_FOUND)
        list(APPEND corrade_SRCS ${corrade_pluginmanager_SRCS})
        list(APPEND corrade_LIBS Corrade::PluginManager)
    endif()
endif()

pybind11_add_module(corrade ${pybind11_add_module_SYSTEM} ${corrade_SRCS})
target_include_directories(corrade PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/python
    ${PROJECT_BINARY_DIR}/src/python) # for static build
target_link_libraries(corrade PRIVATE Corrade::Utility ${corrade_LIBS})
set_target_properties(corrade PROPERTIES
    FOLDER "python"
    OUTPUT_NAME "_corrade"
    LIBRARY_OUTPUT_DIRECTORY ${output_dir})

file(GENERATE OUTPUT ${output_dir}/corrade/__init__.py
    INPUT ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
