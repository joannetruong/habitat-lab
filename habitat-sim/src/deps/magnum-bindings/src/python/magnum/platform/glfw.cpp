/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020, 2021 Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#include <pybind11/pybind11.h>
#include <Magnum/Platform/GlfwApplication.h>

#include "Corrade/PythonBindings.h"

#include "magnum/bootstrap.h"
#include "magnum/platform/application.h"
#include "magnum/platform/holder.h"

namespace magnum { namespace platform {

namespace {
    int argc = 0;
}

void glfw(py::module_& m) {
    m.doc() = "GLFW-based platform integration";

    struct PublicizedApplication: Platform::Application {
        explicit PublicizedApplication(const Configuration& configuration, const GLConfiguration& glConfiguration): Platform::Application{Arguments{argc, nullptr}, configuration, glConfiguration} {}

        void drawEvent() override {
            PyErr_SetString(PyExc_NotImplementedError, "the application has to provide a draw_event() method");
            throw py::error_already_set{};
        }

        void keyPressEvent(KeyEvent&) override {}
        void keyReleaseEvent(KeyEvent&) override {}
        void mousePressEvent(MouseEvent&) override {}
        void mouseReleaseEvent(MouseEvent&) override {}
        void mouseMoveEvent(MouseMoveEvent&) override {}
        void mouseScrollEvent(MouseScrollEvent&) override {}

        /* The base doesn't have a virtual destructor because in C++ it's never
           deleted through a pointer to the base. Here we need it, though. */
        virtual ~PublicizedApplication() {}
    };

    struct PyApplication: PublicizedApplication {
        using PublicizedApplication::PublicizedApplication;

        void drawEvent() override {
            #ifdef __clang__
            /* ugh pybind don't tell me I AM THE FIRST ON EARTH to get a
               warning here. Why there's no PYBIND11_OVERLOAD_NAME_ARG()
               variant *with* arguments and one without? */
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
            #endif
            PYBIND11_OVERLOAD_NAME(
                void,
                PublicizedApplication,
                "draw_event",
                drawEvent
            );
            #ifdef __clang__
            #pragma GCC diagnostic pop
            #endif
        }

        /* PYBIND11_OVERLOAD_NAME() calls object_api::operator() with implicit
           template param, which is return_value_policy::automatic_reference.
           That later gets changed to return_value_policy::copy in
           type_caster_base::cast() and there's no way to override that  */
        void keyPressEvent(KeyEvent& event) override {
            PYBIND11_OVERLOAD_NAME(
                void,
                PublicizedApplication,
                "key_press_event",
                keyPressEvent,
                /* Have to use std::ref() otherwise pybind tries to copy
                   it and fails */
                std::ref(event)
            );
        }
        void keyReleaseEvent(KeyEvent& event) override {
            PYBIND11_OVERLOAD_NAME(
                void,
                PublicizedApplication,
                "key_release_event",
                keyReleaseEvent,
                /* Have to use std::ref() otherwise pybind tries to copy
                   it and fails */
                std::ref(event)
            );
        }

        void mousePressEvent(MouseEvent& event) override {
            PYBIND11_OVERLOAD_NAME(
                void,
                PublicizedApplication,
                "mouse_press_event",
                mousePressEvent,
                /* Have to use std::ref() otherwise pybind tries to copy
                   it and fails */
                std::ref(event)
            );
        }
        void mouseReleaseEvent(MouseEvent& event) override {
            PYBIND11_OVERLOAD_NAME(
                void,
                PublicizedApplication,
                "mouse_release_event",
                mouseReleaseEvent,
                /* Have to use std::ref() otherwise pybind tries to copy
                   it and fails */
                std::ref(event)
            );
        }
        void mouseMoveEvent(MouseMoveEvent& event) override {
            PYBIND11_OVERLOAD_NAME(
                void,
                PublicizedApplication,
                "mouse_move_event",
                mouseMoveEvent,
                /* Have to use std::ref() otherwise pybind tries to copy
                   it and fails */
                std::ref(event)
            );
        }
        void mouseScrollEvent(MouseScrollEvent& event) override {
            PYBIND11_OVERLOAD_NAME(
                void,
                PublicizedApplication,
                "mouse_scroll_event",
                mouseScrollEvent,
                /* Have to use std::ref() otherwise pybind tries to copy
                   it and fails */
                std::ref(event)
            );
        }
    };

    py::class_<PublicizedApplication, PyApplication, ApplicationHolder<PublicizedApplication>> glfwApplication{m, "Application", "GLFW application"};
    /** @todo def_property_writeonly for swap_interval */

    PyNonDestructibleClass<PublicizedApplication::InputEvent> inputEvent_{glfwApplication, "InputEvent", "Base for input events"};
    py::class_<PublicizedApplication::KeyEvent, PublicizedApplication::InputEvent> keyEvent_{glfwApplication, "KeyEvent", "Key event"};
    py::class_<PublicizedApplication::MouseEvent, PublicizedApplication::InputEvent> mouseEvent_{glfwApplication, "MouseEvent", "Mouse event"};
    py::class_<PublicizedApplication::MouseMoveEvent, PublicizedApplication::InputEvent> mouseMoveEvent_{glfwApplication, "MouseMoveEvent", "Mouse move event"};
    py::class_<PublicizedApplication::MouseScrollEvent, PublicizedApplication::InputEvent> mouseScrollEvent_{glfwApplication, "MouseScrollEvent", "Mouse scroll event"};

    application(glfwApplication);
    inputEvent(inputEvent_);
    keyEvent(keyEvent_);
    mouseEvent(mouseEvent_);
    mouseMoveEvent(mouseMoveEvent_);
    mouseScrollEvent(mouseScrollEvent_);
}

}}

#ifndef MAGNUM_BUILD_STATIC
/* TODO: remove declaration when https://github.com/pybind/pybind11/pull/1863
   is released */
extern "C" PYBIND11_EXPORT PyObject* PyInit_glfw();
PYBIND11_MODULE(glfw, m) {
    magnum::platform::glfw(m);
}
#endif
