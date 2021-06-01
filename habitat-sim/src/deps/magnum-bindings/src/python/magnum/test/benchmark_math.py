#!/usr/bin/env python3

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

# Avoid this being run implicitly during unit tests
if __name__ != '__main__': exit()

import timeit

import array
from magnum import *
import numpy as np

repeats = 100000

def timethat(expr: str, *, setup:str = 'pass', title=None):
    if not title:
        if setup != 'pass': title = f'{setup}; {expr}'
        else: title = expr

    print('{:67} {:8.5f} µs'.format(title, timeit.timeit(expr, number=repeats, globals=globals(), setup=setup)*1000000.0/repeats))

def timethat_exception(expr: str):
    return timethat(f"""
try:
    {expr}
except:
    pass
""", title=f'{expr} # throws')

print("  plain list expressions:\n")

timethat('[]')
timethat('list([])')
timethat('[1.0, 2.0, 3.0]')
timethat('list([1.0, 2.0, 3.0])')

print("\n  Vector3 from/to list:\n")

timethat('Vector3()')
timethat('Vector3(1.0, 2.0, 3.0)')
timethat('Vector3([1.0, 2.0, 3.0])')
timethat('list(a)', setup='a = Vector3(1.0, 2.0, 3.0)')
timethat('[a.x, a.y, a.z]', setup='a = Vector3(1.0, 2.0, 3.0)')

print("\n  Vector3 from/to builtin array:\n")

timethat('array.array("f", [])')
timethat('array.array("f", [1.0, 2.0, 3.0])')
timethat('memoryview(a)', setup='a = array.array("f", [1.0, 2.0, 3.0])')
timethat('memoryview(a)', setup='a = Vector3(1.0, 2.0, 3.0)')
timethat('Vector3(a)', setup='a = array.array("f", [1.0, 2.0, 3.0])')

print("\n  Vector3 from/to np.array:\n")

timethat('np.array([])')
timethat('np.array([1.0, 2.0, 3.0])')
timethat('np.array(a)', setup='a = array.array("f", [1.0, 2.0, 3.0])')
timethat('np.array(a)', setup='a = Vector3(1.0, 2.0, 3.0)')
timethat('Vector3(a)', setup='a = np.array([1.0, 2.0, 3.0])')

print("\n  Matrix3 from/to list, equivalent np.array operations:\n")

timethat('Matrix3()')
timethat('Matrix3.from_diagonal(Vector3(1.0, 2.0, 3.0))')
timethat('Matrix3.from_diagonal([1.0, 2.0, 3.0])')
timethat('list(Matrix3.from_diagonal(Vector3(1.0, 2.0, 3.0)))')
timethat('np.diagflat([1.0, 2.0, 3.0])')
timethat('np.array(Matrix3.from_diagonal(Vector3(1.0, 2.0, 3.0)))')

print("\n  exception throwing:\n")

timethat('Vector3()[0] # doesn\'t throw')
timethat_exception('Vector3()[3]')
timethat_exception('raise IndexError()')

print("\n  basic operations:\n")

timethat('a + a', setup='a = Vector4d(1.0, 2.0, 3.0, 4.0)')
timethat('a + a', setup='a = np.array([1.0, 2.0, 3.0, 4.0])')
timethat('a.dot()', setup='a = Vector4d(1.0, 2.0, 3.0, 4.0)')
timethat('np.dot(a, a)', setup='a = np.array([1.0, 2.0, 3.0, 4.0])')
timethat('a@a', setup='a = Matrix4d.from_diagonal([1.0, 2.0, 3.0, 4.0])')
timethat('a@a', setup='a = np.diagflat([1.0, 2.0, 3.0, 4.0])')
