#pragma once

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

// typecasters for various cuda types
// https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html

namespace PYBIND11_NAMESPACE { namespace detail {
    // int2
    template <> struct type_caster<int2> {
        PYBIND11_TYPE_CASTER(int2, const_name("int2"));
        bool load(handle src, bool convert) {
            PyObject* tmp = PySequence_Tuple(src.ptr());

            if (!tmp) return false;

            Py_DECREF(tmp);

            Py_ssize_t size = PyTuple_GET_SIZE(tmp);

            if (size != 2) {
                PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 2");
                return false;
            }

            value.x = PyLong_AsLong(PyTuple_GET_ITEM(tmp, 0));
            value.y = PyLong_AsLong(PyTuple_GET_ITEM(tmp, 1));

            return !PyErr_Occurred();
        }

        static handle cast(int2 src, return_value_policy policy, handle parent) {
            tuple t(2);
            t[0] = src.x;
            t[1] = src.y;
            return t.release();
        }
    };

    // float2
    template <> struct type_caster<float2> {
        PYBIND11_TYPE_CASTER(float2, const_name("float2"));
        bool load(handle src, bool convert) {
            PyObject* tmp = PySequence_Tuple(src.ptr());

            if (!tmp) return false;

            Py_DECREF(tmp);

            Py_ssize_t size = PyTuple_GET_SIZE(tmp);

            if (size != 2) {
                PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 2");
                return false;
            }

            value.x = PyFloat_AsDouble(PyTuple_GET_ITEM(tmp, 0));
            value.y = PyFloat_AsDouble(PyTuple_GET_ITEM(tmp, 1));

            return !PyErr_Occurred();
        }

        static handle cast(float2 src, return_value_policy policy, handle parent) {
            tuple t(2);
            t[0] = src.x;
            t[1] = src.y;
            return t.release();
        }
    };
}} // namespace PYBIND11_NAMESPACE::detail
