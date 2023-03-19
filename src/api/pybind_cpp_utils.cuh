#pragma once

#include "../common.h"

#include <any>
#include <map>
#include <pybind11/pybind11.h>

namespace py = pybind11;

TURBO_NAMESPACE_BEGIN

// Written by GPT-4, prompted by James Perlman

template <typename T>
py::object any_to_py(const std::any& value) {
    return py::cast(std::any_cast<T>(value));
}

py::object any_to_py_object(const std::any& value) {
    #define ANY_TO_PY(TYPE) \
        else if (value.type() == typeid(TYPE)) \
            return any_to_py<TYPE>(value);\
    
    if (false) {}
    
    ANY_TO_PY(bool)
    ANY_TO_PY(int)
    ANY_TO_PY(uint32_t)
    ANY_TO_PY(size_t)
    ANY_TO_PY(double)
    ANY_TO_PY(float)
    ANY_TO_PY(std::string)
    
    else {
        printf("An unsupported data type was encountered in `any_to_py_object`.\n");
        throw std::runtime_error("Unsupported data type.");
    }

    #undef ANY_TO_PY
}

template <typename K, typename V>
py::dict cpp_map_to_py_dict(const std::map<K, V>& cpp_map) {
    py::dict py_dict;
    for (const auto& kv : cpp_map) {
        py_dict[py::cast(kv.first)] = any_to_py_object(kv.second);
    }
    return py_dict;
}

TURBO_NAMESPACE_END
