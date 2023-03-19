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
    if (value.type() == typeid(int)) {
        return any_to_py<int>(value);
    } else if (value.type() == typeid(uint32_t)) {
        return any_to_py<uint32_t>(value);
    } else if (value.type() == typeid(float)) {
        return any_to_py<float>(value);
    } else if (value.type() == typeid(std::string)) {
        return any_to_py<std::string>(value);
    } else {
        throw std::runtime_error("Unsupported data type.");
    }
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
