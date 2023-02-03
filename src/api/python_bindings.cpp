#include <pybind11/pybind11.h>

PYBIND11_MODULE(PyNeRFRenderCore, m) {
    m.doc() = "NeRFRenderCore Python Bindings";
    m.def("test", []() { return "hello world!"; });
}
