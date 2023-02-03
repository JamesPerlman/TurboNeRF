#include <pybind11/pybind11.h>

#include "../models/camera.cuh"
#include "../models/dataset.h"

namespace py = pybind11;

using namespace nrc;

PYBIND11_MODULE(PyNeRFRenderCore, m) {
    m.doc() = "NeRFRenderCore Python Bindings";
    m.def("test", []() { return "hello world!"; });

    py::class_<Camera>(m, "Camera");

    /**
     * Dataset
     *  - constructor(filepath)
     * 
     * Trainer
     *  - load_data
     *  - train
     * 
     * Renderer
     *  - create_render_surface
     *  - render_to_surface
     *  - render_to_file
     * 
     * RenderRequest
     *  - constructor
     *    + camera
     *    + resolution
     *    + nerfs
     * 
     * Camera
     *  - constructor
     * 
     * NeuralField
     *  - constructor
     */
}
