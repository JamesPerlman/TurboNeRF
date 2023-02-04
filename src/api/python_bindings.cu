#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../utils/linalg/transform4f.cuh"
#include "../models/camera.cuh"
#include "../models/dataset.h"

namespace py = pybind11;

using namespace nrc;

PYBIND11_MODULE(PyNeRFRenderCore, m) {
    m.doc() = "NeRFRenderCore Python Bindings";
    m.def("test", []() { return "hello world!"; });

    py::class_<Transform4f>(m, "Transform4f", py::buffer_protocol())
        .def(
            py::init(
                [](py::array_t<float> arr) {
                    const bool is_2d = arr.ndim() == 2;
                    const bool is_3x4 = arr.shape(0) == 3 && arr.shape(1) == 4;
                    const bool is_4x4 = arr.shape(0) == 4 && arr.shape(1) == 4;
                    
                    if (!is_2d || (!is_3x4 && !is_4x4)) {
                        throw std::runtime_error("Invalid shape for Transform4f");
                    }

                    auto buf = arr.request();
                    auto ptr = (float*)buf.ptr;
                    return Transform4f{
                        ptr[0], ptr[1], ptr[2],  ptr[3],
                        ptr[4], ptr[5], ptr[6],  ptr[7],
                        ptr[8], ptr[9], ptr[10], ptr[11]
                    };
                }
            ),
            py::arg("matrix")
        )
        .def_buffer([](Transform4f &t) -> py::buffer_info {
            return py::buffer_info(
                t.data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                {3, 4},
                {sizeof(float) * 4, sizeof(float)}
            );
        });

    py::class_<DistortionParams>(m, "DistortionParams")
        .def(
            py::init<float, float, float, float, float, float>(),
            py::arg("k1") = 0.0f,
            py::arg("k2") = 0.0f,
            py::arg("k3") = 0.0f,
            py::arg("k4") = 0.0f,
            py::arg("p1") = 0.0f,
            py::arg("p2") = 0.0f
        );

    py::class_<Camera>(m, "Camera")
        .def(
            py::init<
                float,
                float,
                float2,
                int2,
                float2,
                Transform4f,
                DistortionParams
            >(),
            py::arg("near"),
            py::arg("far"),
            py::arg("focal_length"),
            py::arg("resolution"),
            py::arg("sensor_size"),
            py::arg("transform"),
            py::arg("dist_params") = DistortionParams()
        )
        .def("get_res_x", &Camera::get_res_x);

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
