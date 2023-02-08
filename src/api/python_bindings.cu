#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "../controllers/nerf-rendering-controller.h"
#include "../controllers/nerf-training-controller.h"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../models/dataset.h"
#include "../models/nerf-proxy.cuh"
#include "../models/render-buffer.cuh"
#include "../models/render-request.cuh"
#include "../services/device-manager.cuh"
#include "../services/nerf-manager.cuh"
#include "../utils/linalg/transform4f.cuh"
#include "pybind_cuda.cuh"

namespace py = pybind11;

using namespace nrc;

#define GET(type, name) \
    #name,\
    [](const type& obj) { return obj.name; }

#define GETSET(type, name) \
    #name,\
    [](const type& obj) { return obj.name; },\
    [](type& obj, const auto& value) { obj.name = value; }

PYBIND11_MODULE(PyNeRFRenderCore, m) {
    m.doc() = "NeRFRenderCore Python Bindings";

    /**
     * Global functions
     */

    m.def(
        "teardown",
        []() {
            DeviceManager::teardown();
        }
    );
    
    /**
     * Utility classes
     */

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
        })
    ;

    /**
     * Model classes
     */

    py::class_<DistortionParams>(m, "DistortionParams")
        .def(
            py::init<float, float, float, float, float, float>(),
            py::arg("k1") = 0.0f,
            py::arg("k2") = 0.0f,
            py::arg("k3") = 0.0f,
            py::arg("k4") = 0.0f,
            py::arg("p1") = 0.0f,
            py::arg("p2") = 0.0f
        )
    ;

    py::class_<Camera>(m, "Camera")
        .def(
            py::init<
                int2,
                float,
                float,
                float2,
                float2,
                Transform4f,
                DistortionParams
            >(),
            py::arg("resolution"),
            py::arg("near"),
            py::arg("far"),
            py::arg("focal_length"),
            py::arg("view_angle"),
            py::arg("transform"),
            py::arg("dist_params") = DistortionParams()
        )
        .def_readonly("resolution", &Camera::resolution)
        .def_readonly("near", &Camera::near)
        .def_readonly("far", &Camera::far)
        .def_readonly("focal_length", &Camera::focal_length)
        .def_readonly("view_angle", &Camera::view_angle)
        .def_readonly("transform", &Camera::transform)
        .def_readonly("dist_params", &Camera::dist_params)
    ;

    py::class_<Dataset>(m, "Dataset")
        .def(
            py::init<const string&>(),
            py::arg("file_path")
        )
        .def_readonly("cameras", &Dataset::cameras)
        .def_readonly("image_dimensions", &Dataset::image_dimensions)
        .def_readonly("bounding_box", &Dataset::bounding_box)
    ;
    
    // TODO: split into Training and Rendering bbox?
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(
            py::init<float>(),
            py::arg("size")
        )
    ;

    py::class_<NeRFProxy>(m, "NeRF");

    py::class_<RenderBuffer>(m, "RenderBuffer")
        .def(
            py::init<const uint32_t&, const uint32_t&>(),
            py::arg("width"),
            py::arg("height")
        )
        .def(
            "save_image",
            [](RenderBuffer& rb, const string& file_path) {
                rb.save_image(file_path);
            },
            py::arg("file_path")
        )
        .def("get_image", [](RenderBuffer& rb) { return rb.get_image(); })
        .def_readonly("width", &RenderBuffer::width)
        .def_readonly("height", &RenderBuffer::height)
    ;

    py::class_<RenderRequest>(m, "RenderRequest")
        .def(
            py::init<
                const Camera&,
                std::vector<NeRFProxy*>&,
                RenderBuffer*
            >(),
            py::arg("camera"),
            py::arg("nerfs"),
            py::arg("output")
        )
    ;

    /**
     * Controller classes
     */

    py::class_<NeRFRenderingController>(m, "Renderer")
        .def(
           py::init<const uint32_t&>(),
           py::arg("batch_size") 
        )
        .def(
            "request",
            &NeRFRenderingController::request,
            py::arg("request")
        )
    ;

    py::class_<NeRFTrainingController>(m, "Trainer")
        .def(
            py::init<
                Dataset&,
                NeRFProxy*,
                const uint32_t&
            >(),
            py::arg("dataset"),
            py::arg("nerf"),
            py::arg("batch_size")
        )
        .def(
            "prepare_for_training",
            &NeRFTrainingController::prepare_for_training,
            "Call this once before starting training."
        )
        .def(
            "update_occupancy_grid",
            &NeRFTrainingController::update_occupancy_grid,
            py::arg("selection_threshold")
        )
        .def(
            "train_step",
            &NeRFTrainingController::train_step
        )
    ;

    /**
     * Service classes
     */

    py::class_<NeRFManager>(m, "Manager")
        .def(py::init<>())
        .def(
            "create_trainable",
            &NeRFManager::create_trainable,
            py::arg("bbox"),
            py::return_value_policy::reference
        )
    ;
}
