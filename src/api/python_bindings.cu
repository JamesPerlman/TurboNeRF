#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "../controllers/nerf-rendering-controller.h"
#include "../controllers/nerf-training-controller.h"
#include "../integrations/blender.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../models/dataset.h"
#include "../models/nerf-proxy.cuh"
#include "../models/render-request.cuh"
#include "../render-targets/cuda-render-buffer.cuh"
#include "../render-targets/opengl-render-surface.cuh"
#include "../services/device-manager.cuh"
#include "../services/nerf-manager.cuh"
#include "../utils/linalg/transform4f.cuh"
#include "pybind_cuda.cuh"

namespace py = pybind11;

using namespace turbo;

#define GET(type, name) \
    #name,\
    [](const type& obj) { return obj.name; }

#define GETSET(type, name) \
    #name,\
    [](const type& obj) { return obj.name; },\
    [](type& obj, const auto& value) { obj.name = value; }

PYBIND11_MODULE(PyTurboNeRF, m) {
    /**
     * Global attributes
     * 
     */

    m.doc() = "TurboNeRF Python Bindings";
    m.attr("__version__") = "0.0.3";

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
        .def("from_nerf", &Transform4f::from_nerf)
    ;

    /**
     * Model classes
     */

    py::class_<DistortionParams>(m, "DistortionParams")
        .def(
            py::init<float, float, float, float, float>(),
            py::arg("k1") = 0.0f,
            py::arg("k2") = 0.0f,
            py::arg("k3") = 0.0f,
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
            py::arg("principal_point"),
            py::arg("transform"),
            py::arg("dist_params") = DistortionParams()
        )
        .def_readonly("resolution", &Camera::resolution)
        .def_readonly("near", &Camera::near)
        .def_readonly("far", &Camera::far)
        .def_readonly("focal_length", &Camera::focal_length)
        .def_readonly("principal_point", &Camera::principal_point)
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
        .def("get_size", [](BoundingBox& bb) { return bb.size_x; })
    ;

    py::class_<NeRFProxy>(m, "NeRF")
        .def("get_bounding_box", &NeRFProxy::get_bounding_box)
    ;

    py::enum_<RenderPattern>(m, "RenderPattern")
        .value("LinearChunks", RenderPattern::LinearChunks)
        .value("HexagonalGrid", RenderPattern::HexagonalGrid)
        .value("RectangularGrid", RenderPattern::RectangularGrid)
    ;

    py::class_<RenderTarget>(m, "RenderTarget")
        .def(
            "save_image",
            [](RenderTarget& rt, const string& file_path) {
                rt.save_image(file_path);
            },
            py::arg("file_path")
        )
        .def(
            "set_size",
            [](RenderTarget& rt, const uint32_t& width, const uint32_t& height) {
                rt.set_size(width, height);
            },
            py::arg("width"),
            py::arg("height")
        )
        .def_readonly("width", &RenderTarget::width)
        .def_readonly("height", &RenderTarget::height)
        .def("free", [](RenderTarget& rt) { rt.free(); })
    ;

    py::class_<CPURenderBuffer, RenderTarget>(m, "CPURenderBuffer")
        .def(py::init<>())
        .def("get_rgba", [](CPURenderBuffer& rb) {
            
            rb.synchronize();
            float* rgba = rb.get_rgba();
            
            const int width = rb.width;
            const int height = rb.height;
            
            py::array::StridesContainer strides = {
                sizeof(float) * 4 * width,
                sizeof(float) * 4,
                sizeof(float)
            };
            
            return py::array_t<float>({ height, width, 4 }, strides, rgba);
        })
    ;

    py::class_<CUDARenderBuffer, RenderTarget>(m, "CUDARenderBuffer")
        .def(py::init<>())
    ;

    py::class_<OpenGLRenderSurface, RenderTarget>(m, "OpenGLRenderSurface")
        .def(py::init<>())
    ;

    py::enum_<RenderFlags>(m, "RenderFlags", py::arithmetic())
        .value("Preview", RenderFlags::Preview)
        .value("Final", RenderFlags::Final)
        .def(py::self | py::self)
        .def(py::self & py::self)
        .def(py::self |= py::self)
        .def(py::self &= py::self)
    ;

    py::class_<RenderRequest, std::shared_ptr<RenderRequest>>(m, "RenderRequest")
        .def(
            py::init<
                const Camera&,
                std::vector<NeRFProxy*>&,
                RenderTarget*,
                RenderFlags,
                OnCompleteCallback,
                OnProgressCallback,
                OnCancelCallback
            >(),
            py::arg("camera"),
            py::arg("nerfs"),
            py::arg("output"),
            py::arg("flags") = RenderFlags::Final,
            py::arg("on_complete") = nullptr,
            py::arg("on_progress") = nullptr,
            py::arg("on_cancel") = nullptr
        )
    ;

    /**
     * Controller classes
     */

    py::class_<NeRFRenderingController>(m, "Renderer")
        .def(
           py::init<const RenderPattern&, const uint32_t&>(),
           py::arg("pattern") = RenderPattern::LinearChunks,
           py::arg("batch_size") = 0
        )
        .def(
            "submit",
            &NeRFRenderingController::submit,
            py::arg("request")
        )
    ;

    py::class_<NeRFTrainingController>(m, "Trainer")
        .def(
            py::init<
                Dataset*,
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
            py::arg("training_step")
        )
        .def("get_training_step", &NeRFTrainingController::get_training_step)
        .def("train_step", &NeRFTrainingController::train_step)
    ;

    /**
     * Integration Modules
     */

    /**
     * Blender is the only integration for now.
     * 
     */

    // TODO: This is a large module.  Consider defining it in a separate file.

    py::enum_<BlenderBridge::ObservableEvent>(m, "BlenderBridgeEvent")
        .value("OnTrainingStarted", BlenderBridge::ObservableEvent::OnTrainingStart)
        .value("OnTrainingStopped", BlenderBridge::ObservableEvent::OnTrainingStop)
        .value("OnTrainingStep", BlenderBridge::ObservableEvent::OnTrainingStep)
        .value("OnPreviewStart", BlenderBridge::ObservableEvent::OnPreviewStart)
        .value("OnPreviewProgress", BlenderBridge::ObservableEvent::OnPreviewProgress)
        .value("OnPreviewComplete", BlenderBridge::ObservableEvent::OnPreviewComplete)
        .value("OnPreviewCancel", BlenderBridge::ObservableEvent::OnPreviewCancel)
        .value("OnRenderStart", BlenderBridge::ObservableEvent::OnRenderStart)
        .value("OnRenderProgress", BlenderBridge::ObservableEvent::OnRenderProgress)
        .value("OnRenderComplete", BlenderBridge::ObservableEvent::OnRenderComplete)
        .value("OnRenderCancel", BlenderBridge::ObservableEvent::OnRenderCancel)
        .value("OnRequestRedraw", BlenderBridge::ObservableEvent::OnRequestRedraw)
    ;

    py::class_<BlenderBridge>(m, "BlenderBridge")
        .def(py::init<>())
        // training
        .def("can_train", &BlenderBridge::can_train)
        .def("is_training", &BlenderBridge::is_training)
        .def(
            "prepare_for_training",
            &BlenderBridge::prepare_for_training,
            py::arg("dataset"),
            py::arg("proxy"),
            py::arg("batch_size")
        )
        .def("start_training", &BlenderBridge::start_training)
        .def("stop_training", &BlenderBridge::stop_training)
        .def("wait_for_runloop_to_stop", &BlenderBridge::wait_for_runloop_to_stop)
        // rendering (final)
        .def("is_rendering", &BlenderBridge::is_rendering)
        .def("get_render_progress", &BlenderBridge::get_render_progress)
        .def("cancel_render", &BlenderBridge::cancel_render)
        .def(
            "request_render",
            &BlenderBridge::request_render,
            py::arg("camera"),
            py::arg("proxies")
        )
        .def("get_render_rgba", [](BlenderBridge& bb) {
            float* rgba = bb.get_render_rgba();
            std::size_t n_pixels = bb.get_render_n_pixels();
            return py::memoryview::from_buffer(
                (void*)rgba, // data
                sizeof(float), // size of one element
                py::format_descriptor<float>::value, // format
                { 4 * n_pixels }, // shape
                { 0 }, // strides,
                true
            );
            
        })
        .def("get_render_n_pixels", &BlenderBridge::get_render_n_pixels)
        .def(
            "resize_render_surface",
            &BlenderBridge::resize_render_surface,
            py::arg("width"),
            py::arg("height")
        )
        // rendering (preview)
        .def("is_previewing", &BlenderBridge::is_previewing)
        .def("get_preview_progress", &BlenderBridge::get_preview_progress)
        .def("cancel_preview", &BlenderBridge::cancel_preview)
        .def(
            "request_preview",
            &BlenderBridge::request_preview,
            py::arg("camera"),
            py::arg("proxies"),
            py::arg("flags")
        )
        .def(
            "resize_preview_surface",
            &BlenderBridge::resize_preview_surface,
            py::arg("width"),
            py::arg("height")
        )
        // drawing
        .def("enqueue_redraw", &BlenderBridge::enqueue_redraw)
        .def("draw", &BlenderBridge::draw)
        // event observers
        .def(
            "add_observer",
            &BlenderBridge::add_observer,
            py::arg("event"),
            py::arg("callback")
        )
        .def(
            "remove_observer",
            &BlenderBridge::remove_observer,
            py::arg("id")
        )
    ;

    /**
     * Service classes
     */

    py::class_<NeRFManager>(m, "Manager")
        .def(py::init<>())
        .def(
            "create",
            &NeRFManager::create,
            py::arg("bbox"),
            py::return_value_policy::reference
        )
    ;
}
