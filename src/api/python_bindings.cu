#include <json/json.hpp>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11_json/pybind11_json.hpp>
#include <string>
#include <vector>

#include "../controllers/nerf-rendering-controller.h"
#include "../controllers/nerf-training-controller.h"
#include "../integrations/blender.cuh"
#include "../math/transform4f.cuh"
#include "../math/matrix4f.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../models/dataset.h"
#include "../models/nerf-proxy.cuh"
#include "../models/render-request.cuh"
#include "../models/updatable-property.cuh"
#include "../render-targets/cuda-render-buffer.cuh"
#include "../render-targets/opengl-render-surface.cuh"
#include "../services/device-manager.cuh"
#include "../services/nerf-manager.cuh"
#include "../services/runtime-manager.cuh"
#include "pybind_cpp_utils.cuh"
#include "pybind_cuda.cuh"

namespace py = pybind11;

using namespace turbo;

// a macro for easily forwarding get/set calls in pybind to UpdatableProperty
// usage: .def_property(UPDATABLE(SomeClass, PropType, attribute_name))
#define UPDATABLE(CLASS, TYPE, ATTR) \
    #ATTR, \
    [](CLASS& obj) -> TYPE { return obj.ATTR.get(); }, \
    [](CLASS& obj, TYPE value) { obj.ATTR.set(value); }

// a utility function for easily binding UpdatableProperty<T>
template <typename T>
void bind_updatable(py::module& m, const std::string& class_name) {
    py::class_<UpdatableProperty<T>>(m, class_name.c_str())
        .def(py::init<>())
        .def("get", &UpdatableProperty<T>::get)
        .def("set", &UpdatableProperty<T>::set)
        .def("set_dirty", &UpdatableProperty<T>::set_dirty)
        .def("is_dirty", &UpdatableProperty<T>::is_dirty)
    ;
}


PYBIND11_MODULE(PyTurboNeRF, m) {
    /**
     * Global attributes
     * 
     */

    m.doc() = "TurboNeRF Python Bindings";
    m.attr("__version__") = "0.0.19";
    m.attr("__build__") = 0;

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

    py::class_<Matrix4f>(m, "Matrix4f", py::buffer_protocol())
        .def(
            py::init(
                [](py::array_t<float> arr) {
                    const bool is_2d = arr.ndim() == 2;
                    const bool is_4x4 = arr.shape(0) == 4 && arr.shape(1) == 4;
                    
                    if (!is_2d || !is_4x4) {
                        throw std::runtime_error("Invalid shape for Matrix4f");
                    }

                    auto buf = arr.request();
                    auto ptr = (float*)buf.ptr;
                    return Matrix4f{
                        ptr[0],  ptr[1],  ptr[2],  ptr[3],
                        ptr[4],  ptr[5],  ptr[6],  ptr[7],
                        ptr[8],  ptr[9],  ptr[10], ptr[11],
                        ptr[12], ptr[13], ptr[14], ptr[15]
                    };
                }
            ),
            py::arg("matrix")
        )
        .def_buffer([](Matrix4f &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                {4, 4},
                {sizeof(float) * 4, sizeof(float)}
            );
        })
    ;

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
        .def("to_nerf", &Transform4f::to_nerf)
        .def("to_matrix", &Transform4f::to_matrix)
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
        .def_readwrite("k1", &DistortionParams::k1)
        .def_readwrite("k2", &DistortionParams::k2)
        .def_readwrite("k3", &DistortionParams::k3)
        .def_readwrite("p1", &DistortionParams::p1)
        .def_readwrite("p2", &DistortionParams::p2)
    ;

    py::class_<Camera>(m, "Camera")
        .def(
            py::init<
                int2,
                float,
                float,
                float2,
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
            py::arg("shift"),
            py::arg("transform"),
            py::arg("dist_params") = DistortionParams()
        )
        .def_readwrite("resolution", &Camera::resolution)
        .def_readwrite("near", &Camera::near)
        .def_readwrite("far", &Camera::far)
        .def_readwrite("focal_length", &Camera::focal_length)
        .def_readwrite("principal_point", &Camera::principal_point)
        .def_readwrite("shift", &Camera::shift)
        .def_readwrite("transform", &Camera::transform)
        .def_readwrite("dist_params", &Camera::dist_params)
        .def_readwrite("show_image_planes", &Camera::show_image_planes)
        .def(py::self == py::self)
        .def(py::self != py::self)
    ;

    py::class_<Dataset>(m, "Dataset")
        .def(
            py::init<const string&>(),
            py::arg("file_path")
        )
        .def("load_transforms", &Dataset::load_transforms)
        .def("copy", &Dataset::copy)
        .def("to_json", &Dataset::to_json)
        .def(
            "set_camera_at",
            [](Dataset& ds, int index, Camera& cam) {
                ds.cameras[index] = cam;
            }
        )
        .def_readwrite("file_path", &Dataset::file_path)
        .def_readwrite("cameras", &Dataset::cameras)
        .def_readwrite("bounding_box", &Dataset::bounding_box)
        .def_readonly("image_dimensions", &Dataset::image_dimensions)
        .def_readonly("contains_multiple_image_dims", &Dataset::contains_multiple_image_dims)
    ;
    
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def(
            py::init<float>(),
            py::arg("size")
        )
        .def("size", &BoundingBox::size)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def_readwrite("max_x", &BoundingBox::max_x)
        .def_readwrite("max_y", &BoundingBox::max_y)
        .def_readwrite("max_z", &BoundingBox::max_z)
        .def_readwrite("min_x", &BoundingBox::min_x)
        .def_readwrite("min_y", &BoundingBox::min_y)
        .def_readwrite("min_z", &BoundingBox::min_z)
    ;

    bind_updatable<BoundingBox>(m, "UpdatableBoundingBox");
    bind_updatable<Transform4f>(m, "UpdatableTransform4f");

    py::class_<NeRFProxy>(m, "NeRF")
        .def("attach_dataset", &NeRFProxy::attach_dataset)
        .def("detach_dataset", &NeRFProxy::detach_dataset)
        .def("is_dirty", &NeRFProxy::is_dirty)
        .def("can_train", &NeRFProxy::can_train)
        .def("is_image_data_loaded", &NeRFProxy::is_image_data_loaded)
        .def_property(UPDATABLE(NeRFProxy, BoundingBox, render_bbox))
        .def_property(UPDATABLE(NeRFProxy, BoundingBox, training_bbox))
        .def_property(UPDATABLE(NeRFProxy, Transform4f, transform))
        .def_readwrite("is_visible", &NeRFProxy::is_visible)
        .def_readwrite("is_dataset_dirty", &NeRFProxy::is_dataset_dirty)
        .def_readonly("dataset", &NeRFProxy::dataset)
        .def_readonly("can_render", &NeRFProxy::can_render)
        .def_readonly("training_step", &NeRFProxy::training_step)
        .def_readonly("id", &NeRFProxy::id)
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

    py::class_<RenderProperties>(m, "RenderProperties")
        .def(py::init<>())
        .def_readwrite("show_near_planes", &RenderProperties::show_near_planes)
        .def_readwrite("show_far_planes", &RenderProperties::show_far_planes)
    ;

    py::class_<RenderModifiers>(m, "RenderModifiers")
        .def(py::init<>())
        .def_readwrite("properties", &RenderModifiers::properties)
    ;

    py::class_<RenderRequest, std::shared_ptr<RenderRequest>>(m, "RenderRequest")
        .def(
            py::init<
                const Camera&,
                std::vector<NeRFProxy*>&,
                RenderTarget*,
                const RenderModifiers&,
                const RenderFlags&,
                OnCompleteCallback,
                OnProgressCallback,
                OnCancelCallback
            >(),
            py::arg("camera"),
            py::arg("nerfs"),
            py::arg("output"),
            py::arg("modifiers") = RenderModifiers(),
            py::arg("flags") = RenderFlags::Final,
            py::arg("on_complete") = nullptr,
            py::arg("on_progress") = nullptr,
            py::arg("on_cancel") = nullptr
        )
    ;

    /**
     * Controller classes
     */

    py::class_<NeRFRenderingController, std::shared_ptr<NeRFRenderingController>>(m, "Renderer")
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
        .def_readwrite("min_step_size", &NeRFRenderingController::min_step_size)
    ;

    // TrainingController helpers

    py::class_<NeRFTrainingController::TrainingMetrics>(m, "TrainingMetrics")
        .def_readonly("step", &NeRFTrainingController::TrainingMetrics::step)
        .def_readonly("loss", &NeRFTrainingController::TrainingMetrics::loss)
        .def_readonly("n_rays", &NeRFTrainingController::TrainingMetrics::n_rays)
        .def_readonly("n_samples", &NeRFTrainingController::TrainingMetrics::n_samples)
    ;

    py::class_<NeRFTrainingController::OccupancyGridMetrics>(m, "OccupancyGridMetrics")
        .def_readonly("n_occupied", &NeRFTrainingController::OccupancyGridMetrics::n_occupied)
        .def_readonly("n_total", &NeRFTrainingController::OccupancyGridMetrics::n_total)
    ;

    // TrainingController class

    py::class_<NeRFTrainingController, std::shared_ptr<NeRFTrainingController>>(m, "Trainer")
        .def(
            py::init<NeRFProxy*>(),
            py::arg("nerf")
        )
        .def(
            "setup_data",
            &NeRFTrainingController::setup_data,
            py::arg("batch_size"),
            "Call this once before starting training."
        )
        .def(
            "teardown",
            &NeRFTrainingController::teardown,
            "Call this before destroying the controller."
        )
        .def(
            "reset_training",
            &NeRFTrainingController::reset_training,
            "Call this to clear the training data."
        )
        .def(
            "load_images",
            [](NeRFTrainingController& tc, py::object on_image_loaded) {
                // coded with a generous amount of help from GPT-4
                if (py::isinstance<py::function>(on_image_loaded)) {
                    // Capture a reference to the on_image_loaded function
                    auto on_image_loaded_func = std::make_shared<py::function>(on_image_loaded.cast<py::function>());

                    // Release the GIL before calling the C++ function
                    py::gil_scoped_release release;

                    tc.load_images([on_image_loaded_func](int n_loaded, int n_total) {
                        // Re-acquire the GIL when calling the Python function
                        py::gil_scoped_acquire acquire;
                        (*on_image_loaded_func)(n_loaded, n_total);
                    });
                } else {
                    tc.load_images();
                }
            },
            py::arg("on_image_loaded") = py::none()
        )
        .def(
            "update_occupancy_grid",
            &NeRFTrainingController::update_occupancy_grid,
            py::arg("training_step")
        )
        .def("train_step", &NeRFTrainingController::train_step)
        .def_readwrite("alpha_selection_threshold", &NeRFTrainingController::alpha_selection_threshold)
        .def_readwrite("alpha_selection_probability", &NeRFTrainingController::alpha_selection_probability)
        .def_readwrite("min_step_size", &NeRFTrainingController::min_step_size)
        .def_readwrite("use_distortion_loss", &NeRFTrainingController::use_distortion_loss)
        .def_readwrite("nerf", &NeRFTrainingController::proxy)
    ;

    /**
     * Integration Modules
     */

    /**
     * Blender is the only integration for now.
     * 
     */

    // TODO: This is a large module.  Consider defining it in a separate file.

    py::enum_<BlenderBridge::Event>(m, "BlenderBridgeEvent")
        .value("OnDestroyNeRF", BlenderBridge::Event::OnDestroyNeRF)
        .value("OnUpdateOccupancyGrid", BlenderBridge::Event::OnUpdateOccupancyGrid)
        .value("OnPreviewStart", BlenderBridge::Event::OnPreviewStart)
        .value("OnPreviewProgress", BlenderBridge::Event::OnPreviewProgress)
        .value("OnPreviewComplete", BlenderBridge::Event::OnPreviewComplete)
        .value("OnPreviewCancel", BlenderBridge::Event::OnPreviewCancel)
        .value("OnRenderStart", BlenderBridge::Event::OnRenderStart)
        .value("OnRenderProgress", BlenderBridge::Event::OnRenderProgress)
        .value("OnRenderComplete", BlenderBridge::Event::OnRenderComplete)
        .value("OnRenderCancel", BlenderBridge::Event::OnRenderCancel)
        .value("OnRequestRedraw", BlenderBridge::Event::OnRequestRedraw)
        .value("OnTrainingImageLoaded", BlenderBridge::Event::OnTrainingImageLoaded)
        .value("OnTrainingImagesLoadComplete", BlenderBridge::Event::OnTrainingImagesLoadComplete)
        .value("OnTrainingImagesLoadStart", BlenderBridge::Event::OnTrainingImagesLoadStart)
        .value("OnTrainingImagesUnloaded", BlenderBridge::Event::OnTrainingImagesUnloaded)
        .value("OnTrainingReset", BlenderBridge::Event::OnTrainingReset)
        .value("OnTrainingStart", BlenderBridge::Event::OnTrainingStart)
        .value("OnTrainingStop", BlenderBridge::Event::OnTrainingStop)
        .value("OnTrainingStep", BlenderBridge::Event::OnTrainingStep)
    ;

    py::class_<BlenderBridge>(m, "BlenderBridge")
        .def(py::init<>())
        // properties
        .def_readonly("previewer", &BlenderBridge::previewer)
        .def_readonly("renderer", &BlenderBridge::renderer)
        // training
        .def("get_trainer_for_nerf", &BlenderBridge::trainer_for_proxy, py::arg("nerf"), py::return_value_policy::reference)
        .def("get_nerf", &BlenderBridge::get_nerf, py::arg("nerf_id"), py::return_value_policy::reference)
        .def("get_nerfs", &BlenderBridge::get_nerfs, py::return_value_policy::reference)
        .def("create_nerf", &BlenderBridge::create_nerf, py::arg("dataset"), py::return_value_policy::reference)
        .def("clone_nerf", &BlenderBridge::clone_nerf, py::arg("nerf"), py::return_value_policy::reference)
        .def("destroy_nerf", &BlenderBridge::destroy_nerf, py::arg("nerf"))
        .def("load_nerf", &BlenderBridge::load_nerf, py::arg("path"), py::return_value_policy::reference)
        .def("save_nerf", &BlenderBridge::save_nerf, py::arg("nerf"), py::arg("path"))
        .def("can_load_training_images", &BlenderBridge::can_load_training_images, py::arg("nerf"))
        .def("load_training_images", &BlenderBridge::load_training_images, py::arg("nerf"))
        .def("can_any_nerf_train", &BlenderBridge::can_any_nerf_train)
        .def("is_training", &BlenderBridge::is_training)
        .def("start_training", &BlenderBridge::start_training)
        .def("stop_training", &BlenderBridge::stop_training)
        .def("is_training_enabled", &BlenderBridge::is_training_enabled, py::arg("nerf"))
        .def("enable_training", &BlenderBridge::enable_training, py::arg("nerf"))
        .def("disable_training", &BlenderBridge::disable_training, py::arg("nerf"))
        .def("reset_training", &BlenderBridge::reset_training, py::arg("nerf"))
        .def("unload_training_images", &BlenderBridge::unload_training_images)
        // rendering (final)
        .def("is_rendering", &BlenderBridge::is_rendering)
        .def("get_render_progress", &BlenderBridge::get_render_progress)
        .def("cancel_render", &BlenderBridge::cancel_render)
        .def(
            "request_render",
            &BlenderBridge::request_render,
            py::arg("camera"),
            py::arg("proxies"),
            py::arg("modifiers") = RenderModifiers()
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
            py::arg("flags"),
            py::arg("modifiers") = RenderModifiers()
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
            [](BlenderBridge& bb, BlenderBridge::Event event, py::function callback) -> uint32_t {
                return bb.event_bus.add_observer(event, [callback](BlenderBridge::EventCallbackParam e) {
                    py::gil_scoped_acquire acquire;
                    callback(cpp_map_to_py_dict(e));
                });
            },
            py::arg("event"),
            py::arg("callback")
        )
        .def(
            "remove_observer",
            [](BlenderBridge& bb, uint32_t id) {
                bb.event_bus.remove_observer(id);
            },
            py::arg("id")
        )
    ;

    /**
     * Service classes
     */

    py::class_<NeRFManager>(m, "NeRFManager")
        .def(py::init<>())
        .def(
            "create",
            &NeRFManager::create,
            py::return_value_policy::reference
        )
        .def(
            "clone",
            &NeRFManager::clone,
            py::arg("nerf"),
            py::return_value_policy::reference
        )
        .def(
            "destroy",
            &NeRFManager::destroy,
            py::arg("nerf")
        )
    ;

    py::class_<FileManager>(m, "FileManager")
        .def_static(
            "load",
            [](NeRFProxy* nerf, std::string& path) {
                FileManager::load(nerf, path);
            },
            py::arg("nerf"),
            py::arg("path")
        )
        .def_static(
            "save",
            [](NeRFProxy* nerf, std::string& path) {
                FileManager::save(nerf, path);
            },
            py::arg("nerf"),
            py::arg("path")
        )
    ;

    /**
     * Runtime Utils (for checking CUDA compatibility)
     */

    py::class_<RuntimeVersion>(m, "RuntimeVersion")
        .def_readonly("major", &RuntimeVersion::major)
        .def_readonly("minor", &RuntimeVersion::minor)
        .def_readonly("subminor", &RuntimeVersion::subminor)
        .def("to_string", &RuntimeVersion::to_string)
        .def_static("CompiledRuntimeVersion", &RuntimeVersion::CompiledRuntimeVersion)
    ;

    py::class_<DeviceArchitecture>(m, "DeviceArchitecture")
        .def_readonly("major", &DeviceArchitecture::major)
        .def_readonly("minor", &DeviceArchitecture::minor)
        .def("to_string", &DeviceArchitecture::to_string)
        .def("arch_name", &DeviceArchitecture::arch_name)
    ;

    py::class_<RuntimeManager>(m, "RuntimeManager")
        .def(py::init<>())
        .def_readonly("cuda_archs", &RuntimeManager::cuda_archs)
        .def_static("required_runtime_version", &RuntimeManager::required_runtime_version)
        .def_static("current_runtime_version", &RuntimeManager::current_runtime_version)
        .def_static("is_driver_version_sufficient", &RuntimeManager::is_driver_version_sufficient)
        .def_static("get_device_architecture", &RuntimeManager::get_device_architecture)
        .def("check_runtime", &RuntimeManager::check_runtime)
    ;
}
