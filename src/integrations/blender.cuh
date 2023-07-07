#pragma once

#include <any>
#include <glad/glad.h>
#include <map>
#include <memory>
#include <optional>

#include "../controllers/nerf-rendering-controller.h"
#include "../controllers/nerf-training-controller.h"
#include "../models/camera.cuh"
#include "../models/nerf-proxy.cuh"
#include "../models/render-request.cuh"
#include "../render-targets/cpu-render-buffer.cuh"
// #include "../services/file-manager.cuh"
#include "../services/nerf-manager.cuh"
#include "../utils/observable.cuh"
#include "../utils/queues.h"
#include "../common.h"

TURBO_NAMESPACE_BEGIN

class BlenderBridge 
{
     /** EVENT OBSERVERS & HELPERS **/
    public:
    
    enum class Event {
        OnDestroyNeRF,
        OnUpdateOccupancyGrid,
        OnPreviewStart,
        OnPreviewProgress,
        OnPreviewComplete,
        OnPreviewCancel,
        OnRenderStart,
        OnRenderProgress,
        OnRenderComplete,
        OnRenderCancel,
        OnRequestRedraw,
        OnTrainingImageLoaded,
        OnTrainingImagesLoadComplete,
        OnTrainingImagesLoadStart,
        OnTrainingImagesUnloaded,
        OnTrainingReset,
        OnTrainingStart,
        OnTrainingStop,
        OnTrainingStep
    };

    using EventCallbackParam = std::map<std::string, std::any>;

    Observable<Event, EventCallbackParam> event_bus;

    // be careful when accessing these
    std::shared_ptr<NeRFRenderingController> previewer = nullptr;
    std::shared_ptr<NeRFRenderingController> renderer = nullptr;

    // one training controller per NeRFProxy
    std::vector<NeRFTrainingController> trainers;

    /** PRIVATE PROPERTIES */
    private:

    NeRFManager _nerf_manager;
    CPURenderBuffer _preview_target;
    CPURenderBuffer _render_target;    

    TwoItemQueue _render_queue;
    DebounceQueue _draw_queue;

    std::future<void> _runloop_future;
    std::future<void> _img_load_future;

    bool _keep_runloop_alive = false;
    bool _is_training = false;
    bool _is_rendering = false;
    bool _is_previewing = false;

    float _preview_progress = 0.0f;
    float _render_progress = 0.0f;

    GLuint _render_tex_id = 0;

    std::vector<uint32_t> _nerf_manager_observers;

    public:

    BlenderBridge()
        : previewer(new NeRFRenderingController(RenderPattern::HexagonalGrid))
        , renderer(new NeRFRenderingController(RenderPattern::LinearChunks))
        , _render_queue()
        , _draw_queue(1)
        , _nerf_manager()
    {
        if (!gladLoadGL()) {
            throw std::runtime_error("Failed to load OpenGL with glad.");
        };

        trainers.resize(NeRFManager::n_max_nerfs());
    };

    // just forward these to the bus for now
    uint32_t add_observer(Event event, std::function<void(EventCallbackParam)> callback) {
        return event_bus.add_observer(event, callback);
    }

    void remove_observer(uint32_t id) {
        event_bus.remove_observer(id);
    }

    private:

    #define HANDLE_EVENT(event_name) case Event::event_name: printf(#event_name "\n"); break

    void log_event(Event event) {
        switch (event) {
            HANDLE_EVENT(OnDestroyNeRF);
            HANDLE_EVENT(OnUpdateOccupancyGrid);
            HANDLE_EVENT(OnPreviewStart);
            HANDLE_EVENT(OnPreviewProgress);
            HANDLE_EVENT(OnPreviewComplete);
            HANDLE_EVENT(OnPreviewCancel);
            HANDLE_EVENT(OnRenderStart);
            HANDLE_EVENT(OnRenderProgress);
            HANDLE_EVENT(OnRenderComplete);
            HANDLE_EVENT(OnRenderCancel);
            HANDLE_EVENT(OnRequestRedraw);
            HANDLE_EVENT(OnTrainingImageLoaded);
            HANDLE_EVENT(OnTrainingImagesLoadComplete);
            HANDLE_EVENT(OnTrainingImagesLoadStart);
            HANDLE_EVENT(OnTrainingImagesUnloaded);
            HANDLE_EVENT(OnTrainingReset);
            HANDLE_EVENT(OnTrainingStart);
            HANDLE_EVENT(OnTrainingStop);
            HANDLE_EVENT(OnTrainingStep);
        }
    }

    #undef HANDLE_EVENT

    /**
     * THE RUN LOOP
     * 
     * Training and rendering can be done in an asynchronous run loop, however they must be performed serially in the same background thread.
     * 
     * This run loop serves as a sort of scheduler for the training and rendering operations.
     * 
     */
    private:

    // this is needed for certain actions to work while training
    bool is_waiting_for_some_work() {
        for (int i = 0; i < _nerf_manager.n_proxies(); ++i) {
            auto proxy = _nerf_manager.proxy_for_id(i);
            if (!proxy->is_valid) {
                continue;
            }

            if (proxy->should_reset || proxy->should_destroy || proxy->should_free_training_data) {
                return true;
            }
        }
        return false;
    }

    void runloop_worker() {
        
        // potential TODO here - the event_bus.dispatch() calls will slow down the run loop depending on how many observers there are, and what the callbacks do
        // so we may want to add them to a queue and event_bus.dispatch them in another thread.  Although this can become problematic too.
        // in general this loop is horrible.  it is robust to poor thread management but there must be a more beautiful architecture.

        do {
            for (int i = 0; i < _nerf_manager.n_proxies(); ++i) {
                auto proxy = _nerf_manager.proxy_for_id(i);
                
                if (!proxy->is_valid) {
                    continue;
                }
                
                auto trainer = trainer_for_proxy(proxy);
                
                // check if we need to train a step
                if (_is_training && proxy->should_train && proxy->can_train()) {
                    // train a single step
                    auto metrics = trainer->train_step();
                    auto training_step = proxy->training_step;
                    
                    if (training_step % 16 == 0) {
                        auto occ_metrics = trainer->update_occupancy_grid(training_step);
                        auto occ_args = occ_metrics.as_map();
                        occ_args["id"] = proxy->id;
                        event_bus.dispatch(Event::OnUpdateOccupancyGrid, occ_args);
                    }

                    auto metrics_args = metrics.as_map();
                    metrics_args["id"] = proxy->id;
                    event_bus.dispatch(Event::OnTrainingStep, metrics_args);
                }
                
                // check if we need to reset training
                if (proxy->should_reset) {
                    proxy->should_reset = false;
                    if (proxy->can_train()) {
                        trainer->reset_training();
                        event_bus.dispatch(Event::OnTrainingReset, {{"id", proxy->id}});
                    }
                }

                // check if we need to delete the nerf
                if (proxy->should_destroy) {
                    proxy->should_destroy = false;
                    if (proxy->can_train()) {
                        trainer->teardown();
                    }

                    auto proxy_id = proxy->id;
                    _nerf_manager.destroy(proxy);
                    event_bus.dispatch(Event::OnDestroyNeRF, {{"id", proxy_id}});
                }

                // check if we need to unload training data
                if (proxy->should_free_training_data) {
                    proxy->should_free_training_data = false;
                    trainer->teardown();
                    event_bus.dispatch(Event::OnTrainingImagesUnloaded, {{"id", proxy->id}});
                }
            }

            // check if we need to render
            _render_queue.work();
            _render_queue.wait();
            
            // this is potentially leaky if a runloop is started and is_waiting_for_some_work doesn't return the most up-to-date value
        } while (_keep_runloop_alive || is_waiting_for_some_work());
    }

    void start_runloop(bool keep_alive) {
        if (keep_alive) {
            _keep_runloop_alive = true;
        }

        bool worker_inactive = !_runloop_future.valid() || _runloop_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        
        if (worker_inactive) {
            // start the run loop
            _runloop_future = std::async(
                std::launch::async,
                [this]() {
                    this->runloop_worker();
                }
            );
        }
    }

    void stop_runloop() {
        _keep_runloop_alive = false;
    }

    void wait_for_runloop() {
        if (_runloop_future.valid()) {
            _runloop_future.wait();
        }
    }

    /** TRAINING **/
    public:

    /** NERF MANAGER -> TRAINERS SYNCHRONIZATION **/

    NeRFTrainingController* trainer_for_proxy(const NeRFProxy* proxy) {
        for (int i = 0; i < _nerf_manager.n_proxies(); ++i) {
            if (proxy == _nerf_manager.proxy_for_id(i)) {
                auto& trainer = trainers[i];
                return &trainer;
            }
        }
        throw std::runtime_error("No trainer found for proxy with id " + std::to_string(proxy->id));
    }
    
    /** NERF OBJECT CREATION / CLONING / DESTRUCTION **/

    NeRFProxy* get_nerf(int id) {
        return _nerf_manager.proxy_for_id(id);
    }

    std::vector<NeRFProxy*> get_nerfs() {
        std::vector<NeRFProxy*> proxies;
        for (int i = 0; i < _nerf_manager.n_proxies(); ++i) {
            auto proxy = _nerf_manager.proxy_for_id(i);
            if (proxy != nullptr && proxy->is_valid) {
                proxies.push_back(proxy);
            }
        }
        return proxies;
    }
    
    NeRFProxy* create_nerf(const Dataset& dataset) {
        auto proxy = _nerf_manager.create();
        proxy->attach_dataset(dataset);
        auto trainer = trainer_for_proxy(proxy);
        trainer->proxy = proxy;
        return proxy;
    }

    NeRFProxy* clone_nerf(NeRFProxy* proxy) {
        auto clone = _nerf_manager.clone(proxy);
        auto trainer = trainer_for_proxy(clone);
        trainer->proxy = clone;
        trainer->setup_data();
        return clone;
    }

    void destroy_nerf(NeRFProxy* proxy) {
        cancel_preview();
        proxy->should_destroy = true;
        start_runloop(false);
    }

    bool can_any_nerf_train() {
        for (int i = 0; i < _nerf_manager.n_proxies(); ++i) {
            NeRFProxy* proxy = _nerf_manager.proxy_for_id(i);
            if (proxy == nullptr) {
                continue;
            }
            
            if (proxy->can_train()) {
                return true;
            }
        }

        return false;
    }

    // detect if any nerfs should train
    bool should_any_nerf_train() {
        for (int i = 0; i < _nerf_manager.n_proxies(); ++i) {
            NeRFProxy* proxy = _nerf_manager.proxy_for_id(i);
            if (proxy->should_train && proxy->can_train()) {
                return true;
            }
        }

        return false;
    }

    bool is_training() const {
        return _is_training;
    }

    void load_training_images(NeRFProxy* proxy) {
        auto trainer = trainer_for_proxy(proxy);
    
        _img_load_future = std::async(
            std::launch::async,
            [this, trainer, proxy]() {
                this->event_bus.dispatch(
                    Event::OnTrainingImagesLoadStart,
                    {
                        {"id", proxy->id},
                        {"n_total", proxy->dataset->images.size()}
                    }
                );

                trainer->setup_data();

                trainer->load_images(
                    [this, proxy](int n_loaded, int n_total) {
                        std::map<std::string, std::any> data{
                            {"id", proxy->id},
                            {"n_loaded", n_loaded},
                            {"n_total", n_total}
                        };
                        this->event_bus.dispatch(
                            Event::OnTrainingImageLoaded,
                            data);
                    }
                );

                proxy->should_train = true;
                
                this->event_bus.dispatch(Event::OnTrainingImagesLoadComplete, {{"id", proxy->id}});
            }
        );
    }

    void unload_training_images(NeRFProxy* proxy) {
        if (!proxy->can_train()) {
            return;
        }

        proxy->should_train = false;
        proxy->should_free_training_data = true;

        if (!should_any_nerf_train()) {
            stop_training();
        }

        start_runloop(false);
    }

    void start_training() {
        if (_is_training == true) {
            return;
        }

        _is_training = true;
        cancel_preview();
        start_runloop(true);
        event_bus.dispatch(Event::OnTrainingStart);
    }
    
    void stop_training() {
        if (!_is_training) {
            return;
        }

        _is_training = false;
        stop_runloop();
        event_bus.dispatch(Event::OnTrainingStop);
    }

    void enable_training(NeRFProxy* proxy) {
        if (proxy->should_train) {
            return;
        }

        proxy->should_train = true;

        // start_training();
    }

    void disable_training(NeRFProxy* proxy) {
        if (!proxy->should_train) {
            return;
        }

        proxy->should_train = false;

        if (!should_any_nerf_train()) {
            stop_training();
        }
    }

    void reset_training(NeRFProxy* proxy) {
        if (!proxy->can_train()) {
            return;
        }

        cancel_preview();
        start_runloop(false);
    }

    void wait_for_runloop_to_stop() {
        while (_keep_runloop_alive) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    /** RENDERING (FINAL) **/
    public:

    bool is_rendering() const {
        return _is_rendering;
    }

    float get_render_progress() const {
        return _render_progress;
    }

    void cancel_render() {
        renderer->cancel();
    }

    void request_render(
        const Camera& camera,
        std::vector<NeRFProxy*>& proxies,
        const RenderModifiers& modifiers = RenderModifiers()
    ) {
        if (_is_rendering) {
            return;
        }

        _is_rendering = true;

        cancel_preview();
        stop_training();

        _render_queue.push([this, camera, proxies, modifiers]() {
            _render_target.set_size(camera.resolution.x, camera.resolution.y);
            auto request = std::make_shared<RenderRequest>(
                camera,
                proxies,
                &this->_render_target,
                modifiers,
                RenderFlags::Final,
                // on_complete
                [this]() {
                    this->_render_progress = 1.0f;
                    this->_render_target.synchronize();
                    this->event_bus.dispatch(Event::OnRenderComplete);
                    this->_is_rendering = false;
                },
                // on_progress
                [this](float progress) {
                    this->_render_progress = progress;
                    this->_render_target.synchronize();
                    this->event_bus.dispatch(Event::OnRenderProgress);
                },
                // on_cancel
                [this]() {
                    this->event_bus.dispatch(Event::OnRenderCancel);
                    this->_is_rendering = false;
                }
            );

            this->event_bus.dispatch(Event::OnRenderStart);
            this->_render_target.clear();
            this->renderer->submit(request);
        });
        
        start_runloop(false);
    }

    void resize_render_surface(const uint32_t& width, const uint32_t& height) {
        _render_target.set_size(width, height);
    }

    size_t get_render_n_pixels() const {
        return _render_target.n_pixels();
    }

    float* get_render_rgba() const {
        return _render_target.get_rgba();
    }

    /** RENDERING (PREVIEW) **/
    public:

    bool is_previewing() const {
        return _is_previewing;
    }

    float get_preview_progress() const {
        return _preview_progress;
    }

    void cancel_preview() {
        previewer->cancel();
    }

    void request_preview(
        const Camera& camera,
        std::vector<NeRFProxy*>& proxies,
        const RenderFlags& flags,
        const RenderModifiers& modifiers = RenderModifiers()    
    ) {
        cancel_preview();

        if (_is_rendering) {
            return;
        }

        _render_queue.push([this, camera, proxies, modifiers, flags]() {
            auto request = std::make_shared<RenderRequest>(
                camera,
                proxies,
                &_preview_target,
                modifiers,
                flags,
                // on_complete
                [this]() {
                    this->_is_previewing = false;
                    this->event_bus.dispatch(Event::OnPreviewComplete);
                },
                // on_progress
                [this](float progress) {
                    this->_preview_progress = progress;
                    this->event_bus.dispatch(Event::OnPreviewProgress);
                },
                // on_cancel
                [this]() {
                    this->_is_previewing = false;
                    this->event_bus.dispatch(Event::OnPreviewCancel);
                }
            );

            this->_is_previewing = true;
            this->event_bus.dispatch(Event::OnPreviewStart);
            this->previewer->submit(request);
        });

        start_runloop(false);
    }

    void resize_preview_surface(const uint32_t& width, const uint32_t& height) {
        _preview_target.set_size(width, height);
    }
    
    void enqueue_redraw() {

        _draw_queue.push([this]() {
            this->_preview_target.synchronize();
            this->event_bus.dispatch(Event::OnRequestRedraw);
        });
        _draw_queue.work();
    }
    
    /**
     * The following functionality was adapted from the Pixar RenderMan for Blender plugin.
     * https://github.com/prman-pixar/RenderManForBlender/blob/main/display_driver/d_blender.cpp#L335
     * 
     */
    private:

    // create or resize render texture
    void update_render_texture() {
        // if texture is not created, create it
        if (_render_tex_id == 0) {
            glGenTextures(1, &_render_tex_id);
            glBindTexture(GL_TEXTURE_2D, _render_tex_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }

        // fetch texture size
        int width, height;
        glBindTexture(GL_TEXTURE_2D, _render_tex_id);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);

        // if texture size is not correct, resize it
        if (width != _preview_target.width || height != _preview_target.height) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _preview_target.width, _preview_target.height, 0, GL_RGBA, GL_FLOAT, _preview_target.get_rgba());
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _preview_target.width, _preview_target.height, GL_RGBA, GL_FLOAT, _preview_target.get_rgba());
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    public:
    
    // This will always be called from Blender's view_draw() function

    void draw()
    {
        update_render_texture();

        // These are the vertices of a quad that covers the entire viewport.
        const float w = static_cast<float>(_preview_target.width);
        const float h = static_cast<float>(_preview_target.height);

        GLfloat vertices[8] = {
            0.0f, 0.0f,
               w, 0.0f,
               w,    h,
            0.0f,    h
        };

        GLfloat tex_coords[8] = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 0.0f,
        };
        
        GLint shader_program_id;
        glGetIntegerv(GL_CURRENT_PROGRAM, &shader_program_id);

        GLuint vertex_array;
        glGenVertexArrays(1, &vertex_array);
        glBindVertexArray(vertex_array);

        // Get the location of the "texCoord" and "pos" attributes in the shader program
        GLuint texcoord_location = glGetAttribLocation(shader_program_id, "texCoord");
        GLuint position_location = glGetAttribLocation(shader_program_id, "pos");

        glEnableVertexAttribArray(texcoord_location);
        glEnableVertexAttribArray(position_location);

        // Generate a vertex buffer object for the texture coordinates
        GLuint texture_vbo_id;
        glGenBuffers(1, &texture_vbo_id);
        glBindBuffer(GL_ARRAY_BUFFER, texture_vbo_id);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), &tex_coords[0], GL_STATIC_DRAW);
        glVertexAttribPointer(texcoord_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), reinterpret_cast<void*>(0));

        GLuint vertex_vbo_id;
        // Generate a vertex buffer object for the positions
        glGenBuffers(1, &vertex_vbo_id);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo_id);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);

        // Enable the "pos" attribute
        glEnableVertexAttribArray(position_location);
        glVertexAttribPointer(position_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), reinterpret_cast<void*>(0));

        glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind the current buffer
        glBindVertexArray(0); // Unbind the vertex array object

        // Activate texture unit 0
        glActiveTexture(GL_TEXTURE0);

        // Upload render target to GPU
        glBindTexture(GL_TEXTURE_2D, _render_tex_id);
        glBindVertexArray(vertex_array);

        // Draw the triangle fan
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        glBindVertexArray(0); // Unbind the vertex array object
        glDeleteVertexArrays(1, &vertex_array); // Delete the vertex array object
        glBindTexture(GL_TEXTURE_2D, 0); // Unbind the texture

        // Clean up

        // Delete the vertex array object
        glDeleteVertexArrays(1, &vertex_array);
        // Delete the vertex buffer objects
        glDeleteBuffers(1, &vertex_vbo_id);
        glDeleteBuffers(1, &texture_vbo_id);
    }
};

TURBO_NAMESPACE_END
