#pragma once

#include <glad/glad.h>
#include <memory>
#include <optional>

#include "../controllers/nerf-rendering-controller.h"
#include "../controllers/nerf-training-controller.h"
#include "../models/camera.cuh"
#include "../models/nerf-proxy.cuh"
#include "../models/render-request.cuh"
#include "../render-targets/cpu-render-buffer.cuh"
#include "../utils/queues.h"
#include "../common.h"

TURBO_NAMESPACE_BEGIN

class BlenderBridge 
{
    public:

    enum class ObservableEvent {
        OnTrainingStart,
        OnTrainingStop,
        OnTrainingStep,
        OnPreviewStart,
        OnPreviewProgress,
        OnPreviewComplete,
        OnPreviewCancel,
        OnRenderStart,
        OnRenderProgress,
        OnRenderComplete,
        OnRenderCancel,
        OnRequestRedraw
    };

    private:

    struct EventObserver {
        uint32_t id;
        ObservableEvent event;        

        // TODO: look into sending arbitrary data with the event (via callback)
        // https://github.com/pybind/pybind11_json
        std::function<void()> callback;

        EventObserver(uint32_t id, ObservableEvent event, std::function<void()> callback)
            : id(id)
            , event(event)
            , callback(callback)
        {};
    };

    NeRFRenderingController _previewer;
    NeRFRenderingController _renderer;
    std::optional<NeRFTrainingController> _trainer;
    CPURenderBuffer _preview_target;
    CPURenderBuffer _render_target;
    std::vector<EventObserver> _event_observers;
    uint32_t _event_observer_id = 0;

    TwoItemQueue _render_queue;
    DebounceQueue _draw_queue;

    std::future<void> _runloop_future;
    bool _keep_runloop_alive = false;
    bool _is_training = false;
    bool _is_rendering = false;
    bool _is_previewing = false;

    float _preview_progress = 0.0f;
    float _render_progress = 0.0f;

    GLuint _render_tex_id = 0;
    
    public:

    BlenderBridge()
        : _previewer(RenderPattern::HexagonalGrid)
        , _renderer(RenderPattern::LinearChunks)
        , _render_queue()
        , _draw_queue(1)
    {
        if (!gladLoadGL()) {
            throw std::runtime_error("Failed to load OpenGL with glad.");
        };
    };

    /** EVENT OBSERVERS **/
    public:

    uint32_t add_observer(ObservableEvent event, std::function<void()> callback) {
        uint32_t id = _event_observer_id++;
        _event_observers.emplace_back(id, event, callback);
        return id;
    }

    void remove_observer(uint32_t id) {
        // first find the index of the observer with find_if
        auto it = std::find_if(_event_observers.begin(), _event_observers.end(), [id](const EventObserver& observer) {
            return observer.id == id;
        });
        // if it's invalid, early return
        if (it == _event_observers.end()) {
            return;
        }

        // remove the observer
        _event_observers.erase(it);

        // if the number of observers is 0, reset the event observer id
        if (_event_observers.size() == 0) {
            _event_observer_id = 0;
        }

        // otherwise set the event observer id to the max id + 1
        else {
            _event_observer_id = 1 + std::max_element(
                _event_observers.begin(),
                _event_observers.end(),
                [](const EventObserver& a, const EventObserver& b) {
                    return a.id < b.id;
                }
            )->id;
        }
    }

    private:
    void dispatch(ObservableEvent event) {
        for (auto& observer : _event_observers) {
            if (observer.event == event) {
                observer.callback();
            }
        }
    }

    /**
     * THE RUN LOOP
     * 
     * Training and rendering can be done in an asynchronous run loop, however they must be performed serially in the same background thread.
     * 
     * This run loop serves as a sort of scheduler for the training and rendering operations.
     * 
     */
    private:

    void runloop_worker() {
        do {
            if (_is_training && _trainer.has_value()) {
                // train a single step
                _trainer->train_step();
                auto training_step = _trainer->get_training_step();
                if (training_step % 16 == 0) {
                    _trainer->update_occupancy_grid(training_step);
                }
            }
            // check if we need to render
            _render_queue.work();
            _render_queue.wait();

        } while (_keep_runloop_alive);
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

    /** TRAINING **/
    public:

    uint32_t get_training_step() const {
        if (_trainer.has_value()) {
            return _trainer->get_training_step();
        }
        return 0;
    }

    bool can_train() const {
        return _trainer.has_value();
    }

    bool is_training() const {
        return _is_training;
    }

    void prepare_for_training(Dataset* dataset, NeRFProxy* proxy, const uint32_t& batch_size = NeRFConstants::batch_size) {
        if (!_trainer.has_value()) {
            _trainer = NeRFTrainingController(dataset, proxy, batch_size);
            _trainer->prepare_for_training();
        }
    }

    void start_training() {
        if (_is_training == true) {
            dispatch(ObservableEvent::OnTrainingStart);
        }

        _is_training = true;
        cancel_preview();
        start_runloop(true);
    }

    void stop_training() {
        if (_is_training) {
            dispatch(ObservableEvent::OnTrainingStop);
        }

        _is_training = false;
        stop_runloop();
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
        _renderer.cancel();
    }

    void request_render(const Camera& camera, std::vector<NeRFProxy*>& proxies) {
        if (_is_rendering) {
            return;
        }

        _is_rendering = true;

        cancel_preview();
        stop_training();

        _render_queue.push([this, camera, proxies]() {
            _render_target.set_size(camera.resolution.x, camera.resolution.y);
            auto request = std::make_shared<RenderRequest>(
                camera,
                proxies,
                &this->_render_target,
                RenderFlags::Final,
                // on_complete
                [this]() {
                    this->_render_progress = 1.0f;
                    this->_render_target.synchronize();
                    this->dispatch(ObservableEvent::OnRenderComplete);
                    this->_is_rendering = false;
                },
                // on_progress
                [this](float progress) {
                    this->_render_progress = progress;
                    this->_render_target.synchronize();
                    this->dispatch(ObservableEvent::OnRenderProgress);
                },
                // on_cancel
                [this]() {
                    this->dispatch(ObservableEvent::OnRenderCancel);
                    this->_is_rendering = false;
                }
            );

            this->dispatch(ObservableEvent::OnRenderStart);
            this->_render_target.clear();
            this->_renderer.submit(request);
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
        _previewer.cancel();
    }

    void request_preview(const Camera& camera, std::vector<NeRFProxy*>& proxies, const RenderFlags& flags) {
        cancel_preview();

        if (_is_rendering) {
            return;
        }

        _render_queue.push([this, camera, proxies, flags]() {
            auto request = std::make_shared<RenderRequest>(
                camera,
                proxies,
                &_preview_target,
                flags,
                // on_complete
                [this]() {
                    this->_is_previewing = false;
                    this->dispatch(ObservableEvent::OnPreviewComplete);
                },
                // on_progress
                [this](float progress) {
                    this->_preview_progress = progress;
                    this->dispatch(ObservableEvent::OnPreviewProgress);
                },
                // on_cancel
                [this]() {
                    this->_is_previewing = false;
                    this->dispatch(ObservableEvent::OnPreviewCancel);
                }
            );

            this->_is_previewing = true;
            this->dispatch(ObservableEvent::OnPreviewStart);
            _previewer.submit(request);
        });

        start_runloop(false);
    }

    void resize_preview_surface(const uint32_t& width, const uint32_t& height) {
        _preview_target.set_size(width, height);
    }
    
    void enqueue_redraw() {

        _draw_queue.push([this]() {
            this->_preview_target.synchronize();
            this->dispatch(ObservableEvent::OnRequestRedraw);
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
