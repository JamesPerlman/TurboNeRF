#pragma once

#include <chrono>
#include <functional>
#include <stdint.h>

#include "../render-targets/render-target.cuh"
#include "../common.h"
#include "camera.cuh"
#include "nerf-proxy.cuh"

NRC_NAMESPACE_BEGIN

struct RenderRequest;

typedef std::function<void(RenderRequest&)> OnCancelCallback;
typedef std::function<void(RenderRequest&)> OnCompleteCallback;
typedef std::function<void(RenderRequest&)> OnProgressCallback;

struct RenderRequest {
private:
    bool _canceled = false;
    OnCompleteCallback _on_complete;
    OnProgressCallback _on_progress;
    OnCancelCallback _on_cancel;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

public:
    const Camera camera;
    std::vector<NeRFProxy*> proxies;
    RenderTarget* output;

    RenderRequest(
        const Camera& camera,
        std::vector<NeRFProxy*>& proxies,
        RenderTarget* output,
        OnCompleteCallback on_complete = nullptr,
        OnProgressCallback on_progress = nullptr,
        OnCancelCallback on_cancel = nullptr
    )
        : camera(camera)
        , proxies(proxies)
        , output(output)
        , _on_complete(on_complete)
        , _on_progress(on_progress)
        , _on_cancel(on_cancel)
        , _start_time(std::chrono::high_resolution_clock::now())
    { };
    
    RenderRequest()
        : camera(Camera())
        , proxies(std::vector<NeRFProxy*>())
        , output(nullptr)
        , _on_complete(nullptr)
        , _on_progress(nullptr)
        , _on_cancel(nullptr)
        , _start_time(std::chrono::high_resolution_clock::now())
    { };

    bool is_canceled() const {
        return _canceled;
    }

    void cancel() {
        _canceled = true;
    }

    // this is probably an anti-pattern...

    void on_cancel() {
        if (_on_cancel) {
            _on_cancel(*this);
        }
    }

    void on_complete() {
        if (_on_complete) {
            _on_complete(*this);
        }
    }

    void on_progress() {
        if (_on_progress) {
            _on_progress(*this);
        }
    }

    // equality operator checks if start time & camera is the same
    bool operator==(const RenderRequest& other) const {
        return _start_time == other._start_time && camera == other.camera;
    }

    bool operator!=(const RenderRequest& other) const {
        return !(*this == other);
    }
};

NRC_NAMESPACE_END
