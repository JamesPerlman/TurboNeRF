#pragma once

#include <mutex>
#include <vector>

#include "../common.h"

TURBO_NAMESPACE_BEGIN

template <typename ObservableEvent, typename EventCallbackParam>
class Observable {
public:

    using EventCallback = std::function<void(EventCallbackParam)>;

    struct EventObserver {
        uint32_t id;
        ObservableEvent event;        

        EventCallback callback;

        EventObserver(uint32_t id, ObservableEvent event, EventCallback callback)
            : id(id)
            , event(event)
            , callback(callback)
        {};
    };

    uint32_t add_observer(ObservableEvent event, EventCallback callback) {
        std::lock_guard<std::mutex> lock(_event_dispatch_mutex);
        uint32_t id = _event_observer_id++;
        _event_observers.emplace_back(id, event, callback);
        return id;
    }

    void remove_observer(uint32_t id) {
        std::lock_guard<std::mutex> lock(_event_dispatch_mutex);
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
    
    void dispatch(ObservableEvent event, EventCallbackParam data = {}) {
        std::lock_guard<std::mutex> lock(_event_dispatch_mutex);
        for (auto& observer : _event_observers) {
            if (observer.event == event) {
                observer.callback(data);
            }
        }
    }
    
private:

    uint32_t _event_observer_id = 0;
    std::vector<EventObserver> _event_observers;
    std::mutex _event_dispatch_mutex;

};

TURBO_NAMESPACE_END