/**
 * Some handy queue implementations.
 * Written by James Perlman on 2023-02-14
 * MIT-licensed
 */

#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <vector>

/**
 * This is a queue that processes one task at a time,
 * It can hold a single extra task on reserve.
 * This next task gets overwritten by the latest pushed task.
 */

class TwoItemQueue {
    std::function<void()> current_task = nullptr;
    std::function<void()> next_task = nullptr;
    std::future<void> worker;
    std::mutex m;

    void worker_thread() {
        do {
            current_task();

            std::lock_guard<std::mutex> lock(m);
            current_task = std::move(next_task);
            next_task = nullptr;
        } while (current_task != nullptr);
    }

    public:
    TwoItemQueue() = default;

    void push(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(m);

        // is the current task empty?
        if (current_task == nullptr) {
            // this task becomes the current task
            current_task = std::move(task);
        } else {
            // otherwise save it for after the current task, overwriting as necessary
            next_task = std::move(task);
        }

        // if the worker is not active
        if (!worker.valid() || worker.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            // start a new worker
            worker = std::async(
                std::launch::async,
                [this] {
                    worker_thread();
                }
            );
        }
    }
};

/**
 * This is not a true debounce queue.
 * It accounts for the time the task takes to run.
 */

class DebounceQueue: TwoItemQueue {
    std::chrono::milliseconds delay;

    public:
    DebounceQueue(long long ms_delay)
        : delay(ms_delay)
    {};

    void push(std::function<void()> task) {
        const std::chrono::milliseconds delay = this->delay;
        TwoItemQueue::push([task, delay] {
            task();
            std::this_thread::sleep_for(delay);
        });
    }
};
