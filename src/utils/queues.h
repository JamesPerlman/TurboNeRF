/**
 * Some handy queue implementations.
 * Written by James Perlman on 2023-02-14
 * MIT-licensed
 */

#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

/**
 * This is a queue that processes one task at a time,
 * It can hold a single extra task on reserve.
 * This next task gets overwritten by the latest pushed task.
 */

class TwoItemQueue {
protected:
    std::function<void()> current_task = nullptr;
    std::function<void()> next_task = nullptr;
    std::future<void> worker;
    std::mutex m;

    virtual void worker_thread() {
        do {
            current_task();

            std::lock_guard lock(m);
            current_task = std::move(next_task);
            next_task = nullptr;
        } while (current_task != nullptr);
    }

public:
    TwoItemQueue() = default;

    virtual void push(std::function<void()> task) {
        std::lock_guard lock(m);

        // is the current task empty?
        if (current_task == nullptr) {
            // this task becomes the current task
            current_task = std::move(task);
        } else {
            // otherwise save it for after the current task, overwriting as necessary
            next_task = std::move(task);
        }
    }

    void work() {
        // is there even any work to do?
        std::unique_lock lock(m);
        if (current_task == nullptr) {
            return;
        }
        lock.unlock();

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

    void wait() {
        if (worker.valid()) {
            worker.wait();
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

    void push(std::function<void()> task) override {
        const std::chrono::milliseconds delay = this->delay;
        TwoItemQueue::push([task, delay] {
            task();
            std::this_thread::sleep_for(delay);
        });
    }

    using TwoItemQueue::work;
    using TwoItemQueue::wait;
};

/**
 * This is a trailing delay queue.
 * New items can be added for near-immediate execution within some delay time after the last item finishes.
 * 
 */

class TrailingDelayQueue: public TwoItemQueue {
    std::chrono::milliseconds delay;
    std::chrono::milliseconds wait_loop_granularity;

protected:
    void worker_thread() override {
        const int n_wait_iters = delay.count() / wait_loop_granularity.count();

        do {
            current_task();

            std::unique_lock lock(m);
            current_task = nullptr;
            lock.unlock();

            // wait for (delay) ms in intervals of (wait_loop_granularity) ms or until a new task is pushed
            auto then = std::chrono::steady_clock::now();
            while (true) {
                std::this_thread::sleep_for(wait_loop_granularity);

                std::lock_guard lock(m);
                if (current_task != nullptr) {
                    break;
                }

                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - then) > delay) {
                    break;
                }
            };
        } while (current_task != nullptr);
    }

public:
    TrailingDelayQueue(long long ms_delay, long long wait_loop_granularity = 10)
        : delay(ms_delay)
        , wait_loop_granularity(wait_loop_granularity)
    {};
};
