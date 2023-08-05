#pragma once
#include "../common.h"

TURBO_NAMESPACE_BEGIN

template <typename T>
struct UpdatableProperty {
private:
    T _value;
    bool _is_dirty;

public:
    __host__
    UpdatableProperty() : _value(T()), _is_dirty(true) {}

    // conversion constructor
    __host__
    UpdatableProperty(const T& other_value) : _value(other_value), _is_dirty(true) {}

    // copy constructor
    __host__
    UpdatableProperty(const UpdatableProperty& other) : _value(other._value), _is_dirty(true) {}

    // copy assigment operators
    __host__
    UpdatableProperty& operator=(const T& other_value) {
        _value = other_value;
        _is_dirty = true;
        return *this;
    }

    __host__
    UpdatableProperty& operator=(const UpdatableProperty& other) {
        _value = other._value;
        _is_dirty = true;
        return *this;
    }

    // getter and setter
    __host__
    void set(T newValue) {
        _value = newValue;
        _is_dirty = true;
    }

    __host__
    T get() const {
        return _value;
    }

    __host__
    void copy_to_device(
        T* device_ptr,
        cudaStream_t stream = 0
    ) {
        cudaMemcpyAsync(
            device_ptr,
            &_value,
            sizeof(T),
            cudaMemcpyHostToDevice,
            stream
        );

        _is_dirty = false;
    }

    // is_dirty getter and setter
    __host__
    bool is_dirty() const {
        return _is_dirty;
    }
    
    __host__
    void set_dirty(const bool& is_dirty = true) {
        _is_dirty = is_dirty;
    }
};

TURBO_NAMESPACE_END
