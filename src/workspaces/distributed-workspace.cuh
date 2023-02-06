#pragma once

#include <vector>

#include "../common.h"
#include "../models/nerf-proxy.cuh"

#include "workspace.cuh"


NRC_NAMESPACE_BEGIN

template <typename T>
struct DistributedWorkspace {
    static_assert(std::is_base_of<Workspace, T>::value, "T must be a subclass of Workspace");

    std::vector<T> workspaces;

    DistributedWorkspace(const int& n_devices = 0) {
        workspaces.reserve(n_devices);
        for (int i = 0; i < n_devices; i++) {
            workspaces.emplace_back(i);
        }
    }
};

NRC_NAMESPACE_END
