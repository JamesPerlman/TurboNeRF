#pragma once

#include "../common.h"

/**
 * Discussion
 * 
 * This is kind of like a database.  It stores references to GPU data buffers and manages their memory.
 * It provides a way to access workspace GPU buffers that are used in training and rendering.
 * 
 */

NRC_NAMESPACE_BEGIN

struct WorkspaceRef {

}

struct WorkspaceManager {

    std::vector<WorkspaceRef> workspaces;
}

NRC_NAMESPACE_END