#pragma once

struct RenderProperties {
    bool show_near_planes = false;
    bool show_far_planes = false;

    RenderProperties() = default;
};

struct RenderModifiers {
    RenderProperties properties;

    RenderModifiers() = default;
};
