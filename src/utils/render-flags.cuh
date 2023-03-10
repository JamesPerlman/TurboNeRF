#pragma once

#include "../common.h"

TURBO_NAMESPACE_BEGIN

enum class RenderFlags: int {
    Preview = 1 << 0,
    Final = 1 << 1,
};

inline RenderFlags operator|(RenderFlags a, RenderFlags b) {
    return static_cast<RenderFlags>(static_cast<int>(a) | static_cast<int>(b));
}

inline RenderFlags operator&(RenderFlags a, RenderFlags b) {
    return static_cast<RenderFlags>(static_cast<int>(a) & static_cast<int>(b));
}

inline RenderFlags& operator|=(RenderFlags& a, RenderFlags b) {
    a = a | b;
    return a;
}

inline RenderFlags& operator&=(RenderFlags& a, RenderFlags b) {
    a = a & b;
    return a;
}

TURBO_NAMESPACE_END
