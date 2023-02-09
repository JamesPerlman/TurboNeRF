#pragma once

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

#include "render-target.cuh"

// Adapted from https://www.informit.com/articles/article.aspx?p=2455391&seqNum=2 - thank you informIT!

NRC_NAMESPACE_BEGIN

class OpenGLRenderSurface: public RenderTarget {
private:
    GLuint pbo;
    GLuint texture;
    cudaGraphicsResource *cuda_pbo_resource;

    // openGL references
public:
    using RenderTarget::RenderTarget;

    GLuint get_texture_id() const {
        return texture;
    }

    void allocate(const cudaStream_t& stream = 0) override {
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsNone);
    }

    void free(const cudaStream_t& stream = 0) override {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &texture);
    }

    void resize(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) override {
        this->width = width;
        this->height = height;
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);
    }

    void open_for_cuda_access(std::function<void(float* rgba)> handle) override {
        float *rgba;
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&rgba, nullptr, cuda_pbo_resource);

        handle(rgba);
        
        cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    }
};

NRC_NAMESPACE_END
