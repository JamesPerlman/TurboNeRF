#pragma once

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "render-target.cuh"

// Adapted from https://www.informit.com/articles/article.aspx?p=2455391&seqNum=2 - thank you informIT!

NRC_NAMESPACE_BEGIN

class OpenGLRenderSurface: public RenderTarget {
private:
    GLuint pbo;
    GLuint texture;
    cudaGraphicsResource *cuda_pbo_resource;

    void allocate(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) override {
        this->width = width;
        this->height = height;

        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

        CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsNone));

        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    void resize(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) override {
        this->width = width;
        this->height = height;

        CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsNone));
    }
    
public:

    using RenderTarget::RenderTarget;

    GLuint get_texture_id() const {
        return texture;
    }

    void free(const cudaStream_t& stream = 0) override {
        if (width == 0 || height == 0)
            return;
        
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &texture);
    }

    void open_for_cuda_access(std::function<void(float* rgba)> handle, const cudaStream_t& stream = 0) override {
        float* rgba;
        CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &cuda_pbo_resource, stream));
        CUDA_CHECK_THROW(cudaGraphicsResourceGetMappedPointer((void **)&rgba, nullptr, cuda_pbo_resource));

        handle(rgba);

        CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, NULL);

        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
};

NRC_NAMESPACE_END
