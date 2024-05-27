// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "jax_rasterize_gl.h"
#include <tuple>

//------------------------------------------------------------------------
// Forward op (OpenGL).
RasterizeGLStateWrapper::RasterizeGLStateWrapper(bool enableDB, bool automatic_, int cudaDeviceIdx_)
{
    pState = new RasterizeGLState();
    automatic = automatic_;
    cudaDeviceIdx = cudaDeviceIdx_;
    memset(pState, 0, sizeof(RasterizeGLState));
    pState->enableDB = enableDB ? 1 : 0;
    rasterizeInitGLContext(NVDR_CTX_PARAMS, *pState, cudaDeviceIdx_);
    releaseGLContext();
}

RasterizeGLStateWrapper::~RasterizeGLStateWrapper(void)
{
    setGLContext(pState->glctx);
    rasterizeReleaseBuffers(NVDR_CTX_PARAMS, *pState);
    releaseGLContext();
    destroyGLContext(pState->glctx);
    delete pState;
}

void RasterizeGLStateWrapper::setContext(void)
{
    setGLContext(pState->glctx);
}

void RasterizeGLStateWrapper::releaseContext(void)
{
    releaseGLContext();
}



void jax_rasterize_fwd_original_gl(cudaStream_t stream,
                          void **buffers,
                          const char *opaque, std::size_t opaque_len) {
    const DiffRasterizeCustomCallDescriptor &d =
        *UnpackDescriptor<DiffRasterizeCustomCallDescriptor>(opaque, opaque_len);
    RasterizeGLStateWrapper& stateWrapper = *d.gl_state_wrapper;


    const float *pos = reinterpret_cast<const float *> (buffers[0]);
    const int *tri = reinterpret_cast<const int *> (buffers[1]);
    const int *_ranges = reinterpret_cast<const int *> (buffers[2]);
    const int *_resolution = reinterpret_cast<const int *> (buffers[3]);

    float *out = reinterpret_cast<float *> (buffers[4]);

    std::vector<int> resolution;
    resolution.resize(2);
    int ranges[2*d.num_objects];

    NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&resolution[0], _resolution, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&ranges[0], _ranges, 2 * d.num_objects * sizeof(int), cudaMemcpyDeviceToHost));
\
    // std::cout << "num_images: " << d.num_images << std::endl;
    // std::cout << "num_objects: " << d.num_objects << std::endl;
    // std::cout << "num_vertices: " << d.num_vertices << std::endl;
    // std::cout << "num_triangles: " << d.num_triangles << std::endl;
    // std::cout << "num_layers: " << d.num_layers << std::endl;



    int height = resolution[0];
    int width  = resolution[1];
    int depth = d.num_images;

    // std::cout << "height: " << height << std::endl;
    // std::cout << "width: " << width << std::endl;
    // std::cout << "depth: " << depth << std::endl;

    cudaMemset(out, 0, d.num_images*width*height*4*sizeof(float));

    // const at::cuda::OptionalCUDAGuard device_guard(at::device_of(pos));
    RasterizeGLState& s = *stateWrapper.pState;


    int posCount = 4 * d.num_images * d.num_vertices;
    int triCount = 3 * d.num_triangles;

    // std::cout << "posCount: " << posCount << std::endl;
    // std::cout << "triCount: " << triCount << std::endl;

    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    bool changes = false;
    rasterizeResizeBuffers(NVDR_CTX_PARAMS, s, changes, posCount, triCount, width, height, depth);

    // std::cout << "resized"<<  std::endl;

    int peeling_idx = -1;
    const float* posPtr = pos;
    const int32_t* rangesPtr = 0;
    const int32_t* triPtr = tri;
    int vtxPerInstance = d.num_vertices;
    rasterizeRender(NVDR_CTX_PARAMS, s, stream, posPtr, posCount, vtxPerInstance, triPtr, triCount, rangesPtr, width, height, depth, peeling_idx);

    // std::cout << "render" << std::endl;

    s.enableDB = 0;
    float* outputPtr[2];
    outputPtr[0] = out;
    outputPtr[1] = NULL;
    rasterizeCopyResults(NVDR_CTX_PARAMS, s, stream, outputPtr, width, height, depth);
    cudaStreamSynchronize(stream);

    // std::cout << "rasterizeCopyResults" << std::endl;

    releaseGLContext();

//     // Determine number of outputs
//     int num_outputs = s.enableDB ? 2 : 1;

//     // Get output shape.
//     int height = resolution[0];
//     int width  = resolution[1];
//     int depth = d.num_images;
//     int num_layers = d.num_layers;
//     // int depth  = instance_mode ? pos.size(0) : ranges.size(0);
//     NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0];");

//     // Get position and triangle buffer sizes in int32/float32.
//     int posCount = 4 * d.num_vertices;
//     int triCount = 3 * d.num_triangles;

//     // Set the GL context unless manual context.
//     if (stateWrapper.automatic)
//         setGLContext(s.glctx);

//     // Resize all buffers.
//     bool changes = false;
//     cudaStreamSynchronize(stream);

//     rasterizeResizeBuffers(NVDR_CTX_PARAMS, s, changes, posCount, triCount, width, height, depth, num_layers);
//     cudaStreamSynchronize(stream);

//     if (changes)
//     {
// #ifdef _WIN32
//         // Workaround for occasional blank first frame on Windows.
//         releaseGLContext();
//         setGLContext(s.glctx);
// #endif
//     }

//     // Allocate output tensors.
//     float* outputPtr[2];
//     outputPtr[0] = out;
//     outputPtr[1] = s.enableDB ? out_db : NULL;
//     cudaMemset(out, 0, d.num_images*width*height*4*sizeof(float));
//     cudaMemset(out_db, 0, d.num_images*width*height*4*sizeof(float));


//     cudaStreamSynchronize(stream);
//     std::vector<float> projMatrix;
//     projMatrix.resize(16);
//     NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&projMatrix[0], projectionMatrix, 16 * sizeof(int), cudaMemcpyDeviceToHost));
//     cudaStreamSynchronize(stream);



//     // for(int i = 0; i < 16; i++) {
//     //     std::cout << firstPose[i] << " ";
//     // }
//     // std::cout << std::endl;

//     // // Copy input data to GL and render.
//     int peeling_idx = -1;
//     const float* posePtr = pose;
//     const float* posPtr = pos;
//     const int32_t* rangesPtr = ranges; // This is in CPU memory.
//     const int32_t* triPtr = tri;
//     cudaStreamSynchronize(stream);
//     rasterizeRender(NVDR_CTX_PARAMS, s, stream, outputPtr, projMatrix, posePtr, posPtr, posCount, d.num_vertices, triPtr, triCount, rangesPtr, d.num_objects, width, height, depth, peeling_idx);
//     cudaStreamSynchronize(stream);


//     // // Copy rasterized results into CUDA buffers.
//     // cudaStreamSynchronize(stream);
//     // rasterizeCopyResults(NVDR_CTX_PARAMS, s, stream, outputPtr, width, height, depth);
//     // cudaStreamSynchronize(stream);

//     // Done. Release GL context and return.
//     if (stateWrapper.automatic)
//         releaseGLContext();

//     cudaStreamSynchronize(stream);
}
