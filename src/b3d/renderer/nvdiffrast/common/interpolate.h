// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

//------------------------------------------------------------------------
// Constants and helpers.

#define IP_FWD_MAX_KERNEL_BLOCK_WIDTH   8
#define IP_FWD_MAX_KERNEL_BLOCK_HEIGHT  8
#define IP_GRAD_MAX_KERNEL_BLOCK_WIDTH  8
#define IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT 8
#define IP_MAX_DIFF_ATTRS               32

//------------------------------------------------------------------------
// CUDA kernel params.

struct InterpolateKernelParams
{
    const int*      triangles;                            // Incoming triangle buffer.
    const int*      faces;                            // Incoming triangle buffer.
    const float*      uvs;                            // Incoming triangle buffer.
    const float*    attributes;                           // Incoming attribute buffer.
    float*          out;                            // Outgoing interpolated attributes.

    int             numTriangles;                   // Number of triangles.
    int             numVertices;                    // Number of vertices.
    int             numAttr;                        // Number of total vertex attributes.

    int             width;                          // Image width.
    int             height;                         // Image height.
    int             depth;                          // Minibatch size.
};

//------------------------------------------------------------------------
