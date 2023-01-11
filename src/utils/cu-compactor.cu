
/*
 * cuCompactor.cu
 *
 *  Created on: 21/mag/2015
 *      Author: knotman
 *
 * Modified on January 7, 2023 by James Perlman
 * Please see LICENSES/knotman90_cuStreamComp.md for license information
 * 
 * original code from https://github.com/knotman90/cuStreamComp
 */

#pragma once

#include <thrust/scan.h>
#include <thrust/device_vector.h>

#include "cu-compactor.cuh"

NRC_NAMESPACE_BEGIN

#define warpSize (32)
#define FULL_MASK 0xffffffff

__host__ __device__ int divup(int x, int y)
{
    return x / y + (x % y ? 1 : 0);
}

__device__ __inline__ int pow2i(int e)
{
    return 1 << e;
}

// predicate is an array of bools
__global__ void computeBlockCounts(
    int n_elements,
    const bool* __restrict__ d_predicate,
    int *d_BlockCounts
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx >= n_elements) return;

    int pred = d_predicate[idx];
    int BC = __syncthreads_count(pred);

    if (threadIdx.x == 0)
    {
        d_BlockCounts[blockIdx.x] = BC; // BC will contain the number of valid elements in all threads of this thread block
    }
}

__global__ void compactK(
    const int n_elements,
    const bool* __restrict__ d_predicate,
    const int* __restrict__ d_BlocksOffset,
    int* __restrict__ d_output
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int warpTotals[];

    if (idx >= n_elements) return;

    int pred = d_predicate[idx];
    int w_i = threadIdx.x / warpSize; // warp index
    int w_l = idx % warpSize;         // thread index within a warp

    // compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
    int t_m = FULL_MASK >> (warpSize - w_l); // thread mask
    int b = __ballot_sync(FULL_MASK, pred) & t_m;
    int t_u = __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

    // last thread in warp computes total valid counts for the warp
    if (w_l == warpSize - 1) {
        warpTotals[w_i] = t_u + pred;
    }

    // need all warps in thread block to fill in warpTotals before proceeding
    __syncthreads();

    // first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
    int numWarps = blockDim.x / warpSize;
    unsigned int numWarpsMask = FULL_MASK >> (warpSize - numWarps);
    if (w_i == 0 && w_l < numWarps)
    {
        int w_i_u = 0;
        for (int j = 0; j <= 5; j++)
        { // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
            int b_j = __ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j));
            w_i_u += (__popc(b_j & t_m)) << j;
            // printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
        }
        warpTotals[w_l] = w_i_u;
    }

    // need all warps in thread block to wait until prefix sum is calculated in warpTotals
    __syncthreads();

    // if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
    if (pred) {
        d_output[t_u + warpTotals[w_i] + d_BlocksOffset[blockIdx.x]] = idx;
    }
}

template <class T>
__global__ void printArray_GPU(T *hd_data, int size, int newline)
{
    int w = 0;
    for (int i = 0; i < size; i++)
    {
        if (i % newline == 0)
        {
            printf("\n%i -> ", w);
            w++;
        }
        printf("%i ", hd_data[i]);
    }
    printf("\n");
}

int generate_compaction_indices(
    const cudaStream_t& stream,
    const int n_elements,
    const int blockSize,
    const bool* d_predicate,
    int* d_output
) {
    int numBlocks = divup(n_elements, blockSize);
    int *d_BlocksCountAndOffset;

    // TODO: these cudaMallocs can be moved out to avoid unnecessary allocations
    CUDA_CHECK_THROW(cudaMallocAsync(&d_BlocksCountAndOffset, 2 * numBlocks * sizeof(int), stream));

    thrust::device_ptr<int> bCount_ptr(d_BlocksCountAndOffset + 0 * numBlocks);
    thrust::device_ptr<int> bOffset_ptr(d_BlocksCountAndOffset + 1 * numBlocks);

    // phase 1: count number of valid elements in each thread block
    computeBlockCounts<<<numBlocks, blockSize, 0, stream>>>(
        n_elements,
        d_predicate,
        bCount_ptr.get()
    );

    // phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
    thrust::exclusive_scan(
        thrust::cuda::par_nosync.on(stream),
        bCount_ptr,
        bCount_ptr + numBlocks,
        bOffset_ptr
    );

    // phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
    compactK<<<numBlocks, blockSize, sizeof(int) * (blockSize / warpSize), stream>>>(
        n_elements,
        d_predicate,
        bOffset_ptr.get(),
        d_output
    );

    // determine number of elements in the compacted list

    // copy last element of thrustPtr_bOffset to host
    int bCount, bOffset;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&bOffset, bOffset_ptr.get() + numBlocks - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(&bCount, bCount_ptr.get() + numBlocks - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream))
    // compact_length = thrustPrt_bOffset[numBlocks - 1] + thrustPrt_bCount[numBlocks - 1];
    cudaFree(d_BlocksCountAndOffset);

    return bOffset + bCount;
}

NRC_NAMESPACE_END
