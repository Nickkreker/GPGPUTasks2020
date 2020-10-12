#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float *a,
                                    __global const float *b,
                                    __global float *c,
                                    int M,
                                    int K,
                                    int N)
{
    // TODO
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    __local float res[TILE_SIZE][TILE_SIZE + 1];
    res[local_j][local_i] = 0;

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK)
    {
        tileA[local_j][local_i] = a[j * K + (tileK * TILE_SIZE + local_i)];
        tileB[local_j][local_i] = b[(tileK * TILE_SIZE + local_j) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            res[local_j][local_i] += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * N + i] = res[local_j][local_i];
}