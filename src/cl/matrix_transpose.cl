#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16

__kernel void matrix_transpose(__global float *a,
                               __global float *at,
                               unsigned int m,
                               unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * k + i];//coalesced чтение

    barrier(CLK_LOCAL_MEM_FENCE);

    /*float tmp = tile[local_j][local_i];
    tile[local_j][local_i] = tile[local_i][local_j];
    tile[local_i][local_j] = tmp;*/

    barrier(CLK_LOCAL_MEM_FENCE);

    int y = i - i % TILE_SIZE;
    int x = j - j % TILE_SIZE;

    at[y * m + local_j * m + x + local_i] = tile[local_i][local_j];

}