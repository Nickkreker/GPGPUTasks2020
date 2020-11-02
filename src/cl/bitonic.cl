#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void bitonic_loc(__global float* as, int n, int K, int J)
{
    int i = get_global_id(0);
    int local_i = get_local_id(0);

    __local float local_as[2 * WORK_GROUP_SIZE];

    if (2 * i + 1 < n)
    {
        local_as[2 * local_i] = as[2 * i];
        local_as[2 * local_i + 1] = as[2 * i + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = J; j > 0; j /= 2)
    {
        int wire_begin, wire_end;//loc_id, box_size, local_block,
        wire_begin = local_i % j + (local_i / j) * 2 * j;

        if (j != K)
            wire_end = wire_begin + j;
        else
            wire_end = (local_i / j) * 2 * j - local_i % j  + 2 * j - 1;


        if (wire_end < 2 * 128 && local_as[wire_begin] > local_as[wire_end])
        {
            float t = local_as[wire_begin];
            local_as[wire_begin] = local_as[wire_end];
            local_as[wire_end] = t;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (2 * i + 1 < n)
    {
        as[2 * i] = local_as[2 * local_i];
        as[2 * i + 1] = local_as[2 * local_i + 1];
    }
}


__kernel void bitonic(__global float* as, int n, int K, int J)
{
    // TODO
    int i = get_global_id(0);
    int local_i = get_local_id(0);

    if (J <= WORK_GROUP_SIZE) {
        bitonic_loc(as, n, K, J);
    }
    else
    {
        int wire_begin, wire_end;//loc_id, box_size, local_block,
        wire_begin = i % J + (i / J) * 2 * J;

        if (J != K)
            wire_end = wire_begin + J;
        else
            wire_end = (i / J) * 2 * J - i % J  + 2 * J - 1;


        if (wire_end < n && as[wire_begin] > as[wire_end])
        {
            float t = as[wire_begin];
            as[wire_begin] = as[wire_end];
            as[wire_end] = t;
        }
    }
}