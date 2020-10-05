// TODO
#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include <cmath>
#endif

#define WORK_GROUP_SIZE 256
__kernel void sum(__global const unsigned int* as,
                  __global unsigned int* res)
{
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local int local_as[WORK_GROUP_SIZE];
    local_as[localId] = as[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);// <- дождались всех

    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * localId < nvalues) {
            int a = local_as[localId];
            int b = local_as[localId + nvalues / 2];
            local_as[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        atomic_add(res, local_as[0]);
    }
}
