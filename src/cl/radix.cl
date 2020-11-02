#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


__kernel void radix(__global unsigned int* as)
{
    // TODO
}


__kernel void prefix_sum(__global unsigned int *as)
{
    int i = get_global_id(0);
    int local_i = get_local_id(0);
    for (int pow = 0; )




}