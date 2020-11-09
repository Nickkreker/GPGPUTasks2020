#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void merge(__global const float * as,
                    __global float * bs,
                    int n, int k)
{
    int i = get_global_id(0);
    int local_i = i % (2 * k);

    //i диагональ
    int a_begin = i / (2 * k) * (2 * k);
    int b_begin = a_begin + k;
    int diag_num = local_i;

    int t1 = min(local_i, k) - 1;
    int t2 = max(0, local_i - k);

    int left = -1;
    int right = min(diag_num, 2 * k - diag_num);

    int start_in_b = b_begin + t1;
    int start_in_a = a_begin + t2;

    while(left < right - 1)
    {
        int c = (left + right) / 2;
        if(as[start_in_a + c] <= as[start_in_b - c])
        {
            left = c;
        }
        else
        {
            right = c;
        }
    }

    if(left + 1 + t2 == k && local_i >= k)
    {
        bs[i] = as[start_in_b + t2 - t1];
    }
    else
    {
        if((left + 1 == 0) && local_i >= k)
        {
            bs[i] = as[start_in_a];
        }
        else
        {
            bs[i] = min(as[start_in_b - left], as[start_in_a + left + 1]);
        }
    }
}