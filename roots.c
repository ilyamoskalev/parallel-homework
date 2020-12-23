#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void roots(__global float *t, __global float *s, __global const int *koeffs, __global int *deg, __global int *res_t,  __global int *res_s, __global int *iter)
{
    int gid = get_global_id(0);
    double buf = 0.0;
    for(int i = iter[0]*gid; i < iter[0]*gid+iter[0]; i++){
        int int_t = t[i];
        int int_s = s[i];
        int degre = deg[0]-1;
        for (int j = degre; j >= 0; j--){
            double x = (double) int_t / (double) int_s;
            buf += (pow((double) x, (double) j)*koeffs[degre-j]);
        };
    
        if(fabs(buf-0.0) < 0.00001){
            res_t[i] = int_t;
            res_s[i] = int_s;
        }  
        buf = 0.0;
    }
    
}