__kernel void divisors(__global int *src, __global const int *flags, __global int *res_p, __global int* res_m)
{
    int gid = get_global_id(0);
    int i = src[gid];
    int n = flags[0];
    int len = flags[2];
    int flag = flags[1];
    if(n % i == 0){
        res_p[gid] = i;
        if(n != 1){
            res_p[len-1-gid] = n/i;
        };
        if(flag == 1){
            if(n != 1){
                res_m[len-1-gid] = -n/i;
            };
            res_m[gid] = -i;
        };
    };
}