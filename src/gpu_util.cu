#include <math_functions.h>

__global__ void symmatu_kernel(double* M, const int n, const int n_tri)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < n_tri; i += n_threads)
    {
        double tmp = floor( (-1.0+sqrt(8.0*i+1.0))/2.0 );
        int c = (int)tmp;  
        int r = i - c*(c+1)/2;

        c+=1;

        M[n*r+c] = M[n*c+r];
    }
}


void symmat(double* M, const int n, char type, const int n_threads) 
{
    int n_tri = n*(n-1)/2;
    int blocks = (n+n_threads-1)/n_threads;
    
    if (type == 'U' || type == 'u')
    {
        symmatu_kernel<<<blocks, n_threads>>>(M, n, n_tri);
    } 
    else if (type == 'L' || type == 'l')
    {
        //RT_ASSERT(false, "symmatl not currently supported.");
        //symmatl_kernel<<<blocks, n_threads>>>(m, n, nn);
    }
    else
    {
        //RT_ASSERT(false, "Unknown type, must be U or L");
    }    
    
    cudaDeviceSynchronize();
}


__global__ void trimatu_kernel(double* M, const int n, const int n_tri, const double val)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < n_tri; i += n_threads)
    {
        double tmp = floor( (-1.0+sqrt(8.0*i+1.0))/2.0 );
        int c = (int)tmp;  
        int r = i - c*(c+1)/2;

        c+=1;

        M[n*r+c] = val;
    }
}


void trimat(double* M, const int n, const char type, const double val, const int n_threads) 
{
    int n_tri = n*(n-1)/2;
    int blocks = (n+n_threads-1)/n_threads;
    
    if (type == 'U' || type == 'u')
    {
        trimatu_kernel<<<blocks, n_threads>>>(M, n, n_tri, val);
    } 
    else if (type == 'L' || type == 'l')
    {
        //RT_ASSERT(false, "symmatl not currently supported.");
        //symmatl_kernel<<<blocks, n_threads>>>(m, n, nn);
    }
    else
    {
        //RT_ASSERT(false, "Unknown type, must be U or L");
    }    
    
    cudaDeviceSynchronize();
}
