#include <enums.hpp>
#include <math_constants.h>

__global__ void nugget_cov_kernel(double* dist, double* cov,
                                  const int n, const int nn,
                                  const double nugget)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < n; i += n_threads) // FIXME
    {
        cov[n*i+i] += nugget;
    }
}


void nugget_cov_gpu(double* dist, double* cov,
                    const int n, const int m,
                    double nugget, int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    if (n==m)
        nugget_cov_kernel<<<blocks, n_threads>>>(dist, cov, n, nm, nugget);

    cudaDeviceSynchronize();
}


__global__ void constant_cov_kernel(double* dist, double* cov,
                                    const int nm, const double sigma2)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < nm; i += n_threads)
    {
        cov[i] += sigma2;
    }
}

void constant_cov_gpu(double* dist, double* cov,
                      const int n, const int m,
                      double sigma2, int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    constant_cov_kernel<<<blocks, n_threads>>>(dist, cov, nm, sigma2);

    cudaDeviceSynchronize();
}



__global__ void exponential_cov_kernel(double* dist, double* cov, const int nm,
                                       const double sigma2, const double phi)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < nm; i += n_threads)
    {
        cov[i] += sigma2 * exp( -dist[i] * phi );
    }
}


void exponential_cov_gpu(double* dist, double* cov,
                         const int n, const int m,
                         double sigma2, double phi,
                         int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    exponential_cov_kernel<<<blocks, n_threads>>>(dist, cov, nm, sigma2, phi);

    cudaDeviceSynchronize();
}


__global__ void gaussian_cov_kernel(double* dist, double* cov, const int nm,
                                    const double sigma2, const double phi)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < nm; i += n_threads)
    {
        cov[i] += sigma2 * exp( -0.5 * pow(dist[i] * phi,2) );
    }
}


void gaussian_cov_gpu(double* dist, double* cov,
                      const int n, const int m,
                      double sigma2, double phi,
                      int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    gaussian_cov_kernel<<<blocks, n_threads>>>(dist, cov, nm, sigma2, phi);

    cudaDeviceSynchronize();
}


__global__ void powered_exponential_cov_kernel(double* dist, double* cov,
                                               const int nm, const double sigma2,
                                               const double phi, const double kappa)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < nm; i += n_threads)
    {
        cov[i] += sigma2 * exp( -pow(dist[i] * phi, kappa) );
    }
}


void powered_exponential_cov_gpu(double* dist, double* cov,
                                 const int n, const int m,
                                 double sigma2, double phi,
                                 double kappa, int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    powered_exponential_cov_kernel<<<blocks, n_threads>>>(dist, cov, nm, sigma2, phi, kappa);

    cudaDeviceSynchronize();
}


__global__ void spherical_cov_kernel(double* dist, double* cov, const int nm,
                                     const double sigma2, const double phi)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < nm; i += n_threads)
    {
        cov[i] +=   (dist[i] <= 1.0/phi)
                  ? sigma2 * (1.0 - 1.5*phi*dist[i] + 0.5*pow(phi*dist[i],3.0))
                  : 0;
    }
}


void spherical_cov_gpu(double* dist, double* cov,
                       const int n, const int m,
                       double sigma2, double phi,
                       int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    spherical_cov_kernel<<<blocks, n_threads>>>(dist, cov, nm, sigma2, phi);

    cudaDeviceSynchronize();
}


__global__ void rational_quadratic_cov_kernel(double* dist, double* cov, const int nm,
                                              const double sigma2, const double phi,
                                              const double alpha)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < nm; i += n_threads)
    {
        cov[i] += sigma2 * pow(1 + 0.5 * pow(phi * dist[i],2)/alpha, -alpha);
    }
}


void rational_quadratic_cov_gpu(double* dist, double* cov,
                                const int n, const int m,
                                double sigma2, double phi,
                                double alpha, int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    rational_quadratic_cov_kernel<<<blocks, n_threads>>>(dist, cov, nm, sigma2, phi, alpha);

    cudaDeviceSynchronize();
}


__global__ void periodic_cov_kernel(double* dist, double* cov, const int nm,
                                    const double sigma2, const double phi,
                                    const double gamma)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < nm; i += n_threads)
    {
        cov[i] += sigma2 * exp(-2.0 * pow(phi * sin(CUDART_PI * dist[i] / gamma),2));
    }
}


void periodic_cov_gpu(double* dist, double* cov,
                      const int n, const int m,
                      double sigma2, double phi,
                      double gamma, int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    periodic_cov_kernel<<<blocks, n_threads>>>(dist, cov, nm, sigma2, phi, gamma);

    cudaDeviceSynchronize();
}


__global__ void exp_periodic_cov_kernel(double* dist, double* cov, const int nm,
                                        const double sigma2, const double phi1,
                                        const double gamma, const double phi2)
{
    int n_threads = gridDim.x * blockDim.x;
    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = pos; i < nm; i += n_threads)
    {
        cov[i] += sigma2 * exp(-2.0 * pow(phi1 * sin(CUDART_PI * dist[i] / gamma),2)
                                -0.5 * pow(phi2 * dist[i],2));
    }
}


void exp_periodic_cov_gpu(double* dist, double* cov,
                          const int n, const int m,
                          double sigma2, double phi1,
                          double gamma, double phi2,
                          int n_threads)
{
    int nm = n*m;
    int blocks = (n+n_threads-1)/n_threads;

    exp_periodic_cov_kernel<<<blocks, n_threads>>>(dist, cov, nm, sigma2, phi1, gamma, phi2);

    cudaDeviceSynchronize();
}






