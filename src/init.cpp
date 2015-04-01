#include <RcppArmadillo.h>

#include "cuda_util.hpp"
#include "assert.hpp"

#ifdef USE_GPU
#include <magma.h>
#include <cuda_runtime.h>
#endif

// [[Rcpp::export]]
void check_gpu_mem()
{
#ifdef USE_GPU
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    Rcpp::Rcout << free_mem  / (1024. * 1024) << "MB of " 
                << total_mem / (1024. * 1024) << "MB free.\n";
#else
    Rcpp::Rcout << "No GPU detected!\n";
#endif
}

// [[Rcpp::export]]
void init(bool verbose = false)
{
#ifdef USE_GPU
    magma_init();
    cublasStatus_t s = cublasInit();
    if( s != CUBLAS_STATUS_SUCCESS ) 
    {
        magma_finalize();
        RT_ASSERT(false, cublas_error(s));
    }
    if (verbose) {
        //magma_print_devices();
        magma_print_environment();

        check_gpu_mem();
    }
#endif
}

// [[Rcpp::export]]
void finalize()
{
#ifdef USE_GPU
    //cublasShutdown();
    magma_finalize();
    cudaDeviceReset();
#endif
}

// [[Rcpp::export]]
void reset()
{
#ifdef USE_GPU
    cudaDeviceReset();
#endif
}
