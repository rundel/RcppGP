#include <magma.h>
#include <RcppArmadillo.h>

#include "init.hpp"
#include "assert.hpp"

void init(bool verbose)
{
    magma_init();
    if( cublasInit() != CUBLAS_STATUS_SUCCESS ) 
    {
        magma_finalize();
        RT_ASSERT(false, "cublasInit failed");
    }
    if (verbose) {
        magma_print_devices();

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        Rcpp::Rcout << free_mem  / (1024. * 1024) << "MB of " 
                    << total_mem / (1024. * 1024) << "MB free.\n";
    }
}

void finalize()
{
    //cublasShutdown();
    //magma_finalize();
}