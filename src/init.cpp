#ifdef USE_GPU
#include <magma.h>
#endif

// [[Rcpp::export]]
void init(bool verbose)
{
#ifdef USE_GPU
    magma_init();
    if( cublasInit() != CUBLAS_STATUS_SUCCESS ) 
    {
        magma_finalize();
        Rcpp::stop("cublasInit failed");
    }
    if (verbose) {
        magma_print_devices();

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        Rcpp::Rcout << free_mem  / (1024. * 1024) << "MB of " 
                    << total_mem / (1024. * 1024) << "MB free.\n";
    }
#endif
}

// [[Rcpp::export]]
void finalize()
{
#ifdef USE_GPU
    //cublasShutdown();
    //magma_finalize();
#endif
}

