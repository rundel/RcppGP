#ifndef GPU_MAT_HPP
#define GPU_MAT_HPP

#include <RcppArmadillo.h>
#include <boost/utility.hpp>
#include "assert.hpp"


#ifdef USE_GPU

#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>

#include <magma.h>

#include "gpu_util.hpp"


static const char* cudasolver_error(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

        case CUSOLVER_STATUS_MAPPING_ERROR:
            return "CUSOLVER_STATUS_MAPPING_ERROR";

        case CUSOLVER_STATUS_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_NOT_SUPPORTED";

        case CUSOLVER_STATUS_ZERO_PIVOT:
            return "CUSOLVER_STATUS_ZERO_PIVOT";

        case CUSOLVER_STATUS_INVALID_LICENSE:
            return "CUSOLVER_STATUS_INVALID_LICENSE";
    }

    return "<unknown>";
}



class gpu_mat : boost::noncopyable 
{
private:
    double *mat;
    bool allocated;

    void alloc_mat()
    {
        cudaError s = cudaMalloc((void**)&mat, n_rows*n_cols*sizeof(double));
        RT_ASSERT(s==cudaSuccess, "CUDA allocation failed");

        allocated = true;
    }

public:
    int const n_rows;
    int const n_cols;

    gpu_mat(arma::mat const& m)
      : n_rows(m.n_rows),
        n_cols(m.n_cols),
        allocated(false)
    {
        alloc_mat();

        cublasSetMatrix(n_rows, n_cols, sizeof(double), m.memptr(), n_rows, mat, n_rows);
    }

    gpu_mat(int r, int c)
      : n_rows(r),
        n_cols(c),
        allocated(false)
    {
        alloc_mat();
    }

    gpu_mat(int r, int c, double init)
      : n_rows(r),
        n_cols(c),
        allocated(false)
    {
        alloc_mat();

        cudaMemset(mat, init, r*c*sizeof(double));
    }

    gpu_mat(double *m, int r, int c)
      : mat(m),
        n_rows(r),
        n_cols(c),
        allocated(true)
    { }


    ~gpu_mat()
    {
        cudaFree(mat);
    }

    arma::mat get_mat()
    {
        RT_ASSERT(allocated,"CUDA matrix not allocated");

        arma::mat m(n_rows, n_cols);

        cublasGetMatrix(n_rows, n_cols, sizeof(double), mat, n_rows, m.memptr(), n_rows);

        return m;
    }

    double const* get_const_ptr() const
    {
        RT_ASSERT(allocated,"CUDA matrix not allocated");

        return mat;
    }

    double* get_ptr()
    {
        RT_ASSERT(allocated,"CUDA matrix not allocated");

        return mat;
    }

    double* get_gpu_mat()
    {
        RT_ASSERT(allocated,"CUDA matrix not allocated");

        double* tmp = mat;

        mat = NULL;
        allocated = false;

        return tmp;
    }

    void chol(char uplo)
    {
        RT_ASSERT(n_rows==n_cols, "Matrix must be square.");
        RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");


        magma_uplo_t magma_uplo = uplo=='U' ? MagmaUpper : MagmaLower;
        int info;

        magma_dpotrf_gpu(magma_uplo, n_rows, mat, n_rows, &info);
        RT_ASSERT(info==0, "Cholesky failed.");
/*
        cusolverDnHandle_t handle;
        cusolverDnCreate(&handle);

        cusolverStatus_t s;

        cublasFillMode_t cuda_uplo = uplo=='U' ? CUBLAS_FILL_MODE_UPPER
                                               : CUBLAS_FILL_MODE_LOWER;
        int lwork;
        s = cusolverDnDpotrf_bufferSize(handle, cuda_uplo, n_rows,
                                        mat, n_rows, &lwork);
        RT_ASSERT(s==CUSOLVER_STATUS_SUCCESS, "Buffer size calc failed.");

        double *work;
        cudaError cs = cudaMalloc((void**)&work, lwork*sizeof(double));
        RT_ASSERT(cs == cudaSuccess, "CUDA allocation failed");

        int info;
        s = cusolverDnDpotrf(handle, cuda_uplo, n_rows,
                             mat, n_rows, 
                             work, lwork, &info);

        cudaFree(work);

        Rcpp::Rcout << "info: " << info 
                    << " lwork: " << lwork 
                    << " s: " << s 
                    << " (" << cudasolver_error(s) << ")\n";

        RT_ASSERT(info==0 && s==CUSOLVER_STATUS_SUCCESS, "Cholesky failed.");
*/        
        trimat(mat, n_rows, uplo, 0.0, 64);
    }

    void inv_chol(char uplo)
    {
        RT_ASSERT(n_rows == n_cols, "Matrix must be square.");
        RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

        int info;

        magma_uplo_t magma_uplo = uplo=='U' ? MagmaUpper : MagmaLower;

        magma_dpotri_gpu(magma_uplo, n_rows, mat, n_rows, &info);

        RT_ASSERT(info==0, "Inverse (dpotri) failed.");

        symmat(mat, n_rows, uplo, 64);
    }

    void inv_sympd()
    {
        chol('U');
        inv_chol('U');
    }
};

#else

// If no GPU default to arma::mat and related operations

class gpu_mat : boost::noncopyable 
{
private:
    arma::mat mat;
    bool allocated;

public:
    int const n_rows;
    int const n_cols;

    gpu_mat(arma::mat const& m)
      : n_rows(m.n_rows),
        n_cols(m.n_cols),
        mat(m)
    { }

    gpu_mat(int r, int c)
      : n_rows(r),
        n_cols(c)
    {
        mat = arma::mat(r,c);
    }

    gpu_mat(int r, int c, double init)
      : n_rows(r),
        n_cols(c)
    {
        mat = arma::mat(r,c);
        mat.fill(init);
    }

    gpu_mat(double *m, int r, int c)
      : n_rows(r),
        n_cols(c)
    { 
        mat = arma::mat(m, r, c, true);
    }


    ~gpu_mat()
    { }

    arma::mat get_mat()
    {
        return mat;
    }

    double const* get_const_ptr() const
    {
        return mat.memptr();
    }

    double* get_ptr()
    {
        return mat.memptr();
    }


    void chol(char uplo)
    {
        RT_ASSERT(n_rows==n_cols, "Matrix must be square.");
        RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

        mat = uplo=='U' ? arma::chol(mat, "upper") : arma::chol(mat, "lower");
    }

    void inv_chol(char uplo)
    {
        RT_ASSERT(n_rows == n_cols, "Matrix must be square.");
        RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

        mat = arma::inv(mat);
    }

    void inv_sympd()
    {
        RT_ASSERT(n_rows == n_cols, "Matrix must be square.");

        mat = arma::inv_sympd(mat);
    }
};

#endif

#endif