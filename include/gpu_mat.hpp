#ifndef GPU_MAT_HPP
#define GPU_MAT_HPP

#ifdef USE_GPU

#include <RcppArmadillo.h>

#include <boost/utility.hpp>

#include <cuda_runtime.h>
#include <cublas.h>

#include "assert.hpp"


class gpu_mat : boost::noncopyable 
{
public:
    int const n_rows;
    int const n_cols;
    double *mat;
    bool allocated;

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

    void alloc_mat()
    {
        cudaError s = cudaMalloc((void**)&mat, n_rows*n_cols*sizeof(double));

        if (s != cudaSuccess) 
        {
            RT_ASSERT(false,"CUDA allocation failed");
        }
        else
        {
            allocated = true;
        }
    }

    arma::mat get_mat()
    {
        if (!allocated)
            RT_ASSERT(false,"CUDA matrix not allocated");

        arma::mat m(n_rows, n_cols);

        cublasGetMatrix(n_rows, n_cols, sizeof(double), mat, n_rows, m.memptr(), n_rows);

        return m;
    }

    double* get_ptr()
    {
        if (!allocated)
            RT_ASSERT(false,"CUDA matrix not allocated");

        double* tmp = mat;
        
        mat = NULL;
        allocated = false;

        return tmp;
    }
};


void chol(gpu_mat& M, char uplo);
void inv_chol(gpu_mat& chol, char uplo);
void inv_sympd(gpu_mat& M);


#endif
#endif