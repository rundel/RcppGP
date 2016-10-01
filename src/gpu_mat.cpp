#include <boost/assert.hpp>
#include "gpu_mat.hpp"


int gpu_mat::get_n_rows() const
{
    return n_rows;
}

int gpu_mat::get_n_cols() const
{
    return n_cols;
}

#ifdef USE_GPU

#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include <curand.h>
//#include <cusolverDn.h>
//#include <magma.h>
//#include "gpu_util.hpp"


void gpu_mat::alloc_mat()
{
    cudaError s = cudaMalloc((void**)&mat, n_rows*n_cols*sizeof(double));
    BOOST_ASSERT_MSG(s==cudaSuccess, "CUDA allocation failed");

    allocated = true;
}

gpu_mat::gpu_mat(gpu_mat&& rhs)
{
    swap(rhs);
}

gpu_mat& gpu_mat::operator=(gpu_mat&& rhs)
{
    swap(rhs);

    return *this;
}


gpu_mat::gpu_mat()
  : allocated(false)
{ }

gpu_mat::gpu_mat(arma::mat const& m)
  : n_rows(m.n_rows),
    n_cols(m.n_cols),
    allocated(false)
{
    alloc_mat();

    cublasSetMatrix(n_rows, n_cols, sizeof(double), m.memptr(), n_rows, mat, n_rows);
}

gpu_mat::gpu_mat(int r, int c)
  : n_rows(r),
    n_cols(c),
    allocated(false)
{
    alloc_mat();
}

gpu_mat::gpu_mat(int r, int c, double init)
  : n_rows(r),
    n_cols(c),
    allocated(false)
{
    alloc_mat();

    cudaMemset(mat, init, r*c*sizeof(double));
}

gpu_mat::gpu_mat(double *m, int r, int c)
  : mat(m),
    n_rows(r),
    n_cols(c),
    allocated(true)
{ }


gpu_mat::~gpu_mat()
{
    release();
}

void gpu_mat::release()
{
    if (allocated)
        cudaFree(mat);

    allocated = FALSE;
}

bool gpu_mat::is_allocated() const
{
    return allocated;
}

arma::mat gpu_mat::get_mat() const
{
    BOOST_ASSERT_MSG(allocated,"CUDA matrix not allocated");

    arma::mat m(n_rows, n_cols);

    cublasGetMatrix(n_rows, n_cols, sizeof(double), mat, n_rows, m.memptr(), n_rows);

    return m;
}

double* gpu_mat::get_ptr()
{
    BOOST_ASSERT_MSG(allocated,"CUDA matrix not allocated");

    return mat;
}

double const* gpu_mat::get_const_ptr() const
{
    BOOST_ASSERT_MSG(allocated,"CUDA matrix not allocated");

    return mat;
}

void gpu_mat::assign(gpu_mat& g)
{
    BOOST_ASSERT_MSG(!allocated,"CUDA matrix being assigned to must not be allocated");

    allocated = TRUE;
    n_rows = g.n_rows;
    n_cols = g.n_cols;

    // Takes ownership of g's memory
    mat = g.mat;
    g.mat = NULL;
    g.allocated = FALSE;
}

void gpu_mat::swap(gpu_mat &g)
{
    BOOST_ASSERT_MSG(allocated & g.allocated,"Both CUDA matrices must be allocated");

    std::swap(mat, g.mat);
    std::swap(n_rows, g.n_rows);
    std::swap(n_cols, g.n_cols);
    std::swap(allocated, g.allocated);
}

gpu_mat gpu_mat::make_copy() const
{
    // Caller takes ownership of memory

    BOOST_ASSERT_MSG(allocated,"CUDA matrix not allocated");

    gpu_mat new_mat(n_rows, n_cols);

    cudaMemcpy(new_mat.get_ptr(), get_const_ptr(), n_rows*n_cols*sizeof(double), cudaMemcpyDeviceToDevice);

    //Rcpp::Rcout << "Making copy (" << n_rows << ", " << n_cols 
    //            << ") => ("        << new_mat.n_rows << ", " << new_mat.n_cols << ")\n";

    return new_mat;
}


#else

//////////////////////////////////////////////////////////////////////////////////////
//
// CPU implementation failover if no GPU
//
//////////////////////////////////////////////////////////////////////////////////////
void gpu_mat::alloc_mat()
{ }

gpu_mat::gpu_mat(arma::mat const& m)
  : n_rows(m.n_rows),
    n_cols(m.n_cols),
    mat(m)
{ }

gpu_mat::gpu_mat(int r, int c)
  : n_rows(r),
    n_cols(c)
{
    mat = arma::mat(r,c);
}

gpu_mat::gpu_mat(int r, int c, double init)
  : n_rows(r),
    n_cols(c)
{
    mat = arma::mat(r,c);
    mat.fill(init);
}

gpu_mat::gpu_mat(double *m, int r, int c)
  : n_rows(r),
    n_cols(c)
{
    mat = arma::mat(m, r, c, true);
}


gpu_mat::~gpu_mat()
{ }
_
double* gpu_mat::get_gpu_mat()
{
    return mat.memptr();
}

double* gpu_mat::get_copy()
{
    arma::mat new_mat(mat);

    return new_mat.memptr();
}

arma::mat gpu_mat::get_mat()
{
    return mat;
}

double const* gpu_mat::get_const_ptr() const
{
    return mat.memptr();
}

double* gpu_mat::get_ptr()
{
    return mat.memptr();
}


void gpu_mat::QR_Q()
{
    arma::mat Q,R;
    qr_econ(Q,R,mat);

    mat = Q;
}


void gpu_mat::chol(char uplo)
{
    BOOST_ASSERT_MSG(n_rows==n_cols, "Matrix must be square.");
    BOOST_ASSERT_MSG(uplo=='U' || uplo=='L', "uplo must be U or L.");

    mat = uplo=='U' ? arma::chol(mat, "upper") : arma::chol(mat, "lower");
}

void gpu_mat::inv_chol(char uplo)
{
    BOOST_ASSERT_MSG(n_rows == n_cols, "Matrix must be square.");
    BOOST_ASSERT_MSG(uplo=='U' || uplo=='L', "uplo must be U or L.");

    mat = arma::inv(mat);
}

void gpu_mat::inv_sympd()
{
    BOOST_ASSERT_MSG(n_rows == n_cols, "Matrix must be square.");

    mat = arma::inv_sympd(mat);
}


#endif
