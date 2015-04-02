#include "assert.hpp"
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
#include <curand.h>
//#include <cusolverDn.h>
#include <magma.h>

#include "gpu_util.hpp"


void gpu_mat::alloc_mat()
{
    cudaError s = cudaMalloc((void**)&mat, n_rows*n_cols*sizeof(double));
    RT_ASSERT(s==cudaSuccess, "CUDA allocation failed");

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
    cudaFree(mat);
}

bool gpu_mat::is_allocated() const
{
    return allocated;
}

arma::mat gpu_mat::get_mat()
{
    RT_ASSERT(allocated,"CUDA matrix not allocated");

    arma::mat m(n_rows, n_cols);

    cublasGetMatrix(n_rows, n_cols, sizeof(double), mat, n_rows, m.memptr(), n_rows);

    return m;
}

double* gpu_mat::get_ptr()
{
    RT_ASSERT(allocated,"CUDA matrix not allocated");

    return mat;
}

double const* gpu_mat::get_const_ptr() const
{
    RT_ASSERT(allocated,"CUDA matrix not allocated");

    return mat;
}

void gpu_mat::assign(gpu_mat& g)
{
    RT_ASSERT(!allocated,"CUDA matrix being assigned to must not be allocated");

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
    RT_ASSERT(allocated & g.allocated,"Both CUDA matrices must be allocated");

    std::swap(mat, g.mat);
    std::swap(n_rows, g.n_rows);
    std::swap(n_cols, g.n_cols);
}

gpu_mat gpu_mat::make_copy()
{
    // Caller takes ownership of memory

    RT_ASSERT(allocated,"CUDA matrix not allocated");

    gpu_mat new_mat(n_rows, n_cols);

    cudaMemcpy(new_mat.get_ptr(), get_const_ptr(), n_rows*n_cols*sizeof(double), cudaMemcpyDeviceToDevice);

    //Rcpp::Rcout << "Making copy (" << n_rows << ", " << n_cols 
    //            << ") => ("        << new_mat.n_rows << ", " << new_mat.n_cols << ")\n";

    return new_mat;
}

void gpu_mat::fill_rnorm(double mu, double sigma)
{
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    curandStatus_t rs = curandGenerateNormalDouble(prng, mat, n_rows*n_cols, mu, sigma);
    RT_ASSERT(rs == CURAND_STATUS_SUCCESS, "Normal sampling failed.");

    curandDestroyGenerator(prng);
}

void gpu_mat::mat_mult(gpu_mat const& Y, char op_X, char op_Y, bool swap_order)
{
    RT_ASSERT(op_X=='N' || op_X=='T', "op_X must be N or T.");
    RT_ASSERT(op_Y=='N' || op_Y=='T', "op_Y must be N or T.");

    cublasOperation_t cuda_op_A, cuda_op_B;
    int m,n,k, ldA, ldB;
    double const* A;
    double const* B;

    if (swap_order) // R = Y * X
    {
        cuda_op_A = op_Y=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
        cuda_op_B = op_X=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;

        m = (op_Y=='N') ? Y.get_n_rows() : Y.get_n_cols();
        n = (op_X=='N') ? n_cols : n_rows;
        k = (op_X=='N') ? n_rows : n_cols;
        int k2 = (op_Y=='N') ? Y.get_n_cols() : Y.get_n_rows();
        RT_ASSERT(k2==k, "dim mismatch for matrix multiplication.");

        A = Y.get_const_ptr();
        B = get_const_ptr();

        ldA = Y.get_n_rows();
        ldB = n_rows;
    }
    else // R = X * Y
    {
        cuda_op_A = op_X=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
        cuda_op_B = op_Y=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;

        m = (op_X=='N') ? n_rows : n_cols;
        n = (op_Y=='N') ? Y.get_n_cols() : Y.get_n_rows();
        k = (op_X=='N') ? n_cols : n_rows;
        int k2 = (op_Y=='N') ? Y.get_n_rows() : Y.get_n_cols();
        RT_ASSERT(k2==k, "dim mismatch for matrix multiplication.");

        A = get_const_ptr();
        B = Y.get_const_ptr();

        ldA = n_rows;
        ldB = Y.get_n_rows();
    }

    //if (swap_order)
    //    Rcpp::Rcout << "Swapped: ";
    //else
    //    Rcpp::Rcout << "       : ";

    //Rcpp::Rcout << m << " x " << k << " times "
    //            << k << " x " << n << "\n";

    gpu_mat result(m,n);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double one = 1.0, zero = 0.0;
    cublasStatus_t bs = cublasDgemm(handle, cuda_op_A, cuda_op_B,
                                    m, n, k, &one,
                                    A, ldA,
                                    B, ldB,
                                    &zero,
                                    result.get_ptr(), m
                                   );
    cublasDestroy(handle);
    RT_ASSERT(bs == CUBLAS_STATUS_SUCCESS, "Matrix multiply failed.");

    swap(result); 
}

void gpu_mat::low_rank_sympd(arma::vec& C, int rank, int over_samp, int qr_iter)
{
    // For symmetric matrices use eigen decomposition of Qt A Q

    RT_ASSERT(n_rows==n_cols,"Matrix must be square");

    gpu_mat A = this->make_copy();
    rand_proj(A, rank, over_samp, qr_iter);

    gpu_mat Q(make_copy());
    A.mat_mult(Q, 'N', 'N', false);
    mat_mult(A,'T','N',false);          // this = Qt * A * Q

    eig_sym(C);                         // U C Ut = this
    mat_mult(Q, 'N', 'N', true);        // U = Q * U;
}

void gpu_mat::eig_sym(arma::vec& vals)
{
    RT_ASSERT(n_rows==n_cols,"Matrix must be square");

    vals.resize(n_rows);

    // query for workspace sizes
    double lwork;
    int liwork, info;

    magma_dsyevd_gpu( MagmaVec, MagmaLower,
                      n_rows, NULL, n_rows, 
                      NULL, NULL, n_rows,
                      &lwork,  -1,
                      &liwork, -1,
                      &info );

    // Calculate eigen decomposition
    arma::ivec iwork(liwork);
    arma::mat  h_R(n_rows,n_rows);
    arma::vec  h_work((int) lwork);

    magma_dsyevd_gpu( MagmaVec, MagmaLower,
                      n_rows, mat, n_rows, 
                      vals.memptr(), h_R.memptr(), n_rows,
                      h_work.memptr(), (int) lwork,
                      iwork.memptr(), liwork,
                      &info );
}

//void gpu_mat::low_rank(int rank, int over_samp, int qr_iter)
//{
//    gpu_mat A(get_copy(), n_rows, n_cols);
//    rand_proj(A, rank, over_samp, qr_iter);
//}


void gpu_mat::rand_proj(gpu_mat const& A, int rank, int over_samp, int qr_iter)
{
    rand_prod(rank+over_samp);   // X = A * Rand
    QR_Q();                      // Q = QR(X).Q

    for(int i=1; i<qr_iter; ++i)
    {
        mat_mult(A,'N','T',true);  // Q_tilde = QR(A' * Q).Q
        QR_Q();

        mat_mult(A,'N','N',true);  // Q_tilde = QR(A * Q).Q
        QR_Q();
    }

    n_cols = rank; // Fake subsetting of columns Q = Q.cols(0,rank-1);
}

void gpu_mat::rand_proj(int rank, int over_samp, int qr_iter)
{
    gpu_mat A = this->make_copy();

    rand_proj(A, rank, over_samp, qr_iter);
}


void gpu_mat::rand_prod(int l)
{
    gpu_mat rand(n_cols, l);
    rand.fill_rnorm(0.0,1.0);

    mat_mult(rand, 'N', 'N',false);
}

void gpu_mat::QR_Q()
{
    int min_rc = n_rows > n_cols ? n_cols : n_rows;
    int nb = magma_get_dgeqrf_nb( n_rows );

    arma::vec tau(min_rc);
    gpu_mat work((2*min_rc + ((n_cols + 31)/32)*32 ), nb);

    int info;
    magma_dgeqrf_gpu( n_rows, n_cols, mat, n_rows, tau.memptr(), work.get_ptr(), &info );
    RT_ASSERT(info==0, "QR failed.");

    magma_dorgqr_gpu( n_rows, n_cols, min_rc, mat, n_rows, tau.memptr(), work.get_ptr(), nb, &info );
    RT_ASSERT(info==0, "Q recovery failed.");
}

void gpu_mat::chol(char uplo)
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

void gpu_mat::inv_chol(char uplo)
{
    RT_ASSERT(n_rows == n_cols, "Matrix must be square.");
    RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

    int info;

    magma_uplo_t magma_uplo = uplo=='U' ? MagmaUpper : MagmaLower;

    magma_dpotri_gpu(magma_uplo, n_rows, mat, n_rows, &info);

    RT_ASSERT(info==0, "Inverse (dpotri) failed.");

    symmat(mat, n_rows, uplo, 64);
}

void gpu_mat::inv_sympd()
{
    chol('U');
    inv_chol('U');
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
    RT_ASSERT(n_rows==n_cols, "Matrix must be square.");
    RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

    mat = uplo=='U' ? arma::chol(mat, "upper") : arma::chol(mat, "lower");
}

void gpu_mat::inv_chol(char uplo)
{
    RT_ASSERT(n_rows == n_cols, "Matrix must be square.");
    RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

    mat = arma::inv(mat);
}

void gpu_mat::inv_sympd()
{
    RT_ASSERT(n_rows == n_cols, "Matrix must be square.");

    mat = arma::inv_sympd(mat);
}


#endif
