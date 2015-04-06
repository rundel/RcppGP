#ifdef USE_GPU

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <magma.h>

#include "assert.hpp"
#include "gpu_util.hpp"
#include "gpu_mat.hpp"
#include "gpu_mat_op.hpp"

gpu_mat trans(gpu_mat const& m)
{
    gpu_mat res(m.get_n_cols(),m.get_n_rows());

    magmablas_dtranspose(m.get_n_rows(), m.get_n_cols(),
                         m.get_const_ptr(), m.get_n_rows(),
                         res.get_ptr(), res.get_n_rows());

    return res;
}

void fill_rnorm(gpu_mat &m, double mu, double sigma)
{
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    curandStatus_t rs = curandGenerateNormalDouble(prng, m.get_ptr(), m.get_n_rows()*m.get_n_cols(), mu, sigma);
    RT_ASSERT(rs == CURAND_STATUS_SUCCESS, "Normal sampling failed.");

    curandDestroyGenerator(prng);
}

void mult_mat_diag(gpu_mat &m, gpu_mat const& d, char side)
{
    // side = 'R' => m x d
    // side = 'L' => d x m

    RT_ASSERT(side=='L' || side=='R', "side must be L or R.");
    RT_ASSERT(d.get_n_rows() == (side=='L' ? m.get_n_rows() : m.get_n_cols()), "dimension mismatch");

    cublasSideMode_t cuda_side = side=='L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t bs = cublasDdgmm(handle, cuda_side,
                                    m.get_n_rows(), m.get_n_cols(),
                                    m.get_ptr(), m.get_n_rows(),
                                    d.get_const_ptr(), 1,
                                    m.get_ptr(), m.get_n_rows()); // Calc in place

    cublasDestroy(handle);
    RT_ASSERT(bs == CUBLAS_STATUS_SUCCESS, "Matrix Diag multiply failed.");
}


gpu_mat mult_mat(gpu_mat const& X, gpu_mat const& Y, char op_X, char op_Y)
{
    RT_ASSERT(op_X=='N' || op_X=='T', "op_X must be N or T.");
    RT_ASSERT(op_Y=='N' || op_Y=='T', "op_Y must be N or T.");

    cublasOperation_t cuda_op_A = op_X=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuda_op_B = op_Y=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;

    int m = (op_X=='N') ? X.get_n_rows() : X.get_n_cols();
    int n = (op_Y=='N') ? Y.get_n_cols() : Y.get_n_rows();
    int k = (op_X=='N') ? X.get_n_cols() : X.get_n_rows();

    int k2 = (op_Y=='N') ? Y.get_n_rows() : Y.get_n_cols();
    RT_ASSERT(k2==k, "dim mismatch for matrix multiplication.");

    //if (swap_order)
    //    Rcpp::Rcout << "Swapped: ";
    //else
    //    Rcpp::Rcout << "       : ";

    //Rcpp::Rcout << m << " x " << k << " times "
    //            << k << " x " << n << "\n";

    gpu_mat res(m,n);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double one = 1.0, zero = 0.0;
    cublasStatus_t bs = cublasDgemm(handle, cuda_op_A, cuda_op_B,
                                    m, n, k, &one,
                                    X.get_const_ptr(), X.get_n_rows(),
                                    Y.get_const_ptr(), Y.get_n_rows(),
                                    &zero,
                                    res.get_ptr(), m
                                   );
    cublasDestroy(handle);
    RT_ASSERT(bs == CUBLAS_STATUS_SUCCESS, "Matrix multiply failed.");

    return res; 
}


void eig_sym(gpu_mat& m, arma::vec& vals)
{
    RT_ASSERT(m.get_n_rows()==m.get_n_cols(),"Matrix must be square");

    vals.resize(m.get_n_rows());

    // query for workspace sizes
    double lwork;
    int liwork, info;

    magma_dsyevd_gpu( MagmaVec, MagmaLower,
                      m.get_n_rows(), NULL, m.get_n_rows(), 
                      NULL, NULL, m.get_n_rows(),
                      &lwork,  -1,
                      &liwork, -1,
                      &info );

    // Calculate eigen decomposition
    arma::ivec iwork(liwork);
    arma::mat  h_R(m.get_n_rows(),m.get_n_rows());
    arma::vec  h_work((int) lwork);

    magma_dsyevd_gpu( MagmaVec, MagmaLower,
                      m.get_n_rows(),
                      m.get_ptr(), m.get_n_rows(),
                      vals.memptr(), h_R.memptr(), m.get_n_rows(),
                      h_work.memptr(), (int) lwork,
                      iwork.memptr(), liwork,
                      &info );
}

gpu_mat rand_prod(gpu_mat const& m, int l)
{
    gpu_mat r(m.get_n_cols(), l);
    fill_rnorm(r, 0.0,1.0);

    return mult_mat(m, r, 'N', 'N');
}

gpu_mat rand_proj(gpu_mat const& A, int rank, int over_samp, int qr_iter)
{
    gpu_mat Q = rand_prod(A, rank+over_samp);   // Q = A * Rand
    QR_Q(Q);                                    // Q = QR(X).Q

    for(int i=0; i<qr_iter; ++i)
    {
        Q = mult_mat(A,Q,'T','N');  // Q_tilde = QR(A' * Q).Q
        QR_Q(Q);

        Q = mult_mat(A,Q,'N','N');  // Q_tilde = QR(A * Q).Q
        QR_Q(Q);
    }

    return Q;
}

void solve(gpu_mat& A, gpu_mat& B, char trans)
{
    RT_ASSERT(A.get_n_rows()==A.get_n_cols(),"Matrix must be square");
    RT_ASSERT(A.get_n_rows()==B.get_n_rows(),"Dimension mismatch");

    int n = A.get_n_rows(),
        k = B.get_n_cols();

    magma_trans_t magma_trans = trans=='T' ? MagmaTrans : MagmaNoTrans;
    int info;

    arma::ivec ipiv(n);

    magma_dgetrf_gpu(n, n, A.get_ptr(), n, ipiv.memptr(), &info);
    RT_ASSERT(info==0, "LU failed.");

    magma_dgetrs_gpu(magma_trans, n, k,
                     A.get_ptr(), n, ipiv.memptr(),
                     B.get_ptr(), n, &info);
    RT_ASSERT(info==0, "LU Solve failed.");
}

void solve_sympd(gpu_mat& A, gpu_mat& B)
{
    RT_ASSERT(A.get_n_rows()==A.get_n_cols(),"Matrix must be square");
    RT_ASSERT(A.get_n_rows()==B.get_n_cols(),"Dimension mismatch");

    int n = A.get_n_rows(),
        k = B.get_n_cols();

    int info;

    magma_dpotrf_gpu(MagmaUpper, n, A.get_ptr(), n, &info);
    RT_ASSERT(info==0, "Chol failed.");

    magma_dpotrs_gpu(MagmaUpper, n, k,
                     A.get_ptr(), n,
                     B.get_ptr(), n,
                     &info);
    RT_ASSERT(info==0, "Chol Solve failed.");
}


gpu_mat low_rank_sympd(gpu_mat const& A, arma::vec& C, int rank, int over_samp, int qr_iter)
{
    RT_ASSERT(A.get_n_rows()==A.get_n_cols(),"Matrix must be square");

    gpu_mat Q = rand_proj(A, rank, over_samp, qr_iter);
    gpu_mat U = mult_mat(A, Q, 'N', 'N');
    U = mult_mat(Q, U, 'T', 'N'); // U = Qt * A * Q

    eig_sym(U,C);                  // U C Ut
    U = mult_mat(Q, U, 'N', 'N');  // U = Q * U;

    return U;
}

gpu_mat low_rank_sympd_op(gpu_mat const& A, arma::vec& C, int rank, int over_samp, int qr_iter)
{
    RT_ASSERT(A.get_n_rows()==A.get_n_cols(),"Matrix must be square");


    gpu_mat O(A.get_n_cols(), rank+over_samp);
    fill_rnorm(O, 0.0,1.0);

    gpu_mat Y = mult_mat(A, O, 'N', 'N');

    gpu_mat Q = Y.make_copy();  // Q = A * Rand
    QR_Q(Q);                    // Q = QR(X).Q

    for(int i=0; i<qr_iter; ++i)
    {
        Q = mult_mat(A,Q,'T','N');  // Q_tilde = QR(A' * Q).Q
        QR_Q(Q);

        Q = mult_mat(A,Q,'N','N');  // Q_tilde = QR(A * Q).Q
        QR_Q(Q);
    }

    arma::mat hO = O.get_mat(),
              hY = Y.get_mat(),
              hQ = Q.get_mat();

    // X = solve( A, B )  =>  X = A^-1 B
    // B (Q' O) = Q' Y
    //   =>  (Q' O)' B' = Y' Q
    //   =>  (O' Q) B' = Y' Q
    //   => B' = (O' Q)^-1 (Y' Q)

    //arma::mat hB = arma::solve(hO.t()*hQ, hY.t()*hQ).t();
    //gpu_mat U(hB);

    gpu_mat M = mult_mat(O,Q,'T','N');
    gpu_mat U = mult_mat(Y,Q,'T','N');

    solve(M, U, 'N');
    eig_sym(U,C);                        // U C Ut
    U = mult_mat(Q, U, 'N', 'N');  // U = Q * U;

    return U;
}


void QR_Q(gpu_mat &m)
{
    int min_rc = m.get_n_rows() > m.get_n_cols() ? m.get_n_cols() : m.get_n_rows();
    int nb = magma_get_dgeqrf_nb( m.get_n_rows() );

    arma::vec tau(min_rc);
    gpu_mat work((2*min_rc + ((m.get_n_cols() + 31)/32)*32), nb);

    int info;
    magma_dgeqrf_gpu(m.get_n_rows(), m.get_n_cols(), m.get_ptr(), 
                     m.get_n_rows(), tau.memptr(), work.get_ptr(), &info);
    RT_ASSERT(info==0, "QR failed.");

    magma_dorgqr_gpu(m.get_n_rows(), m.get_n_cols(), min_rc, m.get_ptr(), 
                     m.get_n_rows(), tau.memptr(), work.get_ptr(), nb, &info);
    RT_ASSERT(info==0, "Q recovery failed.");
}

void chol(gpu_mat& m, char uplo)
{
    RT_ASSERT(m.get_n_rows()==m.get_n_cols(), "Matrix must be square.");
    RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

    magma_uplo_t magma_uplo = uplo=='U' ? MagmaUpper : MagmaLower;
    int info;

    magma_dpotrf_gpu(magma_uplo, m.get_n_rows(),
                     m.get_ptr(), m.get_n_rows(),
                     &info);
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
    trimat(m.get_ptr(), m.get_n_rows(), uplo, 0.0, 64);
}

void inv_chol(gpu_mat& m, char uplo)
{
    RT_ASSERT(m.get_n_rows() == m.get_n_cols(), "Matrix must be square.");
    RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

    int info;
    magma_uplo_t magma_uplo = uplo=='U' ? MagmaUpper : MagmaLower;

    magma_dpotri_gpu(magma_uplo, m.get_n_rows(), 
                     m.get_ptr(), m.get_n_rows(), &info);

    RT_ASSERT(info==0, "Inverse (dpotri) failed.");

    symmat(m.get_ptr(), m.get_n_rows(), uplo, 64);
}

void inv_sympd(gpu_mat& m)
{
    chol(m,'U');
    inv_chol(m,'U');
}


void scale(gpu_mat& m, double const s)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t bs = cublasDscal(handle, m.get_n_rows()*m.get_n_cols(),
                                    &s, m.get_ptr(), 1);
    cublasDestroy(handle);

    RT_ASSERT(bs == CUBLAS_STATUS_SUCCESS, "Matrix multiply failed.");
}


gpu_mat diag(gpu_mat const& m)
{
    RT_ASSERT(m.get_n_rows()==m.get_n_cols(),"Matrix must be square");

    cublasHandle_t handle;
    cublasCreate(&handle);

    gpu_mat res(m.get_n_rows(),1);

    cublasStatus_t bs = cublasDcopy(handle, m.get_n_rows(),
                                    m.get_const_ptr(), m.get_n_rows()+1,
                                    res.get_ptr(), 1);
    cublasDestroy(handle);

    RT_ASSERT(bs == CUBLAS_STATUS_SUCCESS, "Get diagonal failed.");

    return res;
}

void add_diag(gpu_mat& m, gpu_mat const& d)
{
    RT_ASSERT(m.get_n_rows()==m.get_n_cols(),"Matrix must be diagonal");
    RT_ASSERT(m.get_n_rows()==d.get_n_rows(),"Dimension mismatch");

    cublasHandle_t handle;
    cublasCreate(&handle);

    double one = 1.0;
    cublasStatus_t bs = cublasDaxpy(handle, m.get_n_rows(), &one,
                                    d.get_const_ptr(), 1,
                                    m.get_ptr(), m.get_n_rows()+1);
    cublasDestroy(handle);

    RT_ASSERT(bs == CUBLAS_STATUS_SUCCESS, "Diagonal add failed.");
}

void add_mat(gpu_mat& X, gpu_mat const& Y)
{
    RT_ASSERT(X.get_n_rows()==Y.get_n_rows() & X.get_n_cols()==Y.get_n_cols(),"Dimension mismatch");

    cublasHandle_t handle;
    cublasCreate(&handle);

    double one = 1.0;
    cublasStatus_t bs = cublasDaxpy(handle, X.get_n_rows()*Y.get_n_cols(), &one,
                                    Y.get_const_ptr(), 1,
                                    X.get_ptr(), 1);
    cublasDestroy(handle);

    RT_ASSERT(bs == CUBLAS_STATUS_SUCCESS, "Matrix add failed.");
}


gpu_mat inv_lr(gpu_mat const& S, arma::vec& A, int rank, int over_samp, int qr_iter, bool mod)
{
    // Inverse using low rank approx. via Sherman–Morrison–Woodbury formula
    // (A+UCV)^-1 = A^-1 - A^-1 U (C^-1+U' A^-1 U)^-1 U' A^-1

    RT_ASSERT(S.get_n_rows() == S.get_n_cols(), "Matrix must be square.");

    // Assume this gpu_mat is B which will be approximated by U C U'
    arma::vec C;
    gpu_mat U = low_rank_sympd(S, C, rank, over_samp, qr_iter);

    if (mod)
    {
        gpu_mat gC(C);
        gpu_mat cov = U.make_copy();
        mult_mat_diag(cov, gC,'R');
        cov = mult_mat(cov,U,'N','T');

        A += diag(S).get_mat() - diag(cov).get_mat();
    }

    gpu_mat Ainv(1.0/A);
    gpu_mat Cinv(1.0/C);

    gpu_mat Ainv_U = U.make_copy();   // Ainv_U = U
    mult_mat_diag(Ainv_U, Ainv, 'L'); // Ainv_U = A^-1 U

    gpu_mat M = mult_mat(U,Ainv_U,'T','N');  // M = U' A^-1 U
    add_diag(M,Cinv);                              // M = C^-1 + U' A^-1 U
    inv_sympd(M);                                  // M = (C^-1 + U' A^-1 U)^-1

    M = mult_mat(M,Ainv_U,'N','T');  // M = (C^-1+U' A^-1 U)^-1 U' A^-1
    M = mult_mat(Ainv_U,M,'N','N');  // M = A^-1 U (C^-1+U' A^-1 U)^-1 U' A^-1
    scale(M, -1.0);                  // M = - A^-1 U (C^-1+U' A^-1 U)^-1 U' A^-1
    add_diag(M, Ainv);               // M = A^-1 - A^-1 U (C^-1+U' A^-1 U)^-1 U' A^-1

    return M;
}


gpu_mat inv_pp(gpu_mat const& S, gpu_mat const& U, gpu_mat const& C, arma::vec A, bool mod)
{
    // U = Cov_12
    // C = Cov_22
    // A = Nugget

    gpu_mat Cinv = C.make_copy();
    inv_sympd(Cinv);

    if (mod)
    {
        gpu_mat cov = mult_mat(U,Cinv,'N','N');
        cov = mult_mat(cov,U,'N','T');

        A += diag(S).get_mat() - diag(cov).get_mat();
    }

    gpu_mat Ainv(1.0/A);

    // (A+UCV)^-1 = A^-1 - A^-1 U (C^-1+U' A^-1 U)^-1 U' A^-1
    gpu_mat Ut_Ainv = trans(U);
    mult_mat_diag(Ut_Ainv, Ainv, 'R');


    gpu_mat M = mult_mat(Ut_Ainv, U,'N','N');  // M = U' A^-1 U
    add_mat(M, Cinv);                               // M = C^-1 + U' A^-1 U

    gpu_mat tmp = Ut_Ainv.make_copy();
    solve(M, tmp, 'N');
    tmp = mult_mat(Ut_Ainv, tmp,'T','N');   // M = A^-1 U (C^-1+U' A^-1 U)^-1 U' A^-1
    scale(tmp, -1.0);                     // = - A^-1 U (C^-1+U' A^-1 U)^-1 U' A^-1
    add_diag(tmp, Ainv);                  // = A^-1 - A^-1 U (C^-1+U' A^-1 U)^-1 U' A^-1

    return tmp;
}

#endif