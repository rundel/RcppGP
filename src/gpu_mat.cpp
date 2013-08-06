#ifdef USE_GPU

#include <magma.h>

#include "init.hpp"
#include "gpu_mat.hpp"
#include "gpu_util.hpp"

void chol(gpu_mat& M, char uplo)
{
    init();

    int m = M.n_rows;
    int n = M.n_cols;

    RT_ASSERT(m==n, "Matrix must be square.");
    RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

    int info;
    magma_dpotrf_gpu(uplo, n, M.mat, n, &info);
    
    RT_ASSERT(info==0, "Cholesky failed.");

    trimat(M.mat, n, uplo, 0.0, 64);

    finalize();
}

void inv_chol(gpu_mat& chol, char uplo)
{
    init();

    int m = chol.n_rows;
    int n = chol.n_cols;

    RT_ASSERT(m==n, "Matrix must be square.");
    RT_ASSERT(uplo=='U' || uplo=='L', "uplo must be U or L.");

    int info;
    magma_dpotri_gpu(uplo, n, chol.mat, n, &info);

    RT_ASSERT(info==0, "Inverse (dpotri) failed.");

    symmat(chol.mat, n, uplo, 64);

    finalize();
}

void inv_sympd(gpu_mat& M)
{
    chol(M,'U');
    inv_chol(M,'U');
}

#endif
