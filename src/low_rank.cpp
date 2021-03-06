#include <RcppArmadillo.h>

#include <boost/assert.hpp>

arma::mat rand_proj(arma::mat const& A, int rank, int over_samp, int qr_iter)
{
    int l = rank + over_samp;
    int n = A.n_cols;

    BOOST_ASSERT_MSG(l < n, "dim mismatch.");

    arma::mat Q, Q_tilde, R;
    BOOST_ASSERT_MSG(arma::qr_econ(Q, R, A * arma::randn(n, l)), "QR failed.");

    for(int i=0; i<qr_iter; ++i)
    {
        BOOST_ASSERT_MSG(arma::qr_econ(Q_tilde, R, A.t()*Q), "QR failed.");
        BOOST_ASSERT_MSG(arma::qr_econ(Q, R, A*Q_tilde), "QR failed.");
    }

    Q = Q.cols(0,rank-1);

    return Q;
}

void low_rank_sympd(arma::mat const& A, arma::mat& U, arma::vec& C,
                    int rank, int over_samp, int qr_iter)
{
    arma::mat Q = rand_proj(A, rank, over_samp, qr_iter);

    BOOST_ASSERT_MSG(arma::eig_sym(C, U, Q.t() * A * Q), "eig_sym failed.");
    U = Q * U;
}

void low_rank(arma::mat const& A, arma::mat& U, arma::vec& C, arma::mat& V,
              int rank, int over_samp, int qr_iter)
{
    arma::mat Q = rand_proj(A, rank, over_samp, qr_iter);

    BOOST_ASSERT_MSG(arma::svd_econ(U, C, V, Q.t() * A), "svd_econ failed.");
    U = Q * U;
}
