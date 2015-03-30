#include <RcppArmadillo.h>

#include "gpu_mat.hpp"
#include "cov_model.hpp"

// [[Rcpp::export]]
arma::mat test_gpu_mat(arma::mat const& d)
{
    gpu_mat dist(d);

    Rcpp::Rcout << "rows: " << dist.n_rows << " cols: " << dist.n_cols << "\n";

    return dist.get_mat();
}


// [[Rcpp::export]]
arma::mat calc_cov(Rcpp::List model, arma::mat d, arma::vec p, bool gpu = false)
{
    cov_model m(model);

    if (gpu)
    {
        gpu_mat dist(d);
        return m.calc_cov_gpu(dist,p);
    }

    return m.calc_cov(d,p);
}

// [[Rcpp::export]]
arma::mat calc_inv_cov(Rcpp::List model, arma::mat d, arma::vec p, bool gpu = false)
{
    cov_model m(model);

    if (gpu)
    {
        gpu_mat dist(d);
        return m.calc_inv_cov_gpu(dist,p);
    }

    return m.calc_inv_cov(d,p);
}

// [[Rcpp::export]]
arma::mat calc_chol_cov(Rcpp::List model, arma::mat d, arma::vec p, bool gpu = false)
{
    cov_model m(model);

    if (gpu)
    {
        gpu_mat dist(d);
        gpu_mat cov(m.calc_cov_gpu_ptr(dist, p), d.n_rows, d.n_cols);
        cov.chol('U');

        return cov.get_mat();
    }

    return arma::chol(m.calc_cov(d,p));
}
