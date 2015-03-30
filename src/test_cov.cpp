#include <RcppArmadillo.h>

#include "gpu_mat.hpp"
#include "cov_model.hpp"

// [[Rcpp::export]]
void check_gpu_mem()
{
#ifdef HAVE_GPU
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    Rcpp::Rcout << free_mem  / (1024. * 1024) << "MB of " 
                << total_mem / (1024. * 1024) << "MB free.\n";
#else
    Rcpp::Rcout << "No GPU detected!\n";
#endif
}

// [[Rcpp::export]]
arma::mat test_calc_cov(Rcpp::List model, arma::mat d, arma::vec p)
{
    cov_model m(model);

    return m.calc_cov(d,p);
}

arma::mat test_calc_inv_cov(Rcpp::List model, arma::mat d, arma::vec p)
{
    cov_model m(model);

    return m.calc_inv_cov(d,p);
}

// [[Rcpp::export]]
arma::mat test_calc_chol_cov(Rcpp::List model, arma::mat d, arma::vec p)
{
    cov_model m(model);

    return arma::chol(m.calc_cov(d,p));
}


// [[Rcpp::export]]
arma::mat test_calc_cov_gpu(Rcpp::List model, arma::mat d, arma::vec p)
{
    cov_model m(model);

    return m.calc_cov_gpu(d,p);
}

// [[Rcpp::export]]
arma::mat test_calc_inv_cov_gpu(Rcpp::List model, arma::mat d, arma::vec p)
{
    cov_model m(model);

    return m.calc_inv_cov_gpu(d,p);
}


// [[Rcpp::export]]
arma::mat test_calc_chol_cov_gpu(Rcpp::List model, arma::mat d, arma::vec p)
{
    cov_model m(model);

    gpu_mat cov(m.calc_cov_gpu_ptr(d,p), d.n_rows, d.n_cols);
    cov.chol('U');

    return cov.get_mat();
}