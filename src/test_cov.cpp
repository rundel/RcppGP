#include <RcppArmadillo.h>

#include "gpu_mat.hpp"
#include "test_cov.hpp"
#include "cov_model.hpp"

SEXP check_gpu_mem()
{
BEGIN_RCPP

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    Rcpp::Rcout << free_mem  / (1024. * 1024) << "MB of " 
                << total_mem / (1024. * 1024) << "MB free.\n";

END_RCPP
}

SEXP test_calc_cov(SEXP model, SEXP dist, SEXP params)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    return Rcpp::wrap(m.calc_cov(d,p));

END_RCPP
}

SEXP test_calc_inv_cov(SEXP model, SEXP dist, SEXP params)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    return Rcpp::wrap(m.calc_inv_cov(d,p));

END_RCPP
}

SEXP test_calc_chol_cov(SEXP model, SEXP dist, SEXP params)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    return Rcpp::wrap(arma::chol(m.calc_cov(d,p)));

END_RCPP
}


SEXP test_calc_cov_gpu(SEXP model, SEXP dist, SEXP params)
{
BEGIN_RCPP

    cov_model m(model);
    gpu_mat d( Rcpp::as<arma::mat>(dist) );
    arma::vec p = Rcpp::as<arma::vec>(params);

    return Rcpp::wrap(m.calc_cov_gpu(d,p));

END_RCPP
}

SEXP test_calc_inv_cov_gpu(SEXP model, SEXP dist, SEXP params)
{
BEGIN_RCPP

    cov_model m(model);
    gpu_mat d( Rcpp::as<arma::mat>(dist) );
    arma::vec p = Rcpp::as<arma::vec>(params);

    return Rcpp::wrap(m.calc_inv_cov_gpu(d,p));

END_RCPP
}


SEXP test_calc_chol_cov_gpu(SEXP model, SEXP dist, SEXP params)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    gpu_mat cov(m.calc_cov_gpu_ptr(d,p), d.n_rows, d.n_cols);
    chol(cov, 'U');

    return Rcpp::wrap(cov.get_mat());

END_RCPP
}