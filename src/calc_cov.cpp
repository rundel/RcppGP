#include <RcppArmadillo.h>

#include "assert.hpp"
#include "low_rank.hpp"
#include "gpu_mat.hpp"
#include "cov_model.hpp"


// [[Rcpp::export]]
arma::mat calc_cov(Rcpp::List model, arma::mat d, arma::vec p, bool gpu = false)
{
    cov_model m(model);

    if (gpu)
    {
        gpu_mat cov = m.calc_cov(gpu_mat(d),p);
        return cov.get_mat();
    }

    return m.calc_cov(d,p);
}


// [[Rcpp::export]]
arma::mat calc_inv_cov(Rcpp::List model, arma::mat d, arma::vec p, bool gpu = false)
{
    RT_ASSERT(d.n_rows==d.n_cols,"Cov matrix must be symmetric");

    cov_model m(model);

    if (gpu)
    {
        gpu_mat cov = m.calc_cov(gpu_mat(d),p);
        cov.inv_sympd();
        return cov.get_mat();
    }

    return arma::inv_sympd(m.calc_cov(d,p));
}


// [[Rcpp::export]]
Rcpp::List calc_low_rank_cov(Rcpp::List model, arma::mat d, arma::vec p,
                             int rank, int over_samp = 5, int qr_iter = 2,
                             bool gpu = false)
{
    RT_ASSERT(d.n_rows==d.n_cols,"Cov matrix must be symmetric");

    cov_model m(model);

    arma::mat U;
    arma::vec C;

    if (gpu)
    {
        gpu_mat cov = m.calc_cov(gpu_mat(d),p);
        cov.low_rank_sympd(C, rank, over_samp, qr_iter);

        U = cov.get_mat();
    }
    else
    {
        low_rank_sympd(m.calc_cov(d,p),U, C, rank, over_samp, qr_iter);
    }

    return Rcpp::List::create(Rcpp::Named("C") = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(C)),
                              Rcpp::Named("U") = U);
}
