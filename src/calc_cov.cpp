#include <RcppArmadillo.h>

#include "assert.hpp"
#include "low_rank.hpp"
#include "gpu_mat.hpp"
#include "gpu_mat_op.hpp"
#include "cov_model.hpp"


// [[Rcpp::export]]
Rcpp::List calc_low_rank(arma::mat cov, int rank,
                         int over_samp = 0, int qr_iter = 2,
                         bool gpu = false)
{
    RT_ASSERT(cov.n_rows==cov.n_cols,"Cov matrix must be symmetric");

    arma::mat U;
    arma::vec C;

    if (gpu)
    {
        U = low_rank_sympd(gpu_mat(cov), C, rank, over_samp, qr_iter).get_mat();
    }
    else
    {
        low_rank_sympd(cov, U, C, rank, over_samp, qr_iter);
    }

    return Rcpp::List::create(Rcpp::Named("C") = C,
                              Rcpp::Named("U") = U);
}


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
Rcpp::List calc_inv_cov(Rcpp::List model, arma::mat d, arma::vec p, arma::vec nug,
                        arma::mat d_btw, arma::mat d_knots,
                        int rank=1, int over_samp=0, int qr_iter=0,
                        bool gpu = false, bool low_rank = false, bool pred_proc = false,
                        bool mod = false)
{
    RT_ASSERT(d.n_rows==d.n_cols,"Cov matrix must be symmetric");

    arma::wall_clock t;
    double wt;

    cov_model m(model);
    arma::mat res;

    t.tic();
    if (gpu)
    {
        gpu_mat cov = m.calc_cov(gpu_mat(d),p);

        if (low_rank)
        {
            gpu_mat g = inv_lr(cov, nug, rank, over_samp, qr_iter, mod);

            wt = t.toc();
            res = g.get_mat();
        }
        else if (pred_proc)
        {
            gpu_mat cov_btw   = m.calc_cov(gpu_mat(d_btw),p);
            gpu_mat cov_knots = m.calc_cov(gpu_mat(d_knots),p);

            gpu_mat g = inv_pp(cov, cov_btw, cov_knots, nug, mod);

            wt = t.toc();
            res = g.get_mat();
        }
        else
        {
            gpu_mat diag(nug);

            add_diag(cov,diag);
            inv_sympd(cov);

            wt = t.toc();
            res = cov.get_mat();
        }
    }
    else
    {
        res = arma::inv_sympd(m.calc_cov(d,p) + arma::diagmat(nug));
        wt = t.toc();
    }

    return Rcpp::List::create(Rcpp::Named("time") = wt,
                              Rcpp::Named("C") = res);
}

/*

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
*/