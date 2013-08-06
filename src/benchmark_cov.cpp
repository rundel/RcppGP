#include <RcppArmadillo.h>

#include <boost/timer/timer.hpp>

#include "gpu_mat.hpp"
#include "benchmark_cov.hpp"
#include "cov_model.hpp"

SEXP benchmark_calc_cov(SEXP model, SEXP dist, SEXP params, SEXP n_rep)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    int n = Rcpp::as<int>(n_rep);

    int x = 0;

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();
        arma::mat res = m.calc_cov(d,p);
        timer.stop();

        x+=res.n_elem;

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return Rcpp::wrap(rs.mean());

END_RCPP
}

SEXP benchmark_calc_inv_cov(SEXP model, SEXP dist, SEXP params, SEXP n_rep)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    int n = Rcpp::as<int>(n_rep);

    int x = 0;

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();
        arma::mat res = m.calc_inv_cov(d,p);
        timer.stop();

        x+=res.n_elem;

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return Rcpp::wrap(rs.mean());

END_RCPP
}

SEXP benchmark_calc_chol_cov(SEXP model, SEXP dist, SEXP params, SEXP n_rep)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    int n = Rcpp::as<int>(n_rep);

    int x = 0;

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();
        arma::mat res = arma::chol(m.calc_cov(d,p));
        timer.stop();

        x+=res.n_elem;

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return Rcpp::wrap(rs.mean());

END_RCPP
}


SEXP benchmark_calc_cov_gpu(SEXP model, SEXP dist, SEXP params, SEXP n_rep)
{
BEGIN_RCPP

    cov_model m(model);
    gpu_mat d( Rcpp::as<arma::mat>(dist) );
    arma::vec p = Rcpp::as<arma::vec>(params);

    int n = Rcpp::as<int>(n_rep);

    int x = 0;

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();
        arma::mat res = m.calc_cov_gpu(d,p);
        timer.stop();

        x+=res.n_elem;

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return Rcpp::wrap(rs.mean());

END_RCPP
}

SEXP benchmark_calc_inv_cov_gpu(SEXP model, SEXP dist, SEXP params, SEXP n_rep)
{
BEGIN_RCPP

    cov_model m(model);
    gpu_mat d( Rcpp::as<arma::mat>(dist) );
    arma::vec p = Rcpp::as<arma::vec>(params);

    int n = Rcpp::as<int>(n_rep);

    int x = 0;

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();
        arma::mat res = m.calc_inv_cov_gpu(d,p);
        timer.stop();

        x+=res.n_elem;

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return Rcpp::wrap(rs.mean());

END_RCPP
}


SEXP benchmark_calc_chol_cov_gpu(SEXP model, SEXP dist, SEXP params, SEXP n_rep)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    int n = Rcpp::as<int>(n_rep);

    int x = 0;

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();
        gpu_mat cov(m.calc_cov_gpu_ptr(d,p), d.n_rows, d.n_cols);
        chol(cov, 'U');
        arma::mat res = cov.get_mat();
        timer.stop();

        x+=res.n_elem;

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return Rcpp::wrap(rs.mean());

END_RCPP
}