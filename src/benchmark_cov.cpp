#include <RcppArmadillo.h>

#include <boost/timer/timer.hpp>

#include "gpu_mat.hpp"
#include "cov_model.hpp"

// [[Rcpp::export]]
double benchmark_calc_cov(Rcpp::List model, arma::mat d, arma::vec p, int n, bool gpu = false)
{
    cov_model m(model);

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();

        arma::mat res;
        if (gpu)
        {
            gpu_mat g(d);
            res = m.calc_cov_gpu(g,p);
        }
        else
            res = m.calc_cov(d,p);

        timer.stop();

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return rs.mean();
}

// [[Rcpp::export]]
double benchmark_calc_inv_cov(Rcpp::List model, arma::mat d, arma::vec p, int n, bool gpu = false)
{
    cov_model m(model);

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();

        arma::mat res;
        if (gpu)
        {
            gpu_mat g(d);
            res = m.calc_inv_cov_gpu(g,p);
        }
        else
            res = m.calc_inv_cov(d,p);

        timer.stop();

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return rs.mean();
}


// [[Rcpp::export]]
double benchmark_calc_chol_cov(Rcpp::List model, arma::mat d, arma::vec p, int n, bool gpu = false)
{
    cov_model m(model);

    arma::running_stat<double> rs;

    boost::timer::cpu_timer timer;
    for(int i=0; i<n; ++i)
    {
        timer.start();
        arma::mat res;

        if (gpu)
        {
            gpu_mat cov(m.calc_cov_gpu_ptr(d,p), d.n_rows, d.n_cols);
            cov.chol('U');
            res = cov.get_mat();
        }
        else
        {
            res = arma::chol(m.calc_cov(d,p));
        }

        timer.stop();

        rs( timer.elapsed().wall / 1000000000.0L );
    }

    return rs.mean();
}