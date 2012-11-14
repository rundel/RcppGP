#ifndef ADAPT_MCMC_HPP
#define ADAPT_MCMC_HPP

#include <RcppArmadillo.h>
#include "cholesky_util.hpp"

class vihola_adapt
{
private:    
    int n_adapt;
    double target_accept;
    double gamma;
    int n;

    arma::mat S;
    arma::vec U;
    arma::vec jump;

public:
    vihola_adapt(int n_adapt_, double target_accept_, double gamma_, arma::mat tuning_)
      : n_adapt(n_adapt_),
        target_accept(target_accept_),
        gamma(gamma_)
    {
        init(tuning_);
    }

    vihola_adapt(Rcpp::List settings)
    {
        n_adapt = Rcpp::as<int>(settings["n_adapt"]);
        target_accept = Rcpp::as<double>(settings["target_accept"]);
        gamma = Rcpp::as<double>(settings["gamma"]);
        
        init( Rcpp::as<arma::mat>(settings["tuning"]) );
    }

    void init(arma::mat const& tuning) 
    {
        n = tuning.n_rows;
        if (tuning.n_rows == tuning.n_cols) {
            S = arma::chol(tuning).t();             
        } else {
            if (tuning.n_cols == 1)
                S = arma::diagmat( arma::sqrt(tuning) );
            else 
                throw std::runtime_error("Tuning argument must be a square matrix or vector.");
        }
        
        update_jump();
    }

    arma::mat get_S()
    {
        return S;
    }

    arma::vec get_jump()
    {
        return jump;
    }

    void update_jump()
    {
        U = arma::randn<arma::vec>(n);
        jump = S * U;
    }

    void update(int s, double alpha)
    {
        if(s < n_adapt)
        {
            double adapt_rate = std::min(1.0, n * pow(s,-gamma));
            
            double c = adapt_rate*(alpha - target_accept) / arma::dot(U,U);

            arma::vec v = sqrt(fabs(c)) * jump;
            if (c < 0.0)
                chol_downdate(S, v);
            else if (c > 0.0)
                chol_update(S, v);
        }

        update_jump();
    }
};

#endif