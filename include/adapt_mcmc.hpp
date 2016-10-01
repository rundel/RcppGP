#ifndef ADAPT_MCMC_HPP
#define ADAPT_MCMC_HPP

#include <assert.hpp>

class vihola_ind_adapt
{
private:    
    int n_adapt;
    double target_accept;
    double gamma;
    int n;

    arma::vec S;
    arma::vec U;
    arma::vec jump;

public:
    vihola_ind_adapt(int n_adapt_, double target_accept_, double gamma_, arma::vec tuning_)
      : n_adapt(n_adapt_),
        target_accept(target_accept_),
        gamma(gamma_)
    {
        init(tuning_);
    }

    vihola_ind_adapt(Rcpp::List& settings)
    {
        n_adapt = Rcpp::as<int>(settings["n_adapt"]);
        target_accept = Rcpp::as<double>(settings["target_accept"]);
        gamma = Rcpp::as<double>(settings["gamma"]);
        
        init( Rcpp::as<arma::vec>(settings["tuning"]) );
    }

    void init(arma::vec const& tuning) 
    {
        n = tuning.n_rows;
        S = arma::sqrt(tuning);
        
        update_jump();
    }

    arma::vec get_S()
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
        jump = S % U;
    }

    void update(int s, double alpha)
    {
        arma::vec valpha(n);
        valpha.fill(alpha);

        update(s, valpha);
    }

    void update(int s, arma::vec alpha)
    {
        BOOST_ASSERT_MSG(alpha.n_elem == n, "Length mismatch with alpha and tuning.");

        if(s < n_adapt)
        {
            double adapt_rate = std::min(1.0, pow(s,-gamma));
            
            S = S % sqrt(1 + adapt_rate*(alpha - target_accept));
        }

        //Rcpp::Rcout << arma::trans(S.rows(0,5));

        update_jump();
    }
};

#endif