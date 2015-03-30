#include <iostream>
#include <string>

#include <boost/timer/timer.hpp>

#include "assert.hpp"
#include "report.hpp"

#include "adapt_mcmc.hpp"
#include "spLM.hpp"
#include "cov_model.hpp"
#include "model_state_lm.hpp"



SEXP spLM(SEXP Y_r, SEXP X_r, 
          SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
          SEXP cov_model_r,
          SEXP theta_r, SEXP beta_r,
          SEXP is_pp_r, SEXP is_mod_pp_r,
          SEXP n_samples_r, SEXP verbose_r, SEXP n_report_r)
{
BEGIN_RCPP

    Rcpp::RNGScope scope;

    bool is_pp = Rcpp::as<bool>(is_pp_r);
    bool is_mod_pp = Rcpp::as<bool>(is_mod_pp_r);

    arma::vec Y = Rcpp::as<arma::vec>(Y_r);
    arma::mat X = Rcpp::as<arma::mat>(X_r);

    arma::mat coordsD = Rcpp::as<arma::mat>(coordsD_r);
    arma::mat knotsD = Rcpp::as<arma::mat>(knotsD_r);
    arma::mat coordsKnotsD = Rcpp::as<arma::mat>(coordsKnotsD_r);

    int p = X.n_cols;       // # of betas
    int n = X.n_rows;       // # of samples
    int m = knotsD.n_rows;  // # of knots
    
    cov_model cov_settings(cov_model_r);

    Rcpp::List theta_settings(theta_r);
    Rcpp::List beta_settings(beta_r);
    
    
    prior_details beta_prior;
    beta_prior.type = Rcpp::as<std::string>(beta_settings["prior"]);
    
    if (beta_prior.type == "normal")
    {
        Rcpp::List beta_hyperparam_list = Rcpp::as<Rcpp::List>(beta_settings["hyperparam"]);
        RT_ASSERT(beta_hyperparam_list.size() == 2, "Beta normal prior must have 2 hyper parameters (mu and sd).");

        
        beta_prior.mu    = Rcpp::as<arma::vec>(beta_hyperparam_list[0]);
        beta_prior.sigma = Rcpp::as<arma::vec>(beta_hyperparam_list[1]);

        RT_ASSERT(beta_prior.mu.size()    == p, "Length of hyperparameter mu must match number of betas.");
        RT_ASSERT(beta_prior.sigma.size() == p, "Length of hyperparameter sigma must match number of betas.");
    }

    int n_samples = Rcpp::as<int>(n_samples_r);
    int n_report  = Rcpp::as<int>(n_report_r);
    bool verbose  = Rcpp::as<bool>(verbose_r);
    
    int n_theta = cov_settings.nparams;

    arma::mat theta(n_theta, n_samples);
    arma::mat beta(p, n_samples);
    arma::mat w(n, n_samples);
    
    arma::mat w_star, e;
    if (is_pp)
        w_star.resize(m, n_samples); 
    if (is_mod_pp)
        e.resize(n,n_samples);

    int n_loglik = (!is_pp) ? 5 : (!is_mod_pp) ? 5 : 6; 
    arma::mat loglik(n_loglik, n_samples);


    if (verbose) {
        Rcpp::Rcout << "----------------------------------------\n"
                    << "\tGeneral model description\n"
                    << "----------------------------------------\n"
                    << "\n"
                    << "Fitting LM ...\n" 
                    << "Observations: " << n << "\n"
                    << "Covariates  : " << p << "\n"
                    << "MCMC samples: " << n_samples << "\n"
                    << "Using PP    : " << is_pp << "\n";
        if (is_pp)
        {
            Rcpp::Rcout << "Using mod PP: " << is_mod_pp << "\n"
                        << "Knots       : " << m << "\n";
        }

        Rcpp::Rcout << "\n";

        Rcpp::Rcout << "Priors and hyperpriors:\n";
        //Rcpp::Rcout << "Using the " << cov_settings << " spatial correlation model.\n";
        
        // FIXME
    
        report_start();
    } 

    Rcpp::List accept_results;
    boost::timer::cpu_timer timer;
  

    vihola_adapt theta_amcmc(theta_settings);

    model_state_lm cur_state(&cov_settings, &beta_prior, is_pp, is_mod_pp);

    cur_state.theta = Rcpp::as<arma::vec>(theta_settings["start"]);
    if (!is_pp) cur_state.update_covs(coordsD);
    else        cur_state.update_covs(knotsD, coordsKnotsD);
    
    cur_state.update_beta(X, Y);            
    cur_state.update_w();

    cur_state.calc_all_loglik();
    
    int accept_theta = 0, batch_accept_theta = 0;
    std::vector<double> acc_rate_theta;

    for (int s = 0; s < n_samples; s++) 
    {
        // Update beta
        cur_state.update_beta(X, Y);        
        cur_state.calc_w_loglik();
        cur_state.calc_beta_loglik();
        cur_state.calc_loglik();

        
        // Update theta

        model_state_lm cand_state(cur_state);
        
        cand_state.update_theta(theta_amcmc.get_jump());
        if (!is_pp) cand_state.update_covs(coordsD);
        else        cand_state.update_covs(knotsD, coordsKnotsD);
        
        cand_state.calc_all_loglik();

        double alpha = std::min(1.0, exp(cand_state.loglik - cur_state.loglik));
        if (Rcpp::runif(1)[0] <= alpha)
        {
            cur_state = cand_state;

            accept_theta++;
            batch_accept_theta++;
        }
        theta_amcmc.update(s, alpha);

        // Update w
    
        cur_state.update_w();

        w.col(s) = cur_state.w;
        beta.col(s) = cur_state.beta;
        theta.col(s) = cur_state.theta;
        if (is_pp)
            w_star.col(s) = cur_state.ws;

        if ((s+1) % n_report == 0)
        {
            if (verbose)
            {
                long double wall_sec = timer.elapsed().wall / 1000000000.0L;

                report_sample(s+1, n_samples, wall_sec);
                Rcpp::Rcout << "Log likelihood: " << cur_state.loglik << "\n";
                report_accept("theta - ", s+1, accept_theta, batch_accept_theta, n_report);
                report_line();
            }

            acc_rate_theta.push_back(1.0*accept_theta/(s+1));
            batch_accept_theta = 0;
        }
    }

    Rcpp::List results;

    results["beta"]     = beta;
    results["theta"]    = theta;
    results["accept"]   = accept_results;
    results["w"]        = w;
    if (is_pp)
        results["w_star"]   = w_star;
    

    return results;

END_RCPP
}
