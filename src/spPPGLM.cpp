#include <limits>

#include "assert.hpp"
#include "report.hpp"
#include "spPPGLM.hpp"
#include "distributions.hpp"
#include "adapt_mcmc.hpp"
#include "model_state_pp_glm.hpp"
#include "model_state_mpp_glm.hpp"

SEXP spPPGLM(SEXP Y_r, SEXP X_r,
             SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
             SEXP family_r, SEXP weights_r,
             SEXP beta_r, SEXP ws_r, SEXP e_r,
             SEXP cov_model_r, SEXP is_mod_pp_r,
             SEXP n_samples_r, SEXP verbose_r, SEXP n_report_r,
             SEXP n_adapt_r, SEXP target_acc_r, SEXP gamma_r)
{    
//BEGIN_RCPP

    Rcpp::RNGScope scope;

    arma::vec Y = Rcpp::as<arma::vec>(Y_r);
    arma::mat X = Rcpp::as<arma::mat>(X_r);

    arma::mat coordsD = Rcpp::as<arma::mat>(coordsD_r);
    arma::mat knotsD = Rcpp::as<arma::mat>(knotsD_r);
    arma::mat coordsKnotsD = Rcpp::as<arma::mat>(coordsKnotsD_r);
    
    int p = X.n_cols;       // # of betas
    int n = X.n_rows;       // # of samples
    int m = knotsD.n_rows;  // # of knots
  
    std::string family = Rcpp::as<std::string>(family_r);
    arma::vec weights = Rcpp::as<arma::vec>(weights_r);

    bool is_mod_pp = Rcpp::as<bool>(is_mod_pp_r);

    cov_model cov_settings(cov_model_r);

    Rcpp::List beta_settings(beta_r);
    Rcpp::List ws_settings(ws_r);
    Rcpp::List e_settings(e_r);

    std::string beta_prior = Rcpp::as<std::string>(beta_settings["prior"]);
    
    std::vector<arma::vec> beta_hyperparam;
    if (beta_prior == "normal")
    {
        Rcpp::List beta_hyperparam_list = Rcpp::as<Rcpp::List>(beta_settings["hyperparam"]);
        RT_ASSERT(beta_hyperparam_list.size() == 2, "Beta normal prior must have 2 hyper parameters (mu and sd).");

        for(int i=0; i!=beta_hyperparam_list.size(); ++i)
        {
            beta_hyperparam.push_back( Rcpp::as<arma::vec>(beta_hyperparam_list[i]) );
            RT_ASSERT(beta_hyperparam[i].size() == p, "Length of hyperparameter must match number of betas.");
        }
    }

    int n_samples = Rcpp::as<int>(n_samples_r);
    int n_report  = Rcpp::as<int>(n_report_r);
    bool verbose  = Rcpp::as<bool>(verbose_r);
    

    int n_theta = cov_settings.nparams;
    int n_total_p = n_theta + p;


    if(verbose){
        Rcpp::Rcout << "----------------------------------------\n";
        Rcpp::Rcout << "\tGeneral model description\n";
        Rcpp::Rcout << "----------------------------------------\n";
        Rcpp::Rcout << "Model fit with " << n << " observations.\n";
        Rcpp::Rcout << "Number of covariates " << p << " (including intercept if specified).\n";
        //Rcpp::Rcout << "Using the " << covModel << " spatial correlation model.\n";
        
        std::string mod_str = (is_mod_pp) ? "modified" : "non-modified"; 
        Rcpp::Rcout << "Using " << mod_str <<  " predictive process with " << m << " knots.\n";
        Rcpp::Rcout << "Number of MCMC samples " << n_samples << ".\n\n";

        Rcpp::Rcout << "Priors and hyperpriors:\n";
        
        // FIXME
    } 



    arma::mat w(n, n_samples);
    arma::mat w_star(m, n_samples); 
    arma::mat beta(p, n_samples);
    arma::mat theta(n_theta, n_samples);
    arma::mat loglik(6, n_samples);

    if (verbose) 
        report_start();

    vihola_adapt p_amcmc(beta_settings);
    vihola_adapt ws_amcmc(ws_settings);
    vihola_adapt e_amcmc(e_settings);

    double loglik_cur_theta, loglik_cur_beta, loglik_cur_link, loglik_cur_ws, loglik_cur_e;
    double loglik_cur = -std::numeric_limits<double>::max();
    
    int accept    = 0, batch_accept    = 0;
    int accept_ws = 0, batch_accept_ws = 0;
    int accept_e  = 0, batch_accept_e  = 0;

    std::vector<double> acc_rate, acc_rate_ws, acc_rate_e;
    Rcpp::List accept_results;

    if (!is_mod_pp)
    {
        model_state_pp_glm cur_state(&cov_settings);

        cur_state.theta = cov_settings.param_start;
        cur_state.beta = Rcpp::as<arma::vec>(beta_settings["start"]);
        cur_state.ws = Rcpp::as<arma::vec>(ws_settings["start"]);

        arma::wall_clock t;
        t.tic();

        for(int s = 0; s < n_samples; s++){

            model_state_pp_glm cand_state = cur_state;

            /////////////////////////
            // Update beta & theta
            ////////////////////////

            arma::vec U = p_amcmc.get_jump();
            arma::vec beta_jump  = U(arma::span(0,p-1));
            arma::vec theta_jump = U(arma::span(p,p+n_theta-1));
            
            cand_state.update_beta( beta_jump );
            cand_state.update_theta( theta_jump );
            
            cand_state.update_covs(knotsD, coordsKnotsD);
            cand_state.update_w();

            double loglik_cand_theta, loglik_cand_beta, loglik_cand_ws, loglik_cand_link;

            loglik_cand_theta = cand_state.calc_theta_loglik();
            loglik_cand_beta  = (beta_prior == "normal") ? cand_state.calc_beta_norm_loglik(beta_hyperparam[0],beta_hyperparam[1]) : 0.0;
            loglik_cand_ws    = cand_state.calc_ws_loglik();

            loglik_cand_link = 0.0;
            if      (family == "binomial") loglik_cand_link = cand_state.calc_binomial_loglik(Y, X, weights);
            else if (family == "poisson")  loglik_cand_link = cand_state.calc_poisson_loglik(Y, X);

            double loglik_cand = loglik_cand_theta + loglik_cand_beta + loglik_cand_ws + loglik_cand_link;

            double alpha = std::min(1.0, exp(loglik_cand - loglik_cur));
            

            if (Rcpp::runif(1)[0] <= alpha)
            {
                cur_state = cand_state;
                
                loglik_cur = loglik_cand;
                                
                loglik_cur_theta = loglik_cand_theta;
                loglik_cur_beta  = loglik_cand_beta;
                loglik_cur_ws    = loglik_cand_ws; 
                loglik_cur_link  = loglik_cand_link;

                accept++;
                batch_accept++;
            } else {
                cand_state = cur_state;
            }
            p_amcmc.update(s, alpha);

            /////////////////////////
            // Update ws
            ////////////////////////

            arma::vec ws_jump = ws_amcmc.get_jump();
            cand_state.update_ws( ws_jump );
            cand_state.update_w();

            loglik_cand_ws   = cand_state.calc_ws_loglik();
            loglik_cand_link = 0.0;
            if      (family == "binomial") loglik_cand_link = cand_state.calc_binomial_loglik(Y, X, weights);
            else if (family == "poisson")  loglik_cand_link = cand_state.calc_poisson_loglik(Y, X);

            double alpha_ws = std::min(1.0, exp(loglik_cand_ws + loglik_cand_link - loglik_cur_ws - loglik_cur_link));
            if (Rcpp::runif(1)[0] <= alpha_ws)
            {
                cur_state = cand_state;
                
                loglik_cur += loglik_cand_ws + loglik_cand_link - loglik_cur_ws - loglik_cur_link;
                
                loglik_cur_ws    = loglik_cand_ws; 
                loglik_cur_link  = loglik_cand_link;

                accept_ws++;
                batch_accept_ws++;
            }
            ws_amcmc.update(s, alpha_ws);

            ////////////////////
            // Save Results
            ////////////////////

            w_star.col(s) = cur_state.ws;
            w.col(s) = cur_state.w;
            beta.col(s) = cur_state.beta;
            theta.col(s) = cur_state.theta;
            loglik(0,s) = loglik_cur;
            loglik(1,s) = loglik_cur_theta;
            loglik(2,s) = loglik_cur_beta;
            loglik(3,s) = loglik_cur_link;
            loglik(4,s) = loglik_cur_ws;
            loglik(5,s) = loglik_cur_e;

            if ((s+1) % n_report == 0)
            {
                if (verbose)
                {
                    report_sample(s+1, n_samples, t.toc());
                    Rcpp::Rcout << "Loglikelihood: " << loglik_cur << "\n";
                    report_accept("theta & beta", s+1, accept, batch_accept, n_report);
                    report_accept("w_star      ", s+1, accept_ws, batch_accept_ws, n_report);
                    report_line();
                }

                acc_rate.push_back(1.0*accept/s);
                acc_rate_ws.push_back(1.0*accept_ws/s);

                batch_accept = 0;
                batch_accept_ws = 0;
            }
        }
        
        accept_results["params"] = acc_rate;
        accept_results["w_star"] = acc_rate_ws;
    } 
    else
    {
        model_state_mpp_glm cur_state(&cov_settings);

        cur_state.theta = cov_settings.param_start;
        cur_state.beta  = Rcpp::as<arma::vec>(beta_settings["start"]);
        cur_state.ws    = Rcpp::as<arma::vec>(ws_settings["start"]);
        cur_state.e     = Rcpp::as<arma::vec>(e_settings["start"]);
        
        arma::wall_clock t;
        t.tic();

        for(int s = 0; s < n_samples; s++)
        {            
            model_state_mpp_glm cand_state = cur_state;

            /////////////////////////
            // Update beta & theta
            ////////////////////////

            arma::vec U = p_amcmc.get_jump();
            arma::vec beta_jump = U(arma::span(0, p-1));
            arma::vec theta_jump = U(arma::span(p, p+n_theta-1));

            cand_state.update_beta( beta_jump );
            cand_state.update_theta( theta_jump );
            
            cand_state.update_covs(knotsD, coordsKnotsD);
            cand_state.update_w();
            
            double loglik_cand_theta = cand_state.calc_theta_loglik();
            double loglik_cand_beta = (beta_prior == "normal") ? cand_state.calc_beta_norm_loglik(beta_hyperparam[0], beta_hyperparam[1]) : 0.0;    

            double loglik_cand_link = 0;
            if      (family == "binomial") loglik_cand_link = cand_state.calc_binomial_loglik(Y, X, weights);
            else if (family == "poisson")  loglik_cand_link = cand_state.calc_poisson_loglik(Y, X);
    
            double loglik_cand_ws = cand_state.calc_ws_loglik();
            double loglik_cand_e  = cand_state.calc_e_loglik();

            double loglik_cand = loglik_cand_theta + loglik_cand_beta + loglik_cand_link + loglik_cand_ws + loglik_cand_e;

            double alpha = std::min(1.0, exp(loglik_cand - loglik_cur));
            if (Rcpp::runif(1)[0] <= alpha)
            {
                cur_state = cand_state;
                
                loglik_cur       = loglik_cand;
                loglik_cur_theta = loglik_cand_theta;
                loglik_cur_beta  = loglik_cand_beta;
                loglik_cur_link  = loglik_cand_link;
                loglik_cur_ws    = loglik_cand_ws;
                loglik_cur_e     = loglik_cand_e;

                accept++;
                batch_accept++;
            } 
            else 
            {
                cand_state = cur_state;
            }
            
            p_amcmc.update(s, alpha_p);
            
            ////////////////////
            // Update ws
            ////////////////////

            cand_state.update_ws( ws_amcmc.get_jump() );
            cand_state.update_w();

            loglik_cand_ws = cand_state.calc_ws_loglik();
            loglik_cand_link = 0.0;
            if      (family == "binomial") loglik_cand_link = cand_state.calc_binomial_loglik(Y, X, weights);
            else if (family == "poisson")  loglik_cand_link = cand_state.calc_poisson_loglik(Y, X);

            double delta = loglik_cand_ws + loglik_cand_link - loglik_cur_ws - loglik_cur_link
            double alpha_ws = std::min(1.0, exp(delta));
            if (Rcpp::runif(1)[0] <= alpha_ws)
            {
                cur_state = cand_state;
                
                loglik_cur += delta;
                loglik_cur_link = loglik_cand_link;
                loglik_cur_ws = loglik_cand_ws;
                
                accept_ws++;
                batch_accept_ws++;
            }
            else
            {
                cand_state = cur_state;
            }
            
            ws_amcmc.update(s, alpha_ws);

            ////////////////////
            // Update e
            ////////////////////

            
            cand_state.update_e( e_amcmc.get_jump() );
            cand_state.update_w();

            loglik_cand_e = cand_state.calc_e_loglik();
            loglik_cand_link = 0;
            if      (family == "binomial") loglik_cand_link = cand_state.calc_binomial_loglik(Y, X, weights);
            else if (family == "poisson")  loglik_cand_link = cand_state.calc_poisson_loglik(Y, X);
  
            double delta = loglik_cand_e + loglik_cand_link - loglik_cur_e - loglik_cur_link;
            double alpha_e = std::min(1.0, exp(delta));
            
            if (Rcpp::runif(1)[0] <= alpha_e)
            {
                cur_state = cand_state;
                
                loglik_cur += delta;
                loglik_cur_link = loglik_cand_link;
                loglik_cur_e = loglik_cand_e;
                
                accept_e++;
                batch_accept_e++;
            }
            e_amcmc.update(s, alpha_e);

            ////////////////////
            // Save Results
            ////////////////////

            w_star.col(s) = cur_state.ws;
            w.col(s) = cur_state.w;
            beta.col(s) = cur_state.beta;
            theta.col(s) = cur_state.theta;
            
            loglik(0,s) = loglik_cur;
            loglik(1,s) = loglik_cur_theta;
            loglik(2,s) = loglik_cur_beta;
            loglik(3,s) = loglik_cur_link;
            loglik(4,s) = loglik_cur_ws;
            loglik(5,s) = loglik_cur_e;

            if ((s+1) % n_report == 0)
            {
                if (verbose)
                {
                    report_sample(s+1, n_samples, t.toc());
                    Rcpp::Rcout << "Loglikelihood: " << loglik_cur << "\n";
                    report_accept("theta & beta", s+1, accept,    batch_accept,    n_report);
                    report_accept("w star      ", s+1, accept_ws, batch_accept_ws, n_report);
                    report_accept("e           ", s+1, accept_e,  batch_accept_e,  n_report);
                    report_line();
                }

                acc_rate.push_back(1.0*accept/s);
                acc_rate_ws.push_back(1.0*accept_ws/s);
                acc_rate_e.push_back(1.0*accept_e/s);

                batch_accept    = 0;
                batch_accept_ws = 0;
                batch_accept_e  = 0;
            }
        }

        accept_results["params"] = acc_rate;
        accept_results["w_star"] = acc_rate_ws;
        accept_results["e"] = acc_rate_e;
    }

    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("theta") = theta,
                              Rcpp::Named("accept") = accept_results,
                              Rcpp::Named("w") = w,
                              Rcpp::Named("w_star") = w_star,
                              Rcpp::Named("loglik") = loglik,
                              Rcpp::Named("p_adapt") = p_amcmc.get_S(),
                              Rcpp::Named("ws_adapt") = ws_amcmc.get_S(),
                              Rcpp::Named("e_adapt") = e_amcmc.get_S()
                              );

//END_RCPP
}

