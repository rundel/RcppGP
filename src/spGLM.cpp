#include <boost/timer/timer.hpp>

#include "assert.hpp"
#include "report.hpp"
#include "spGLM.hpp"
#include "distributions.hpp"
#include "adapt_mcmc.hpp"
#include "model_state_glm.hpp"

SEXP spGLM(SEXP Y_r, SEXP X_r,
           SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
           SEXP family_r, SEXP weights_r,
           SEXP cov_model_r,
           SEXP theta_r, SEXP beta_r, SEXP w_r, SEXP ws_r, SEXP e_r,
           SEXP is_pp_r, SEXP is_mod_pp_r,
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

    bool is_pp = Rcpp::as<bool>(is_pp_r);
    bool is_mod_pp = Rcpp::as<bool>(is_mod_pp_r);

    cov_model cov_settings(cov_model_r);

    Rcpp::List theta_settings(theta_r);
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
    
    int n_loglik = (!is_pp) ? 4 : (!is_mod_pp) ? 5 : 6; 
    arma::mat loglik(n_loglik, n_samples);

    if (verbose) 
        report_start();

    vihola_adapt theta_amcmc(theta_settings);
    vihola_adapt beta_amcmc(beta_settings);
    vihola_adapt ws_amcmc(ws_settings);
    vihola_ind_adapt e_amcmc(e_settings);

    Rcpp::List accept_results;

    if (is_pp)
    {
        model_state_glm cur_state(&cov_settings, is_pp, is_mod_pp);

        cur_state.theta = Rcpp::as<arma::vec>(theta_settings["start"]);
        cur_state.beta = Rcpp::as<arma::vec>(beta_settings["start"]);
        cur_state.ws = Rcpp::as<arma::vec>(ws_settings["start"]);
        if (is_mod_pp)
            cur_state.e = Rcpp::as<arma::vec>(e_settings["start"]);

        //cur_state.calc_theta_loglik();
        //cur_state.calc_beta_loglik(beta_prior, beta_hyperparam);
        //cur_state.calc_link_loglik(family, Y, X, weights);
        //cur_state.calc_ws_loglik();
        //if (is_mod_pp)
        //    cur_state.calc_e_loglik();
        //cand_state.calc_loglik();

        int accept_theta = 0, batch_accept_theta = 0;
        int accept_beta  = 0, batch_accept_beta  = 0;
        int accept_ws    = 0, batch_accept_ws    = 0;
        std::vector<double> acc_rate_theta, acc_rate_beta, acc_rate_ws;

        arma::uvec accept_e  = arma::zeros<arma::uvec>(n);
        arma::uvec batch_accept_e = arma::zeros<arma::uvec>(n);
        std::vector<arma::vec> acc_rate_e;

        for(int s = 0; s < n_samples; s++)
        {            
            model_state_glm cand_state = cur_state;

            /////////////////////////
            // Update theta
            ////////////////////////
            
            cand_state.update_theta( theta_amcmc.get_jump() );
            cand_state.update_covs(knotsD, coordsKnotsD);
            cand_state.update_w();
            
            cand_state.calc_theta_loglik();
            cand_state.calc_beta_loglik(beta_prior, beta_hyperparam);
            cand_state.calc_link_loglik(family, Y, X, weights);
            cand_state.calc_ws_loglik();
            if (is_mod_pp)
                cand_state.calc_e_loglik();
            cand_state.calc_loglik();
            
            double alpha_theta = std::min(1.0, exp(cand_state.loglik - cur_state.loglik));
            if (Rcpp::runif(1)[0] <= alpha_theta)
            {
                cur_state = cand_state;
                accept_theta++;
                batch_accept_theta++;
            } 
            else 
            {
                cand_state = cur_state;
            }
            
            theta_amcmc.update(s, alpha_theta);
            
            /////////////////////////
            // Update beta
            ////////////////////////

            cand_state.update_beta( beta_amcmc.get_jump() );

            cand_state.calc_beta_loglik(beta_prior, beta_hyperparam);
            cand_state.calc_link_loglik(family, Y, X, weights);
            
            double delta_beta =  cand_state.loglik_beta + arma::accu(cand_state.loglik_link)
                                - cur_state.loglik_beta - arma::accu(cur_state.loglik_link);

            double alpha_beta = std::min(1.0, exp(delta_beta));
            if (Rcpp::runif(1)[0] <= alpha_beta)
            {
                cand_state.loglik += delta_beta;
                cur_state = cand_state;

                accept_beta++;
                batch_accept_beta++;
            }
            else
            {
                cand_state = cur_state;
            }

            beta_amcmc.update(s, alpha_beta);


            ////////////////////
            // Update ws
            ////////////////////

            cand_state.update_ws( ws_amcmc.get_jump() );
            cand_state.update_w();

            cand_state.calc_ws_loglik();
            cand_state.calc_link_loglik(family, Y, X, weights);

            double delta_ws =  cand_state.loglik_ws + arma::accu(cand_state.loglik_link)
                              - cur_state.loglik_ws - arma::accu(cur_state.loglik_link);
            double alpha_ws = std::min(1.0, exp(delta_ws));
            if (Rcpp::runif(1)[0] <= alpha_ws)
            {
                cand_state.loglik += delta_ws;
                cur_state = cand_state;

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

            if (is_mod_pp)
            {


                cand_state.update_e( e_amcmc.get_jump() );
                cand_state.update_w();

                cand_state.calc_e_loglik();
                cand_state.calc_link_loglik(family, Y, X, weights);

                arma::vec delta_e =  cand_state.loglik_e + cand_state.loglik_link 
                                    - cur_state.loglik_e - cur_state.loglik_link;
                
                arma::vec alpha_e = arma::exp(delta_e);
                
                arma::vec U = arma::randu<arma::vec>(n);
                for(int i=0; i!=n; ++i)
                {
                    if (alpha_e[i] > 1.0)
                        alpha_e[i] = 1.0;

                    if (U[i] <= alpha_e[i])
                    {
                        cur_state.e[i] = cand_state.e[i];
                        cur_state.w[i] = cand_state.w[i];
                        
                        cur_state.loglik += delta_e[i];
                        cur_state.loglik_link[i] = cand_state.loglik_link[i];
                        cur_state.loglik_e[i]    = cand_state.loglik_e[i];
                        
                        accept_e[i]++;
                        batch_accept_e[i]++;
                    }
                }

                e_amcmc.update(s, alpha_e);
            }

            ////////////////////
            // Save Results
            ////////////////////

            w_star.col(s) = cur_state.ws;
            w.col(s) = cur_state.w;
            beta.col(s) = cur_state.beta;
            theta.col(s) = cur_state.theta;

            loglik(0,s) = cur_state.loglik;
            loglik(1,s) = cur_state.loglik_theta;
            loglik(2,s) = cur_state.loglik_beta;
            loglik(3,s) = arma::accu(cur_state.loglik_link);
            loglik(4,s) = cur_state.loglik_ws;
            if (is_mod_pp)
                loglik(5,s) = arma::accu(cur_state.loglik_e);

            if ((s+1) % n_report == 0)
            {
                if (verbose)
                {
                    long double wall_sec = timer.elapsed().wall / 1000000000.0L;

                    report_sample(s+1, n_samples, wall_sec);//t.toc());
                    Rcpp::Rcout << "Log likelihood: " << cur_state.loglik << "\n";
                    report_accept("theta : ", s+1, accept_theta, batch_accept_theta, n_report);
                    report_accept("beta  : ", s+1, accept_beta,  batch_accept_beta,  n_report);
                    report_accept("w*    : ", s+1, accept_ws,    batch_accept_ws,    n_report);
                    if (is_mod_pp)
                        report_accept("e     : ", (s+1), arma::mean(accept_e),  arma::mean(batch_accept_e),  n_report);
                    Rcpp::Rcout << "Beta Prop: \n" << beta_amcmc.get_S();
                    Rcpp::Rcout << "Theta Prop: \n" << theta_amcmc.get_S();

                    report_line();
                }

                acc_rate_theta.push_back(1.0*accept_theta/s);
                acc_rate_beta.push_back(1.0*accept_beta/s);
                acc_rate_ws.push_back(1.0*accept_ws/s);
                
                batch_accept_theta = 0;
                batch_accept_beta = 0;
                batch_accept_ws    = 0;
                
                if (is_mod_pp)
                {
                    acc_rate_e.push_back(arma::conv_to<arma::vec>::from(accept_e) / s);
                    batch_accept_e.fill(0);
                }
            }
        }

        accept_results["theta"] = acc_rate_theta;
        accept_results["beta"] = acc_rate_beta;
        accept_results["w_star"] = acc_rate_ws;
        if (is_mod_pp)
            accept_results["e"] = acc_rate_e;
    } 
    
    Rcpp::List results;

    results["beta"]     = beta;
    results["theta"]    = theta;
    results["accept"]   = accept_results;
    results["w"]        = w;
    results["w_star"]   = w_star;
    results["loglik"]   = loglik;
    results["theta_adapt"] = theta_amcmc.get_S();
    results["beta_adapt"]  = beta_amcmc.get_S();
    results["ws_adapt"]    = ws_amcmc.get_S();
    if (is_mod_pp)
        results["e_adapt"] = e_amcmc.get_S();

    return results;

//END_RCPP
}

