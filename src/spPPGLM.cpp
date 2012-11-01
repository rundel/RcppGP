#include <limits>

#include "spPPGLM.hpp"
#include "distributions.hpp"
#include "model_state_glm.hpp"

SEXP spPPGLM(SEXP Y_r, SEXP X_r,
             SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
             SEXP family_r, SEXP weights_r,
             SEXP is_mod_pp_r,
             SEXP beta_prior_r, SEXP beta_start_r, SEXP beta_tuning_r, SEXP beta_mu_r, SEXP beta_sd_r,
             SEXP ws_tuning_r,
             SEXP cov_model_r,
             SEXP n_samples_r, SEXP verbose_r, SEXP n_report_r,
             SEXP n_adapt_r, SEXP target_acc_r, SEXP gamma_r)
{    
BEGIN_RCPP

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

    std::string beta_prior = Rcpp::as<std::string>(beta_prior_r);
    arma::vec beta_start = Rcpp::as<arma::vec>(beta_start_r);
    arma::vec beta_tuning = Rcpp::as<arma::vec>(beta_tuning_r);
    arma::vec beta_mu = Rcpp::as<arma::vec>(beta_mu_r);
    arma::vec beta_sd = Rcpp::as<arma::vec>(beta_sd_r);

    arma::vec ws_tuning = Rcpp::as<arma::vec>(ws_tuning_r);

    cov_model settings(cov_model_r);

    int n_samples = Rcpp::as<int>(n_samples_r);
    int verbose  = Rcpp::as<int>(verbose_r);
    int n_report  = Rcpp::as<int>(n_report_r);

    int n_params = settings.nparams;
    int n_total_p = n_params + p;


    // Adaption Settings
    int n_adapt = Rcpp::as<int>(n_adapt_r); //5000;
    double target_acc = Rcpp::as<double>(target_acc_r); //0.234;
    double gamma = Rcpp::as<double>(gamma_r); // 0.5;

    if(verbose){
        Rcpp::Rcout << "----------------------------------------\n";
        Rcpp::Rcout << "\tGeneral model description\n";
        Rcpp::Rcout << "----------------------------------------\n";
        Rcpp::Rcout << "Model fit with " << n << " observations.\n";
        Rcpp::Rcout << "Number of covariates " << p << " (including intercept if specified).\n";
        //Rcpp::Rcout << "Using the " << covModel << " spatial correlation model.\n";
        
        std::string mod_str = (is_mod_pp) ? "modified" : "non-modified"; 
        Rcpp::Rcout << "Using " << mod_str <<  " predictive process with %i knots.\n";
        Rcpp::Rcout << "Number of MCMC samples " << n_samples << ".\n\n";

        Rcpp::Rcout << "Priors and hyperpriors:\n";
        
        // FIXME
    } 



    arma::mat w(n, n_samples);
    arma::mat w_star(m, n_samples); 
    arma::mat beta(p, n_samples);
    arma::mat params(n_params, n_samples);



    if (verbose) {
        Rcpp::Rcout << "-------------------------------------------------\n"
                    << "                   Sampling                      \n"
                    << "-------------------------------------------------\n";
    }


    arma::mat M = arma::diagmat( arma::join_cols(beta_tuning, settings.param_tuning) );
    arma::mat S = arma::chol(M).t();

    model_state_glm cur_state(&settings, beta_start);

    double loglik_cur = std::numeric_limits<double>::min();

    int status=0, accept=0, batch_accept = 0;
    for(int s = 0; s < n_samples; s++){

        model_state_glm cand_state = cur_state;

        arma::vec U = S * arma::randn<arma::mat>(p+n_params);

        arma::vec beta_jump  = U(arma::span(0,p-1));
        arma::vec param_jump = U(arma::span(p,p+n_params-1));

        // Propose
        cand_state.update_beta( beta_jump );
        cand_state.update_params( param_jump );
        cand_state.update_ws( sqrt(ws_tuning) % arma::randn<arma::vec>(m) );
        cand_state.update_covs(knotsD, coordsKnotsD);
        cand_state.update_w();

        // Log Likelihood
        double loglik_cand = 0.0;

        loglik_cand += cand_state.calc_param_loglik();
        loglik_cand += cand_state.calc_mvn_loglik();

        if (beta_prior == "normal") {
            loglik_cand += arma::accu( -arma::log(beta_sd) - arma::pow((cand_state.beta-beta_mu)/beta_sd,2)/2 );    
        }

        if (family == "binomial") {
            loglik_cand += binomial_logpost(Y, X*cand_state.beta, cand_state.w, weights);
        } else if (family == "poisson") {
            loglik_cand += poisson_logpost(Y, X*cand_state.beta, cand_state.w);
        } else {
            throw std::invalid_argument("Unknown model family: " + family + ".");
        }

        double alpha = std::min(1.0, exp(loglik_cand - loglik_cur));
        if (Rcpp::runif(1)[0] <= alpha)
        {
            cur_state = cand_state;
            loglik_cur = loglik_cand;
            accept++;
            batch_accept++;
        }

        if(s < n_adapt) {
            double adapt_rate = std::min(1.0, n_total_p * pow(s,-gamma));
            
            M = S * (arma::eye<arma::mat>(n_total_p,n_total_p) + adapt_rate*(alpha - target_acc) * U * U.t() / arma::dot(U,U)) * S.t();
            S = arma::chol(M).t();
        }

        w_star.col(s) = cur_state.ws;
        w.col(s) = cur_state.w;
        beta.col(s) = cur_state.beta;
        params.col(s) = cur_state.params;

        if (verbose && status == n_report){
            Rcpp::Rcout << "Sampled: " << s << " of " <<  n_samples << " (" << floor(1000*s/n_samples)/10 << "%)\n"
                        << "Report interval Metrop. Acceptance rate: " << floor(1000*batch_accept/n_report)/10 << "%\n"
                        << "Overall Metrop. Acceptance rate: " << floor(1000*accept/s)/10 << "%\n"
                        << "-------------------------------------------------\n";
           
            status = 0;
            batch_accept = 0;

            Rcpp::Rcout << "\n" << M
                        << "\n-------------------------------------------------\n";
        }

        status++;
    }
  
    if (verbose){
        Rcpp::Rcout << "Sampled: " << n_samples << " of " <<  n_samples << " (100%)\n"
                    << "Report interval Metrop. Acceptance rate: " << floor(1000*batch_accept/n_report)/10 << "%\n"
                    << "Overall Metrop. Acceptance rate: " << floor(1000*accept/n_samples)/10 << "%\n"
                    << "-------------------------------------------------\n";
    }

    return Rcpp::List::create(//Rcpp::Named("p.samples") = params,
                              Rcpp::Named("beta") = beta,
                              Rcpp::Named("params") = params,
                              Rcpp::Named("acceptance") = 100.0*accept/n_samples,
                              Rcpp::Named("sp.effects") = w,
                              Rcpp::Named("sp.effects.knots") = w_star,
                              Rcpp::Named("cov.jump") = M);

END_RCPP
}

