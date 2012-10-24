#include <iostream>
#include <string>

#include "spPPLM.hpp"
#include "util.hpp"
#include "model_state.hpp"



SEXP spmPPLM(SEXP Y_r, SEXP X_r, 
             SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
             SEXP nugget_r, 
             SEXP sigmaSqIG_r, SEXP tauSqIG_r, SEXP nuUnif_r, SEXP phiUnif_r,
             SEXP covModel_r, SEXP nSamples_r, SEXP verbose_r, SEXP nReport_r)
{
BEGIN_RCPP

    Rcpp::RNGScope scope;

    arma::vec Y = Rcpp::as<arma::vec>(Y_r);
    arma::mat X = Rcpp::as<arma::mat>(X_r);
    
    arma::mat coordsD = Rcpp::as<arma::mat>(coordsD_r);
    arma::mat knotsD = Rcpp::as<arma::mat>(knotsD_r);
    arma::mat coordsKnotsD = Rcpp::as<arma::mat>(coordsKnotsD_r);
    
    int p = X.n_cols;
    int n = X.n_rows;
    int m = knotsD.n_rows;

    //arma::vec betaStarting = Rcpp::as<arma::vec>(betaStarting_r);
  
    prior sigmaSqIG(sigmaSqIG_r);
    prior phiUnif(phiUnif_r);
    prior tauSqIG(tauSqIG_r);
    prior nuUnif(nuUnif_r);
  
    int nSamples = Rcpp::as<int>(nSamples_r);
    int verbose  = Rcpp::as<int>(verbose_r);
    int nReport  = Rcpp::as<int>(nReport_r);
  
    bool nugget = Rcpp::as<bool>(nugget_r);

    std::string cv = Rcpp::as<std::string>(covModel_r);
  
    cov_model covModel;
    if (cv == "exponential")    covModel = exp_cov;
    else if (cv == "powexp")    covModel = powexp_cov;
    else if (cv == "matern")    covModel = matern_cov;
    else if (cv == "spherical") covModel = sphere_cov;
    else if (cv == "gaussian")  covModel = gauss_cov;
    
  
    if (verbose) {
        Rcpp::Rcout << "----------------------------------------\n"
                    << "\tGeneral model description\n"
                    << "----------------------------------------\n"
                    << "Model fit with " << n << " observations.\n"
                    << "Number of covariates " << p << " (including intercept if specified).\n"
                    << "Using the " << cv << " spatial correlation model.\n"
                    << "Using modified predictive process with " << m << " knots.\n"
                    << "Number of MCMC samples " << nSamples << ".\n"
                    << "\n"
                    << "Priors and hyperpriors:\n"
                    << "\tbeta - Flat.\n"
                    << "\tsigma.sq - Starting   = " << sigmaSqIG.starting << "\n"
                    << "\t           Tuning Var = " << sigmaSqIG.tuning << "\n" 
                    << "\t           IG hyperprior shape = " << sigmaSqIG.a << "\n"
                    << "\t                     and scale = " << sigmaSqIG.b << "\n";
        
        if (nugget)
            Rcpp::Rcout << "\ttau.sq - Starting   = " << tauSqIG.starting << "\n"
                        << "\t         Tuning Var = " << tauSqIG.tuning << "\n" 
                        << "\t         IG hyperpriors shape = " << tauSqIG.a << "\n"
                        << "\t                    and scale = " << tauSqIG.b << "\n";
        else
            Rcpp::Rcout << "\ttau.sq - Fixed = 0.0\n";

        Rcpp::Rcout << "\tphi - Starting   = " << phiUnif.starting << "\n"
                    << "\t      Tuning Var = " << phiUnif.tuning << "\n" 
                    << "\t      Unif hyperpriors a = " << phiUnif.a << "\n"
                    << "\t                   and b = " << phiUnif.b << "\n";
        
        if (covModel == matern_cov)
            Rcpp::Rcout << "\tnu - Starting   = " << nuUnif.starting << "\n"
                        << "\t     Tuning Var = " << nuUnif.tuning << "\n" 
                        << "\t     Unif hyperpriors a = " << nuUnif.a << "\n"
                        << "\t                  and b = " << nuUnif.b << "\n";

        Rcpp::Rcout << "\n";
    } 
  
  
    //set starting
    int nParams = 4;
    arma::mat w(n, nSamples);
    arma::mat w_star(m, nSamples); 
    arma::mat beta(p, nSamples);
    arma::mat params(nParams, nSamples);
  
    
    /*****************************************
       Set-up MCMC alg. vars. matrices etc.
    *****************************************/
    int status=0, rtnStatus=0, accept=0, batchAccept = 0;
    
    bool first = true, accepted = true;
  
  
    if (verbose){
        Rcpp::Rcout << "-------------------------------------------------\n"
                    << "\t\tSampling\n"
                    << "-------------------------------------------------\n";
    }

    arma::mat M = arma::zeros<arma::mat>(nParams,nParams);
    M.diag() += 0.01;
    arma::mat S = arma::chol(M).t();

    model_state cur(sigmaSqIG, phiUnif, tauSqIG, nuUnif);
    cur.update_covs(covModel, knotsD, coordsKnotsD);
    cur.update_Sigma_ws();

    for (int s = 0; s < nSamples; s++) {

        cur.update_beta(X, Y);        
        
        double loglik_cur = cur.calc_loglik(covModel, nugget);


        model_state cand(cur);
        arma::vec U = S * arma::randn<arma::vec>(nParams);
        cand.update_theta(U);
        cand.update_covs(covModel, knotsD, coordsKnotsD);
        double loglik_cand = cand.calc_loglik(covModel, nugget);

        double alpha = std::min(1.0, exp(loglik_cand - loglik_cur));
        if (Rcpp::runif(1)[0] <= alpha) {
            cur = cand;
            cur.update_Sigma_ws();
        
            accept++;
            batchAccept++;
        }

        int n_adapt = 5000;
        double acc_rate = 0.234;
        double gamma = 0.5;
        if(s < n_adapt) {
            double adapt_rate = std::min(1.0, nParams * pow(s,-gamma));
            
            M = S * (arma::eye<arma::mat>(nParams,nParams) + adapt_rate*(alpha - acc_rate) * U * U.t() / arma::dot(U,U)) * S.t();
            S = arma::chol(M).t();
        }
        
        arma::vec mu_ws = cur.Sigma_ws * cur.Einv_ct_Csi.t() * cur.R;
        arma::vec ws_cur = mu_ws + cur.Sigma_ws_U.t() * arma::randn<arma::vec>(m); 
        arma::vec w_cur = cur.ct_Csi * ws_cur;

        w_star.col(s) = ws_cur;
        w.col(s) = w_cur;
        beta.col(s) = cur.beta;
        params.col(s) = cur.get_params();
    
        if (verbose && status == nReport){
            Rcpp::Rcout << "Sampled: " << s << " of " <<  nSamples << " (" << floor(1000*s/nSamples)/10 << "%)\n"
                        << "Report interval Metrop. Acceptance rate: " << floor(1000*batchAccept/nReport)/10 << "%\n"
                        << "Overall Metrop. Acceptance rate: " << floor(1000*accept/s)/10 << "%\n"
                        << "-------------------------------------------------\n";
           
            status = 0;
            batchAccept = 0;

            Rcpp::Rcout << "\n" << M
                        << "\n-------------------------------------------------\n";
        }

        status++;
    }
  
    if (verbose){
        Rcpp::Rcout << "Sampled: " << nSamples << " of " <<  nSamples << " (100%)\n"
                    << "Report interval Metrop. Acceptance rate: " << floor(1000*batchAccept/nReport)/10 << "%\n"
                    << "Overall Metrop. Acceptance rate: " << floor(1000*accept/nSamples)/10 << "%\n"
                    << "-------------------------------------------------\n";
    }

    return Rcpp::List::create(//Rcpp::Named("p.samples") = params,
                              Rcpp::Named("beta") = beta,
                              Rcpp::Named("params") = params,
                              Rcpp::Named("acceptance") = 100.0*accept/nSamples,
                              Rcpp::Named("sp.effects") = w,
                              Rcpp::Named("sp.effects.knots") = w_star,
                              Rcpp::Named("cov.jump") = M);

END_RCPP
}
