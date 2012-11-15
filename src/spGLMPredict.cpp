#include "spGLMPredict.hpp"

#include "assert.hpp"
#include "report.hpp"
#include "cov_model.hpp"


SEXP spGLMPredict(SEXP obj_r, 
                  SEXP pred_X_r, SEXP pred_D_r, SEXP between_D_r, 
                  SEXP verbose_r, SEXP n_report_r) 
{
BEGIN_RCPP

    Rcpp::RNGScope scope;

    Rcpp::List obj(obj_r);

    bool verbose = Rcpp::as<bool>(verbose_r);
    int n_report = Rcpp::as<int>(n_report_r);

    std::string family = Rcpp::as<std::string>(obj["family"]);
    
    bool is_pp = Rcpp::as<bool>(obj["is_pp"]);
    bool is_mod_pp = Rcpp::as<bool>(obj["modified_pp"]);

    arma::vec Y = Rcpp::as<arma::vec>(obj["X"]);
    arma::mat X = Rcpp::as<arma::mat>(obj["Y"]);    
    int p = X.n_cols;
    int n = X.n_rows;

    arma::mat beta = Rcpp::as<arma::mat>(obj["beta"]);
    arma::mat theta = Rcpp::as<arma::mat>(obj["theta"]);
    
    int n_samples = beta.n_cols;
    RT_ASSERT(n_samples == theta.n_cols,  "Inconsistent number of samples between beta and theta.");


    arma::mat pred_X = Rcpp::as<arma::mat>(pred_X_r);
    RT_ASSERT(p == pred_X.n_cols,  "Inconsistent number of predictors in pred X.");
    int q = pred_X.n_rows;
    
    cov_model cm(Rcpp::as<Rcpp::List>(obj["cov_model"]));
     
    if (verbose) 
        report_start_predict();
    
    arma::mat w_pred(q, n_samples);
    arma::mat y_pred(q, n_samples);

    arma::mat between_D = Rcpp::as<arma::mat>(between_D_r);

    if (!is_pp)
    {
        arma::mat w      = Rcpp::as<arma::mat>(obj["w"]);
        arma::mat obs_D  = Rcpp::as<arma::mat>(obj["coord_D"]);
        arma::mat pred_D = Rcpp::as<arma::mat>(pred_D_r);
        
        RT_ASSERT(n_samples == w.n_cols, "Inconsistent number of samples between beta and w.");
        RT_ASSERT(q == pred_D.n_rows, "Inconsistent number of prediction locations.");

        for (int s = 0; s < n_samples; s++) {
            
            arma::vec cur_params = theta.col(s);
            arma::mat obs_cov = cm.calc_cov(obs_D, cur_params);
            arma::mat obs_cov_inv = arma::inv(arma::sympd(obs_cov));
            
            arma::mat between_cov = cm.calc_cov(between_D, cur_params);
            arma::mat pred_cov = cm.calc_cov(pred_D, cur_params);


            arma::mat btw_obsi = between_cov * obs_cov_inv;

            arma::vec mu = btw_obsi * w.col(s);
            arma::mat sigma = pred_cov - btw_obsi * between_cov.t();

            w_pred.col(s) = mu + arma::chol(sigma).t() * arma::randn<arma::vec>(n); 

            arma::vec XB = pred_X * beta.col(s);

            if (family == "binomial")      y_pred.col(s) = 1.0/(1.0 + arma::exp(-1.0 * (XB + w_pred.col(s)))); 
            else if (family == "poisson")  y_pred.col(s) = arma::exp(XB+w_pred.col(s));
            else if (family == "identity") y_pred.col(s) = XB+w_pred.col(s);    
            else throw std::runtime_error("Family misspecification in spGLMPredict\n");
        

            if (verbose && (s+1) % n_report == 0) 
                report_predict(s+1, n_samples);
        }
    }
    else //predictive process prediction
    {
        arma::mat w_star  = Rcpp::as<arma::mat>(obj["w_star"]);
        arma::mat knots_D = Rcpp::as<arma::mat>(obj["knots_D"]);        
        int m = knots_D.n_rows;

        RT_ASSERT(n_samples == w_star.n_cols, "Inconsistent number of samples between beta and w_star.");

        for (int s = 0; s < n_samples; s++)
        {    
            //got w* above now get the mean components MVN(XB + ct C^{*-1} w* + \tild{\eps}, (sigma^2 - ct C^{*-1} c)^{-1})
            
            arma::vec cur_params = theta.col(s);
            
            arma::mat Cs = cm.calc_cov(knots_D, cur_params);
            arma::mat c = cm.calc_cov(between_D, cur_params);
            arma::vec C_diag = cm.calc_cov(arma::zeros<arma::vec>(q), cur_params);
            
            arma::mat c_Csi = c * arma::inv(arma::sympd(Cs));
            arma::mat c_Csi_ct = c_Csi * c.t();
            
            w_pred.col(s) = c_Csi * w_star.col(s);

            if (is_mod_pp)
                w_pred.col(s) += arma::randn<arma::vec>(q) % arma::sqrt(C_diag - arma::diagvec(c_Csi_ct));
            
            arma::vec XB = pred_X * beta.col(s);
    
            if (family == "binomial")      y_pred.col(s) = 1.0/(1.0 + arma::exp(-1.0 * (XB + w_pred.col(s))));
            else if (family == "poisson")  y_pred.col(s) = arma::exp(XB + w_pred.col(s));
            else if (family == "identity") y_pred.col(s) = XB + w_pred.col(s);
            else throw std::runtime_error("Family misspecification in spGLMPredict\n"); 
     
            if (verbose && (s+1) % n_report == 0) 
                report_predict(s+1, n_samples);
        }
    }

    return Rcpp::List::create(Rcpp::Named("w_pred") = w_pred,
                              Rcpp::Named("y_pred") = y_pred
                             );
END_RCPP
}

