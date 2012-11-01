#include <RcppArmadillo.h>

#include "assert.hpp"
#include "cov_model.hpp"
#include "cov_funcs.hpp"
#include "cov_method.hpp"
#include "distributions.hpp"
#include "transforms.hpp"


cov_model::cov_model(SEXP covModel_r)
{
    Rcpp::List cov_model_opts(covModel_r);

    nmodels = Rcpp::as<int>(cov_model_opts["nmodels"]);
    nparams = Rcpp::as<int>(cov_model_opts["nparams"]);
    
    method = cov_method_map::from_string( Rcpp::as<std::string>(cov_model_opts["method"]) );

    model_nparams = Rcpp::as<std::vector<int> >(cov_model_opts["model_nparams"]);
    model_names = Rcpp::as<std::vector<std::string> >(cov_model_opts["model_names"]);

    std::vector<std::string> funcs = Rcpp::as<std::vector<std::string> >(cov_model_opts["model_funcs"]);   
    Rcpp::List mp = Rcpp::as<Rcpp::List>(cov_model_opts["model_params"]);
    for (int i=0; i != nmodels; ++i) {
        model_funcs.push_back( cov_func_map::from_string(funcs[i]) );
        model_params.push_back( Rcpp::as<arma::uvec>(mp[i]) - 1 ); // adjust for R and C++ offset difference
    }


    param_names  = Rcpp::as<std::vector<std::string> >(cov_model_opts["param_names"]);
    param_start  = Rcpp::as<arma::vec>(cov_model_opts["param_start"]);
    param_tuning = Rcpp::as<arma::vec>(cov_model_opts["param_tuning"]);
    param_nhyper = Rcpp::as<std::vector<int> >(cov_model_opts["param_nhyper"]);

    std::vector<std::string> dists = Rcpp::as<std::vector<std::string> >(cov_model_opts["param_dists"]);
    std::vector<std::string> trans = Rcpp::as<std::vector<std::string> >(cov_model_opts["param_trans"]);
    Rcpp::List hyp = Rcpp::as<Rcpp::List>(cov_model_opts["param_hyper"]);
    
    for (int i=0; i != nparams; ++i) {
        param_dists.push_back( param_dists_map::from_string(dists[i]) );
        param_trans.push_back( param_trans_map::from_string(trans[i]) );

        param_hyper.push_back( Rcpp::as<arma::vec>(hyp[i]) );
    }
}

template<>
arma::mat cov_model::calc_cov<add_method>(arma::mat const& d, arma::vec const& params)
{
    arma::mat cov = arma::zeros<arma::mat>(d.n_rows, d.n_cols);

    for (int i=0; i != nmodels; ++i)
    {
        arma::vec mparams = params.elem( model_params[i] );
        int type = model_funcs[i];
        
        cov += cov_func(type,d,mparams);
    }

    return cov;
}

template<>
arma::mat cov_model::calc_cov<prod_method>(arma::mat const& d, arma::vec const& params)
{
    arma::mat cov = arma::ones<arma::mat>(d.n_rows, d.n_cols);
                      
    for (int i=0; i != nmodels; ++i)
    {
        arma::vec mparams = params.elem( model_params[i] );
        int type = model_funcs[i];

        cov = cov % cov_func(type,d,mparams);
    }

    return cov;
}

arma::mat cov_model::calc_cov(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(nparams == params.n_elem, "Number of given parameters does not match the number expected.");

    if      (method == add_method ) return calc_cov<add_method>(d, params);
    else if (method == prod_method) return calc_cov<prod_method>(d, params);
    else    throw std::range_error("Unknown covariance model construction method.");
}

SEXP test_calc_cov(SEXP model, SEXP dist, SEXP params)
{
BEGIN_RCPP

    cov_model m(model);
    arma::mat d = Rcpp::as<arma::mat>(dist);
    arma::vec p = Rcpp::as<arma::vec>(params);

    return Rcpp::wrap(m.calc_cov(d,p));

END_RCPP
}
