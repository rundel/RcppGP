#include <RcppArmadillo.h>

#include <boost/timer/timer.hpp>

#include "assert.hpp"
#include "cov_model.hpp"
#include "cov_funcs.hpp"
#include "distributions.hpp"
#include "transforms.hpp"


#ifdef USE_GPU

#include <magma.h>

#include "gpu_mat.hpp"
#include "gpu_util.hpp"
#include "init.hpp"

#endif

cov_model::cov_model(SEXP covModel_r)
{
    Rcpp::List cov_model_opts(covModel_r);

    nmodels = Rcpp::as<int>(cov_model_opts["nmodels"]);
    nparams = Rcpp::as<int>(cov_model_opts["nparams"]);

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

    param_nfree  = Rcpp::as<int>(cov_model_opts["param_nfree"]);
    param_nfixed = Rcpp::as<int>(cov_model_opts["param_nfixed"]);
    
    param_free_index = Rcpp::as<arma::uvec>(cov_model_opts["param_free_index"]) - 1;

}


arma::mat cov_model::calc_cov(arma::mat const& d, arma::vec const& params) const
{
    RT_ASSERT(nparams == params.n_elem, "Number of given parameters does not match the number expected.");
    
    arma::mat cov = arma::zeros<arma::mat>(d.n_rows, d.n_cols);

    for (int i=0; i != nmodels; ++i)
    {
        arma::vec mparams = params.elem( model_params[i] );
        int type = model_funcs[i];
        
        cov += cov_func(type,d,mparams);
    }

    return cov;
}

arma::mat cov_model::calc_inv_cov(arma::mat const& d, arma::vec const& params) const
{
    return arma::inv(arma::sympd(calc_cov(d, params)));
}

#ifdef USE_GPU

double* cov_model::calc_cov_gpu_ptr(gpu_mat const& d, arma::vec const& params) const
{
    RT_ASSERT(nparams == params.n_elem, "Number of given parameters does not match the number expected.");

    int m = d.n_rows;
    int n = d.n_cols;

    gpu_mat cov(m,n,0.0);

    for (int i=0; i != nmodels; ++i)
    {
        arma::vec mparams = params.elem( model_params[i] );
        int type = model_funcs[i];

        cov_func_gpu(type, d.mat, cov.mat, m, n, 64, mparams);
    }

    return cov.get_ptr();
}

arma::mat cov_model::calc_cov_gpu(gpu_mat const& d, arma::vec const& params) const
{
    int m = d.n_rows;
    int n = d.n_cols;

    gpu_mat cov( calc_cov_gpu_ptr(d, params), m, n );

    return cov.get_mat();
}

double* cov_model::calc_inv_cov_gpu_ptr(gpu_mat const& d, arma::vec const& params) const
{
    RT_ASSERT(nparams == params.n_elem, "Number of given parameters does not match the number expected.");
    RT_ASSERT(d.n_rows == d.n_cols, "Cannot invert non-square covariance matrix.");

    int n = d.n_rows;
      
    gpu_mat A(n,n,0.0);

    for (int i=0; i != nmodels; ++i)
    {
        arma::vec mparams = params.elem( model_params[i] );
        int type = model_funcs[i];

        cov_func_gpu(type, d.mat, A.mat, n, n, 64, mparams);
    }

    inv_sympd(A);

    return A.get_ptr();
} 

arma::mat cov_model::calc_inv_cov_gpu(gpu_mat const& d, arma::vec const& params) const
{
    int m = d.n_rows;
    int n = d.n_cols;

    gpu_mat cov( calc_inv_cov_gpu_ptr(d, params), m, n );

    return cov.get_mat();
}

#endif
