#ifndef COV_MODEL_HPP
#define COV_MODEL_HPP

#include "cov_method.hpp"

struct cov_model 
{
    int nmodels;
    int nparams;
    int method;
    
    std::vector<int>            model_nparams;
    std::vector<std::string>    model_names;
    std::vector<int>            model_funcs;
    std::vector<arma::uvec>     model_params;
    
    std::vector<std::string>    param_names;        
    arma::vec                   param_start;
    arma::vec                   param_tuning;
    std::vector<int>            param_nhyper;
    std::vector<int>            param_dists;
    std::vector<int>            param_trans;
    std::vector<arma::vec>      param_hyper;

    cov_model(SEXP covModel_r);
    
    arma::mat calc_cov(arma::mat const& d, arma::vec const& params);
    template<int i> arma::mat calc_cov(arma::mat const& d, arma::vec const& params);
};

RcppExport SEXP test_calc_cov(SEXP model, SEXP dist, SEXP params);

#endif