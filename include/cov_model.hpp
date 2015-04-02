#ifndef COV_MODEL_HPP
#define COV_MODEL_HPP

#include "gpu_mat.hpp"

struct cov_model
{
    int nmodels;
    int nparams;

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

    int param_nfree;
    int param_nfixed;

    arma::uvec param_free_index;

    cov_model(Rcpp::List model);

    arma::mat calc_cov(arma::mat const& d, arma::vec const& params) const;
    gpu_mat   calc_cov(gpu_mat   const& d, arma::vec const& params) const;

    //arma::mat calc_inv_cov(arma::mat const& d, arma::vec const& params) const;
    //arma::mat calc_inv_cov_gpu(gpu_mat const& d, arma::vec const& params) const;
    //double*   calc_inv_cov_gpu_ptr(gpu_mat const& d, arma::vec const& params) const;

    //void calc_cov_low_rank(arma::mat const& d, arma::vec const& params,
    //                       arma::mat& U, arma::vec& C,
    //                       int rank, int over_samp, int qr_iter) const;
};


#endif