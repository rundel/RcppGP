#include <RcppArmadillo.h>

#include <boost/timer/timer.hpp>

#include "assert.hpp"
#include "cov_model.hpp"
#include "cov_funcs.hpp"
//#include "distributions.hpp"
//#include "transforms.hpp"

#include "gpu_mat.hpp"

#ifdef USE_GPU
#include <magma.h>
#include "gpu_util.hpp"
#endif

cov_model::cov_model(Rcpp::List opts)
{
    nmodels = Rcpp::as<int>(opts["nmodels"]);
    nparams = Rcpp::as<int>(opts["nparams"]);

    model_nparams = Rcpp::as<std::vector<int> >(opts["model_nparams"]);
    model_names   = Rcpp::as<std::vector<std::string> >(opts["model_names"]);

    std::vector<std::string> funcs = Rcpp::as< std::vector<std::string> >(opts["model_funcs"]);

    Rcpp::List mp = Rcpp::as<Rcpp::List>(opts["model_params"]);
    for (int i=0; i != nmodels; ++i) {
        model_funcs.push_back( cov_func_map::from_string(funcs[i]) );
        model_params.push_back( Rcpp::as<arma::uvec>(mp[i]) - 1 ); // adjust for R and C++ offset difference
    }


    //param_names  = Rcpp::as<std::vector<std::string> >(opts["param_names"]);
    //param_start  = Rcpp::as<arma::vec>(opts["param_start"]);
    //param_tuning = Rcpp::as<arma::vec>(opts["param_tuning"]);
    //param_nhyper = Rcpp::as<std::vector<int> >(opts["param_nhyper"]);

    //std::vector<std::string> dists = Rcpp::as<std::vector<std::string> >(opts["param_dists"]);
    //std::vector<std::string> trans = Rcpp::as<std::vector<std::string> >(opts["param_trans"]);
    //Rcpp::List hyp = Rcpp::as<Rcpp::List>(opts["param_hyper"]);

    //for (int i=0; i != nparams; ++i) {
    //    param_dists.push_back( param_dists_map::from_string(dists[i]) );
    //    param_trans.push_back( param_trans_map::from_string(trans[i]) );
    //
    //    param_hyper.push_back( Rcpp::as<arma::vec>(hyp[i]) );
    //}

    //param_nfree  = Rcpp::as<int>(opts["param_nfree"]);
    //param_nfixed = Rcpp::as<int>(opts["param_nfixed"]);

    //param_free_index = Rcpp::as<arma::uvec>(opts["param_free_index"]) - 1;
}

///
/// Calculate covariance matrix
///

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

double* cov_model::calc_cov_gpu_ptr(gpu_mat const& d, arma::vec const& params) const
{
    RT_ASSERT(nparams == params.n_elem, "Number of given parameters does not match the number expected.");

    int m = d.get_n_rows();
    int n = d.get_n_cols();

    gpu_mat cov(m,n,0.0);

    for (int i=0; i != nmodels; ++i)
    {
        arma::vec mparams = params.elem( model_params[i] );
        int type = model_funcs[i];

        cov_func_gpu(type, d.get_const_ptr(), cov.get_ptr(), m, n, 64, mparams);
    }

    return cov.get_gpu_mat();
}

arma::mat cov_model::calc_cov_gpu(gpu_mat const& d, arma::vec const& params) const
{
    int m = d.get_n_rows();
    int n = d.get_n_cols();

    gpu_mat cov( calc_cov_gpu_ptr(d, params), m, n );

    return cov.get_mat();
}


///
/// Calculate covariance matrix inverse
///

arma::mat cov_model::calc_inv_cov(arma::mat const& d, arma::vec const& params) const
{
    return arma::inv_sympd(calc_cov(d, params));
}


double* cov_model::calc_inv_cov_gpu_ptr(gpu_mat const& d, arma::vec const& params) const
{
    RT_ASSERT(nparams == params.n_elem, "Number of given parameters does not match the number expected.");
    RT_ASSERT(d.get_n_rows() == d.get_n_cols(), "Cannot invert non-square covariance matrix.");

    int n = d.get_n_rows();

    gpu_mat A(n,n,0.0);

    for (int i=0; i != nmodels; ++i)
    {
        arma::vec mparams = params.elem( model_params[i] );
        int type = model_funcs[i];

        cov_func_gpu(type, d.get_const_ptr(), A.get_ptr(), n, n, 64, mparams);
    }

    A.inv_sympd();

    return A.get_gpu_mat();
}

arma::mat cov_model::calc_inv_cov_gpu(gpu_mat const& d, arma::vec const& params) const
{
    int m = d.get_n_rows();
    int n = d.get_n_cols();

    gpu_mat cov( calc_inv_cov_gpu_ptr(d, params), m, n );

    return cov.get_mat();
}



///
/// Calculate covariance matrix inverse
///


void cov_model::calc_cov_low_rank(arma::mat const& d, arma::vec const& params, 
                                  arma::mat& U, arma::vec& C,
                                  int rank, int over_samp, int qr_iter) const
{
    arma::mat Sigma = calc_cov(d, params);

    int l = rank + over_samp;
    int n = Sigma.n_cols;

    arma::mat Q, Q_tilde, R;
    RT_ASSERT(arma::qr_econ(Q, R, Sigma * arma::randn(n, l)), "QR failed.");

    for(int i=1; i<qr_iter; ++i)
    {
        RT_ASSERT(arma::qr_econ(Q_tilde, R, Sigma.t()*Q), "QR failed.");
        RT_ASSERT(arma::qr_econ(Q, R, Sigma*Q_tilde), "QR failed.");
    }

    Q = Q.cols(0,rank-1);

    RT_ASSERT(arma::eig_sym(C, U, Q.t() * Sigma * Q), "Eig_sym failed.");
    U = Q * U;
}
