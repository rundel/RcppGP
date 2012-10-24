#ifndef COV_MODEL_HPP
#define COV_MODEL_HPP

#include "cov_funcs.hpp"

enum cov_method {add_method, prod_method};




struct cov_model 
{

    int nmodels;
    int nparams;

    cov_method method;
    std::vector<arma::ivec> model_params;
    Rcpp::IntergerVector model_types;

    cov_model(SEXP method_r, SEXP model_params, SEXP model_types)
    {

    }

    cov_method method_from_string(std::string const& str) 
    {
        if      (str == "nugget"             ) return nugget_cov;
        else if (str == "exponential"        ) return exp_cov;
        else if (str == "gaussian"           ) return gauss_cov;
        else if (str == "powered_exponential") return powexp_cov;
        else if (str == "spherical"          ) return sphere_cov;
        else if (str == "matern"             ) return matern_cov;
        else if (str == "rational_quadratic" ) return rq_cov;
        else if (str == "periodic"           ) return periodic_cov;
        else throw std::range_error("Unknown covariance function type: " + str + ".");

        return -1;
    }

    arma::mat calc_covariance(arma::mat const& d, arma::vec const& params)
    {
        arma::mat cov = arma::zeroes<arma::mat>(d.n_rows, d.n_cols);

        for (int i=0; i != nmodels; ++i) {

            arma::vec mparams = params.elem( model_params[i] );
            
            if (method == add_method)
                cov += cov_func<model_types[i]>(d, mparams);
            else if (method == prod_method)
                cov *= cov_func<model_types[i]>(d, mparams);
            else
                throw std::range_error("Unknown covariance model construction method.");
        }

        return cov;
    }
};