#ifndef MODEL_STATE_GLM_HPP
#define MODEL_STATE_GLM_HPP

#include <boost/assert.hpp>
#include "transforms.hpp"
#include "distributions.hpp"
#include "cov_model.hpp"

struct model_state_glm {

    arma::vec beta;
    arma::vec params;
    arma::mat Cs;
    arma::mat c;
    arma::mat Cs_U;
    arma::mat Cs_inv;
    arma::vec ws;
    arma::vec w;

    cov_model *m;

    model_state_glm(cov_model *m_, arma::vec const& beta_) 
      : m(m_),
        beta(beta_)
    {
        params = m->param_start;
    }

    void update_ws(arma::vec const& jump)
    {
        ws += jump;
    }

    void update_w()
    {
        w = c * Cs_inv * ws;
    }

    void update_beta(arma::vec const& jump)
    {
        beta += jump;
    }

    void update_params(arma::vec const& jump)
    {
        for(int i=0; i!=m->nparams; ++i){
            params(i) = param_update(m->param_trans[i], params(i), jump(i), m->param_hyper[i]);
        }
    }

    void update_covs(arma::mat const& knotsD, arma::mat const& coordsKnotsD)
    {
        Cs = m->calc_cov(knotsD, params);
        c = m->calc_cov(coordsKnotsD, params);

        Cs_U = arma::chol(Cs);
        arma::mat Cs_U_inv = arma::inv( arma::trimatu(Cs_U) );
        Cs_inv = Cs_U_inv * Cs_U_inv.t();
    }

    double calc_param_loglik()
    {
        double ll = 0.0;

        for(int i=0; i!=m->nparams; ++i){
            ll += log_likelihood(m->param_dists[i], params(i), m->param_hyper[i]);
            ll += jacobian_adj(m->param_trans[i], params(i), m->param_hyper[i]);
        }

        return ll;
    }

    double calc_mvn_loglik()
    {
        return  -arma::accu(arma::log(arma::diagvec(Cs_U))) - arma::as_scalar(ws.t() * Cs_inv * ws) / 2;
    }
    

};

#endif