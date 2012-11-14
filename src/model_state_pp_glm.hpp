#ifndef MODEL_STATE_PP_GLM_HPP
#define MODEL_STATE_PP_GLM_HPP

#include "transforms.hpp"
#include "distributions.hpp"
#include "cov_model.hpp"

struct model_state_pp_glm {

    arma::vec beta;
    arma::vec theta;
    arma::mat Cs;
    arma::mat c;
    arma::mat Cs_U;
    arma::mat Cs_inv;
    arma::mat ct_Csi;
    arma::vec ws;
    arma::vec w;

    cov_model *m;

    model_state_pp_glm(cov_model *m_) 
      : m(m_)
    { }

    void update_ws(arma::vec const& jump)
    {
        ws += jump;
    }

    void update_w()
    {
        w = ct_Csi * ws;
    }

    void update_beta(arma::vec const& jump)
    {
        beta += jump;
    }

    void update_theta(arma::vec const& jump)
    {
        RT_ASSERT(jump.n_elem == m->nparams, "Jump size differs from number of covariance parameters.");

        for (int i=0; i!=m->nparams; ++i) {
            theta[i] = param_update(m->param_trans[i], theta[i], jump[i], m->param_hyper[i]);
        }
    }

    void update_covs(arma::mat const& knotsD, arma::mat const& coordsKnotsD)
    {
        Cs = m->calc_cov(knotsD, theta);
        c = m->calc_cov(coordsKnotsD, theta);

        Cs_U = arma::chol(Cs);
        arma::mat Cs_U_inv = arma::inv( arma::trimatu(Cs_U) );
        Cs_inv = Cs_U_inv * Cs_U_inv.t();
        ct_Csi = c.t() * Cs_inv;
    }

    double calc_theta_loglik()
    {
        double ll = 0.0;

        for (int i=0; i!=m->nparams; ++i) {
            ll += log_likelihood(m->param_dists[i], theta[i], m->param_hyper[i]);
            ll += jacobian_adj(m->param_trans[i], theta[i], m->param_hyper[i]);
        }

        return ll;
    }

    double calc_beta_norm_loglik(arma::vec const& beta_mu, arma::vec const& beta_sd)
    {
        return arma::accu( -arma::log(beta_sd) - 0.5 * arma::square((beta-beta_mu)/beta_sd) );    
    }

    double calc_ws_loglik()
    {
        return -arma::accu(arma::log(arma::diagvec(Cs_U))) - 0.5 * arma::as_scalar(ws.t() * Cs_inv * ws);
    }
    
    double calc_binomial_loglik(arma::vec const& Y, arma::mat const& X, arma::vec const& weights)
    {
        arma::vec p = 1.0/(1+arma::exp(-X*beta-w));
        arma::vec loglik = Y % arma::log(p) + (weights-Y) % arma::log(1-p);
      
        return arma::accu(loglik);
    }

    double calc_poisson_loglik(arma::vec const& Y, arma::mat const& X)
    {
        arma::vec l = X * beta;
        arma::vec loglik = -arma::exp(l+w) + Y % (l+w);

        return arma::accu(loglik);
    }

};

#endif