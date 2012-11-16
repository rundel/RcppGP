#ifndef MODEL_STATE_MPP_GLM_HPP
#define MODEL_STATE_MPP_GLM_HPP

#include "transforms.hpp"
#include "distributions.hpp"
#include "cov_model.hpp"

struct model_state_mpp_glm
{
    arma::vec beta;
    arma::vec theta;
    
    arma::mat Cs_inv;
    double Cs_ldet;
    arma::mat ct_Csi;
    
    arma::vec ws;
    arma::vec w;
    arma::vec e;

    arma::vec e_sd;
    arma::vec log_e_sd;

    cov_model *m;

    model_state_mpp_glm(cov_model *m_) 
      : m(m_)
    { }

    void update_ws(arma::vec const& jump)
    {
        ws += jump;
    }

    void update_e(arma::vec const& jump)
    {
        e += jump;
    }

    void update_beta(arma::vec const& jump)
    {
        beta += jump;
    }

    void update_theta(arma::vec const& jump)
    {
        RT_ASSERT(jump.n_elem == m->nparams, "Jump size differs from number of covariance parameters.");

        for (int i=0; i!=m->nparams; ++i) {
            theta[i] = param_update(m->param_trans[i], theta[i], jump(i), m->param_hyper[i]);
        }
    }

    void update_covs(arma::mat const& knotsD, arma::mat const& coordsKnotsD)
    {
        arma::mat Cs = m->calc_cov(knotsD, theta);
        arma::mat c = m->calc_cov(coordsKnotsD, theta);
        arma::vec C_diag = m->calc_cov(arma::zeros<arma::vec>(c.n_cols), theta);

        arma::mat Cs_U = arma::chol(Cs);
        arma::mat Cs_U_inv = arma::inv( arma::trimatu(Cs_U) );
        Cs_inv = Cs_U_inv * Cs_U_inv.t();
        Cs_ldet = 2.0 * arma::accu(arma::log(arma::diagvec(Cs_U)));

        ct_Csi = c.t() * Cs_inv;
        
        arma::vec ct_Csi_c_diag = arma::sum(ct_Csi % c.t(), 1);
        
        e_sd = arma::sqrt(C_diag - ct_Csi_c_diag);
        log_e_sd = arma::log(e_sd);
    }

    void update_w()
    {
        w = ct_Csi * ws + e;
    }

    double calc_theta_loglik()
    {
        double ll = 0.0;

        for (int i=0; i!=m->nparams; ++i)
        {
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
        // no 0.5 for the det since we are calcing based on Cholesky
        return -0.5 * Cs_ldet - 0.5 * arma::as_scalar(ws.t() * Cs_inv * ws);
    }
    
    double calc_e_loglik()
    {
        return -0.5 * e_cov_ldet - 0.5 * arma::as_scalar(e.t() * e_cov_inv * e);
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