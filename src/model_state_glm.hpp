#ifndef MODEL_STATE_GLM_HPP
#define MODEL_STATE_GLM_HPP

#include <limits>

#include "transforms.hpp"
#include "distributions.hpp"
#include "cov_model.hpp"

struct model_state_glm
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
    bool is_pp;
    bool is_mod_pp;

    double    loglik;
    double    loglik_theta;
    double    loglik_beta;
    double    loglik_ws;
    arma::vec loglik_link;
    arma::vec loglik_e;

    model_state_glm(cov_model *m_, bool is_pp_, bool is_mod_pp_) 
      : m(m_),
        is_pp(is_pp_),
        is_mod_pp(is_mod_pp_)
    { 
        loglik = -std::numeric_limits<double>::max();
    

    }

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
        
        if (is_mod_pp)
        {
            arma::vec ct_Csi_c_diag = arma::sum(ct_Csi % c.t(), 1);
            
            e_sd = arma::sqrt(C_diag - ct_Csi_c_diag);
            log_e_sd = arma::log(e_sd);
        }
    }

    void update_w()
    {
        w = ct_Csi * ws;
        if (is_mod_pp)
            w += e;
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

    void calc_beta_loglik(std::string const& prior, std::vector<arma::vec> const& beta_hyperp) //arma::vec const& beta_mu, arma::vec const& beta_sd)
    {
        if (prior == "normal") 
        {
            RT_ASSERT(beta_hyperp.size() == 2, "Beta normal prior expects 2 hyperparameters.");
            loglik_beta = arma::accu( -arma::log(beta_hyperp[1]) - 0.5 * arma::square((beta - beta_hyperp[0])/beta_hyperp[1]) );
        }
        else if (prior == "flat")
        {
             loglik_beta = 0.0;
        }
        else
        {
            throw std::runtime_error("Unknown prior on beta.");
        }
    }

    double calc_ws_loglik()
    {
        loglik_ws = -0.5 * Cs_ldet - 0.5 * arma::as_scalar(ws.t() * Cs_inv * ws);
    }

    void calc_e_loglik()
    {
        loglik_e = -log_e_sd - 0.5 * arma::square(e / e_sd);
    }

    void calc_link_loglik(std::string const& family, arma::vec const& Y, arma::mat const& X, arma::vec const& weights)
    {
        if (family == "binomial")
        {
            arma::vec p = 1.0/(1+arma::exp(-X*beta-w));
            loglik_link = Y % arma::log(p) + (weights-Y) % arma::log(1-p);
        }
        else if (family == "poisson")
        {
            arma::vec l = X * beta;
            loglik_link = -arma::exp(l+w) + Y % (l+w);
        }
        else if (family == "identity")
        {
            loglik_link = arma::zeros<arma::vec>(Y.n_rows);
        }
        else
        {
            throw std::runtime_error("Unknown family.");
        }
    }

    void calc_loglik()
    {
        loglik = loglik_theta + loglik_beta + loglik_ws + arma::accu(loglik_link);
        if (is_mod_pp)
            loglik += arma::accu(loglik_e);
    }
};

#endif