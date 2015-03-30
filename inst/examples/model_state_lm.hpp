#ifndef MODEL_STATE_LM_HPP
#define MODEL_STATE_LM_HPP

#include "transforms.hpp"
#include "distributions.hpp"
#include "cov_model.hpp"
#include "lm_prior.hpp"

struct model_state_lm
{
    arma::vec R;          // - n x 1

    arma::mat C_inv;
    arma::mat C_U;
    arma::mat Cs_inv;     // (C^*)^-1                 inverse of Cs - m x m
    arma::mat ct_Csi;     // c' C^*^-1                                 - m x m
    arma::mat Einv;       // - n x n
    arma::mat Cinv;       // - n x n
    
    arma::vec beta;
    arma::vec theta;
    
    arma::vec ws;
    arma::vec w;

    bool is_pp;
    bool is_mod_pp;

    prior_details const* beta_prior;

    cov_model const* m;
    
    double logDet;
    double C_ldet;
    
    double loglik;
    double loglik_theta;
    double loglik_beta;
    double loglik_w;

    model_state_lm(cov_model const* m_, prior_details const* beta_prior_,
                   bool is_pp_, bool is_mod_pp_) 
      : m(m_),
        beta_prior(beta_prior_),
        is_pp(is_pp_),
        is_mod_pp(is_mod_pp_)
    { }

    void update_covs(arma::mat const& coordsD)
    {
        arma::mat C = m->calc_cov(coordsD, theta);

        C_U = arma::chol(C);
        arma::mat C_U_inv = arma::inv( arma::trimatu(C_U) );

        logDet = 2*arma::sum(arma::log(arma::diagvec(C_U)));

        Cinv = C_U_inv * C_U_inv.t();
    }


    void update_covs(arma::mat const& knotsD, arma::mat const& coordsKnotsD)
    {
        arma::mat Cs = m->calc_cov(knotsD, theta);
        arma::mat c  = m->calc_cov(coordsKnotsD, theta);

        arma::mat Cs_U = arma::chol(Cs);
        arma::mat Cs_U_inv = arma::inv( arma::trimatu(Cs_U) );
        Cs_inv = Cs_U_inv * Cs_U_inv.t();
        
        ct_Csi = c.t() * Cs_inv;
        arma::mat ct_Csi_c = ct_Csi * c;

        if (is_mod_pp) 
        {
            arma::vec C_diag = m->calc_cov(arma::zeros<arma::vec>(c.n_cols), theta);

            // Sherman-Woodbury-Morrison
            arma::vec E = C_diag - arma::diagmat(ct_Csi_c);
            Einv = arma::diagmat(1.0/E);
            arma::mat Einv_ct = Einv * c.t();
            arma::mat c_Einv_ct = c * Einv_ct;
            
            arma::mat J = Cs + c_Einv_ct;
            arma::mat J_U = arma::chol(J);
            arma::mat J_U_inv = arma::inv( arma::trimatu(J_U) );
            arma::mat J_inv = J_U_inv * J_U_inv.t();
            
            double logDetCs = 2*arma::accu(arma::log(arma::diagvec(Cs_U)));
            double logDetE = arma::accu(arma::log(arma::diagvec(E)));
            double logDetJ = 2*arma::accu(arma::log(arma::diagvec(J_U)));
            
            logDet = logDetE + logDetJ - logDetCs;
            
            Cinv = Einv - Einv_ct * J_inv * Einv_ct.t();
        }
        else
        {
            arma::mat C = ct_Csi_c;

            arma::mat C_U = arma::chol(C);
            arma::mat C_U_inv = arma::inv( arma::trimatu(C_U) );
            Cinv = C_U_inv * C_U_inv.t();
        
            logDet = 2*arma::sum(arma::log(arma::diagvec(C_U)));
        }
    }

    void update_beta(arma::mat const& X, arma::vec const& Y)
    {
        arma::mat Xt_Cinv = X.t() * Cinv;  // (p x n) (n x n) = p x n
        
        if (beta_prior->type == "flat") 
        {
            arma::mat S  = arma::inv(arma::sympd( Xt_Cinv * X ));
            
            arma::vec mu = S * Xt_Cinv * Y;
            
            beta = mu + arma::chol(S).t() * arma::randn<arma::vec>(S.n_rows);
            R = Y-X*beta;
        }
        else if (beta_prior->type  == "normal") 
        {
            arma::mat S  = arma::inv(arma::sympd( Xt_Cinv * X + arma::diagmat(1.0/beta_prior->sigma) ));

            arma::vec mu = S * Xt_Cinv * Y + arma::diagmat(1.0/beta_prior->sigma) * beta_prior->mu;
            
            beta = mu + arma::chol(S).t() * arma::randn<arma::vec>(S.n_rows); // p x 1
            R = Y-X*beta;
        }
        else
        {
            throw std::runtime_error("Unknown prior on beta.");
        }
    }


    void update_w()
    {
        if (!is_pp)
        {
            w = R + C_U.t() * arma::randn<arma::vec>(R.n_elem);
        }
        else
        {
            // w* ~ MVN(mu_w, Sigma_w)
            //unmodified
            //  Sigma_w = [C^{*-1} + C^{*-1} C (1/tau^2 I_n) C' C^{*-1}]^{-1}
            //  mu_w = Sigma_w [C^{*-1} C (1/tau^2 I_n) (Y-XB)]

            // modified            
            //  Sigma_w = [C^{*-1} + C^{*-1} C (1/E) C' C^{*-1}]^{-1}
            //  mu_w = Sigma_w [C^{*-1} C (1/E) (Y-XB)]
            //  where E = I \otimes (Psi + A'A) - Diag[C'(s_i) C^{*-1} C(s_i)]^n_{i=1}
            //
            // then  w = C' C^{*-1} w*

            arma::mat Sigma_ws;
            arma::vec mu_ws;

            if (!is_mod_pp)
            {
                throw std::runtime_error(std::string("Not implemented"));
            }
            else
            {
                arma::mat Einv_ct_Csi = Einv * ct_Csi;
                Sigma_ws = arma::inv(arma::sympd(Cs_inv + ct_Csi.t() * Einv_ct_Csi));

                mu_ws = Sigma_ws * Einv_ct_Csi.t() * R;
            }

            ws = mu_ws + arma::chol(Sigma_ws).t() * arma::randn<arma::vec>(mu_ws.n_elem);
            w = ct_Csi * ws;
        }
    }

    void update_theta(arma::vec const& jump)
    {
        RT_ASSERT(jump.n_elem == m->param_nfree, "Jump size differs from number of free covariance parameters.");

        for (int i=0; i!=m->param_nfree; ++i) 
        {
            int j = m->param_free_index[i];
            theta[j] = param_update(m->param_trans[j], theta[j], jump(i), m->param_hyper[j]);
        }
    }


    void calc_loglik()
    {
        loglik = loglik_theta + loglik_beta + loglik_w;
    }

    void calc_all_loglik()
    {
        calc_theta_loglik();
        calc_w_loglik();
        calc_beta_loglik();
        calc_loglik();
    }

    arma::vec get_logliks()
    {
        int n = 4;
        
        arma::vec res(n);
        res[0] = loglik;
        res[1] = loglik_theta;
        res[2] = loglik_beta;
        res[3] = loglik_w;
        
        return res;
    }

    void calc_theta_loglik()
    {
        loglik_theta = 0.0;

        for (int i=0; i!=m->nparams; ++i)
        {
            loglik_theta += log_likelihood(m->param_dists[i], theta[i], m->param_hyper[i]);
            loglik_theta += jacobian_adj(m->param_trans[i], theta[i], m->param_hyper[i]);
        }
    }

    void calc_w_loglik()
    {
        loglik_w = -0.5 * logDet - 0.5 * arma::as_scalar(R.t() * Cinv * R);
    }

    void calc_beta_loglik()
    {
        if (beta_prior->type == "flat")
        {
             loglik_beta = 0.0;
        }
        else if (beta_prior->type == "normal") 
        {
            loglik_beta = arma::accu( -arma::log(beta_prior->sigma) - 0.5 * arma::square( (beta - beta_prior->mu)/beta_prior->sigma) );
        }
        else
        {
            throw std::runtime_error("Unknown prior on beta.");
        }
    }

};

#endif