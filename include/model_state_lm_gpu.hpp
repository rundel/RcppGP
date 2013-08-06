#ifndef MODEL_STATE_LM_GPU_HPP
#define MODEL_STATE_LM_GPU_HPP

#include "transforms.hpp"
#include "distributions.hpp"
#include "cov_model.hpp"
#include "lm_prior.hpp"

struct model_state_lm_gpu
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

    model_state_lm_gpu(cov_model const* m_, prior_details const* beta_prior_,
                       bool is_pp_, bool is_mod_pp_) 
      : m(m_),
        beta_prior(beta_prior_),
        is_pp(is_pp_),
        is_mod_pp(is_mod_pp_)
    { }

    void update_covs(gpu_mat const& coordsD)
    {
        gpu_mat C( m->calc_cov_gpu_ptr(coordsD, theta), 
                   coordsD.n_rows, coordsD.n_cols );

        chol(C,'U');
        C_U = C.get_mat();
        
        inv_chol(C,'U');
        Cinv = C.get_mat();

        logDet = 2*arma::sum(arma::log(arma::diagvec(C_U)));
    }


    void update_covs(gpu_mat const& knotsD, gpu_mat const& coordsKnotsD)
    {
        gpu_mat Cs( m->calc_cov_gpu_ptr(knotsD, theta), knotsD.n_rows, knotsD.n_cols );
        arma::mat c = m->calc_cov_gpu(coordsKnotsD, theta);

        chol(Cs,'U');

        arma::mat Cs_U = Cs.get_mat();
        
        inv_chol(Cs,'U');
        
        Cs_inv = Cs.get_mat();
        
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
            
            arma::mat J = Cs.get_mat() + c_Einv_ct;
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
            gpu_mat C( ct_Csi_c );

            chol(C,'U');

            arma::mat C_U = C.get_mat();

            inv_chol(C,'U');
            
            Cinv = C.get_mat();
        
            logDet = 2*arma::sum(arma::log(arma::diagvec(C_U)));
        }
    }

    void update_beta(arma::mat const& X, arma::vec const& Y)
    {
        arma::mat Xt_Cinv = X.t() * Cinv;  // (p x n) (n x n) = p x n
        
        arma::mat tmp = Xt_Cinv * X;
        if (beta_prior->type == "flat") 
        {
        
        }
        else if (beta_prior->type  == "normal") 
        {
            tmp += arma::diagmat(1.0/beta_prior->sigma);
        }
        else
        {
            throw std::runtime_error("Unknown prior on beta.");
        }

        gpu_mat prec(tmp);

        inv_sympd( prec );

        arma::vec mu = prec.get_mat() * Xt_Cinv * Y;
        if (beta_prior->type  == "normal") 
            mu += arma::diagmat(beta_prior->mu/beta_prior->sigma);

        chol(prec,'L');

        arma::mat S_L = prec.get_mat();

        beta = mu + S_L * arma::randn<arma::vec>(S_L.n_rows); // p x 1
        R = Y-X*beta;
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



            if (!is_mod_pp)
            {
                throw std::runtime_error(std::string("Not implemented"));
            }
            else
            {
                arma::mat Einv_ct_Csi = Einv * ct_Csi;
                

                gpu_mat Sigma_ws(Cs_inv + ct_Csi.t() * Einv_ct_Csi);
                inv_sympd(Sigma_ws);

                arma::vec mu_ws = Sigma_ws.get_mat() * Einv_ct_Csi.t() * R;

                chol(Sigma_ws,'L');
                arma::mat Sigma_ws_L = Sigma_ws.get_mat();

                ws = mu_ws + Sigma_ws_L * arma::randn<arma::vec>(mu_ws.n_elem);
                w = ct_Csi * ws;
            }
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