#ifndef MODEL_STATE_HPP
#define MODEL_STATE_HPP

#include "util.hpp"

enum cov_model {exp_cov, powexp_cov, matern_cov, sphere_cov, gauss_cov}; 

struct model_state {

    prior sigmaSqIG;
    prior phiUnif;
    prior tauSqIG;
    prior nuUnif;

    double sigmaSq;
    double tauSq;
    double phi;
    double nu;

    arma::vec beta;       // - p x 1
    arma::vec R;          // - n x 1

    //arma::mat C;          // C   [ C(S,S) | theta     ] covariance of sample coords - n x n
    arma::mat Cs;         // C^* [ C(S^*,S^* | theta) ] covariance of the knot locations - m x m
    arma::mat c;          // c   [ C(S,S^* | theta)   ] covariance of sample coords and knots - n x m

    arma::mat Cs_inv;     // (C^*)^-1                 inverse of Cs - m x m
    arma::mat ct_Csi;     // c' C^*^-1                                 - m x m
    //arma::mat ct_Csi_c;   // c' C^*^-1 c                                - n x n
    //arma::mat E;          // D_{\tilde{eps}+eps} =         diagonal matrix - n x n
    arma::mat Einv;       // - n x n
    //arma::mat Einv_c;     // - n x m
    //arma::mat ct_Einv_c;  // - m x m
    //arma::mat J;          // - m x m
    //arma::mat Jinv;       // - m x m
    arma::mat Cinv;       // - n x n
    double logDet;

    arma::mat Einv_ct_Csi;
    arma::mat Sigma_ws;
    arma::mat Sigma_ws_U;

    model_state(prior const& sigmaSqIG_, prior const& phiUnif_, prior const& tauSqIG_,   prior const& nuUnif_)
      : sigmaSqIG(sigmaSqIG_),
        phiUnif(phiUnif_),
        tauSqIG(tauSqIG_),
        nuUnif(nuUnif_)
    {
        sigmaSq = sigmaSqIG.starting;
        tauSq = tauSqIG.starting;
        phi = phiUnif.starting;
        nu = nuUnif.starting;
    }

    arma::mat calc_cov(cov_model m, arma::mat d)
    {
        arma::mat r;
        switch(m) {
            case exp_cov:
                r = exp(-phi*d);
                break;
      
            case powexp_cov:
                r = exp(-pow(phi*d,nu));
                break;

            case sphere_cov:
                r.set_size(d.n_rows, d.n_cols);
                if (d.n_rows == d.n_cols) {
                    for(int i=0; i!=r.n_rows; ++i) {
                        for(int j=0; j!=i+1; ++j) {
                            r(i,j) = (d(i,j) <= 1.0/phi) ? 1.0 - 1.5*phi*d(i,j) + 0.5*pow(phi*d(i,j),3) : 0; 
                            r(j,i) = r(i,j);
                        }
                    }
                } else {
                    for(int i=0; i!=r.n_rows; ++i) {
                        for(int j=0; j!=r.n_cols; ++j) {
                            r(i,j) = (d(i,j) <= 1.0/phi) ? 1.0 - 1.5*phi*d(i,j) + 0.5*pow(phi*d(i,j),3) : 0; 
                        }
                    }    
                }  
                break;

            case gauss_cov: 
                r = arma::exp(-arma::pow(phi*d,2));
                break;

            case matern_cov:
                r.set_size(d.n_rows, d.n_cols);
                if (d.n_rows == d.n_cols) {
                    for(int i=0; i!=r.n_rows; ++i) {
                        for(int j=0; j!=i+1; ++j) {
                            r(i,j) = pow( phi*d(i,j), nu ) * Rf_bessel_k( phi*d(i,j), nu, 1.0) / (pow(2, nu-1) * Rf_gammafn(nu));
                            r(j,i) = r(i,j);
                        }
                    }
                } else {
                    for(int i=0; i!=r.n_rows; ++i) {
                        for(int j=0; j!=r.n_cols; ++j) {
                            r(i,j) = pow( phi*d(i,j), nu ) * Rf_bessel_k( phi*d(i,j), nu, 1.0) / (pow(2, nu-1) * Rf_gammafn(nu));
                        }
                    }    
                }  
                break;
            }

        return r;
    }

    void update_covs(cov_model m, arma::mat const& knotsD, arma::mat const& coordsKnotsD)
    {
        Cs = sigmaSq * calc_cov(m, knotsD);
        c = sigmaSq * calc_cov(m, coordsKnotsD);
        update_Cinv();
    }

    void update_Cinv(bool brute=FALSE)
    {
        //if (!brute) {

            /******************************
               Sherman-Woodbury-Morrison
            *******************************/
        
            int n = c.n_cols;
            int m = c.n_rows;

            arma::mat Cs_U = arma::chol(Cs);
            arma::mat Cs_U_inv = arma::inv( arma::trimatu(Cs_U) );
            Cs_inv = Cs_U_inv * Cs_U_inv.t();
            
            ct_Csi = c.t() * Cs_inv;
            arma::mat ct_Csi_c = ct_Csi * c;
            
            arma::mat E = (tauSq + sigmaSq)*arma::eye<arma::mat>(n,n) - arma::diagmat(ct_Csi_c);
            Einv = arma::inv(arma::diagmat(E));                                 // Einv - n x n
            arma::mat Einv_ct = Einv * c.t();                                   // Einv_ct - n x m
            arma::mat c_Einv_ct = c * Einv_ct;                                  // c_Einv_ct - m x m
            
            arma::mat J = Cs + c_Einv_ct;                                       // J - m x m
            arma::mat J_U = arma::chol(J);                                      
            arma::mat J_U_inv = arma::inv( arma::trimatu(J_U) );
            arma::mat J_inv = J_U_inv * J_U_inv.t();
            
            double logDetCs = 2*arma::accu(arma::log(arma::diagvec(Cs_U)));
            double logDetE = arma::accu(arma::log(arma::diagvec(E)));
            double logDetJ = 2*arma::accu(arma::log(arma::diagvec(J_U)));
            
            logDet = logDetE + logDetJ - logDetCs;
            
            Cinv = Einv - Einv_ct * J_inv * Einv_ct.t();                        // Cinv - n x n
        
        /*} else {
    
            arma::mat Cs_U = chol(C_str);
            arma::mat Cs_U_inv = arma::inv( arma::trimatu(C_str_U) );
            Cs_inv = Cs_U_inv * Cs_U_inv.t();
        
            arma::mat ct_Csi_c = c.t() * Cs_inv * c;
            arma::mat C = (tauSq+sigmaSq)*arma::eye(n,n)+ct_Csi_c;
            arma::mat C_U = chol(C);
            
            arma::mat C_U_inv = arma::inv( arma::trimatu(C_U) );
            Cinv = C_U_inv * C_U_inv.t();
        
            logDet = 2*arma::sum(arma::log(arma::diagvec(C_U)));
        }*/
    }

    void update_beta(arma::mat const& X, arma::vec const& Y)
    {
        arma::mat Xt_Cinv = X.t() * Cinv;  // (p x n) (n x n) = p x n
            
        arma::mat S_beta = arma::inv(arma::sympd( Xt_Cinv * X )); // p x p
        arma::mat S_betaU = arma::chol(S_beta); // p x p
    
        arma::vec mu_beta = S_beta * Xt_Cinv * Y; // (p x p) (p x n) (n x 1) = p x 1
        
        beta = mu_beta + S_betaU.t() * arma::randn<arma::vec>(S_betaU.n_rows); // p x 1
        R = Y-X*beta;
    }

    void update_theta() 
    {
        sigmaSq = exp_propose(sigmaSq, sigmaSqIG);
        tauSq   = exp_propose(tauSq,   tauSqIG);
        phi     = logit_propose(phi,   phiUnif);
        nu      = logit_propose(nu,    nuUnif);
    }

    void update_theta(arma::vec const& step)
    {
        sigmaSq = exp_propose(sigmaSq, sigmaSqIG, step(0));
        tauSq   = exp_propose(tauSq,   tauSqIG,   step(1));
        phi     = logit_propose(phi,   phiUnif,   step(2));
        nu      = logit_propose(nu,    nuUnif,    step(3));

    }

    double calc_loglik(cov_model m, bool nugget)
    {
        double ll = 0;
        ll += inv_gamma_loglikj(sigmaSq, sigmaSqIG);           
        ll += (nugget) ? inv_gamma_loglikj(tauSq, tauSqIG) : 0;
        ll += unif_loglikj(phi, phiUnif);                      
        ll += (m == matern_cov) ? unif_loglikj(nu, nuUnif) : 0;
        
        ll += -0.5 * logDet - 0.5 * arma::as_scalar(R.t() * Cinv * R); //Rcpp::Rcout << ll << "\n";
    
        return ll;
    }

    // w* ~ MVN(mu_w, Sigma_w)
    // modified            
    // Sigma_w = [C^{*-1} + C^{*-1} C (1/E) C' C^{*-1}]^{-1}
    // mu_w = Sigma_w [C^{*-1} C (1/E) (Y-XB)]
    // where E = I \otimes (Psi + A'A) - Diag[C'(s_i) C^{*-1} C(s_i)]^n_{i=1}
    // then w = C' C^{*-1} w*

    void update_Sigma_ws() 
    {
        Einv_ct_Csi = Einv * ct_Csi;   // (n x n) (n x m) = n x m
        Sigma_ws    = arma::inv(arma::sympd(Cs_inv + ct_Csi.t() * Einv_ct_Csi));
        Sigma_ws_U  = arma::chol(Sigma_ws);
    }

    arma::vec get_params()
    {
        arma::vec p(4);
        p(0) = sigmaSq;
        p(1) = tauSq;
        p(2) = phi;
        p(3) = nu;

        return p;
    }
};

#endif