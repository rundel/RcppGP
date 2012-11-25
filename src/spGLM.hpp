#ifndef SPGLM_HPP
#define SPGLM_HPP

#include <RcppArmadillo.h>

RcppExport 
SEXP spGLM(SEXP Y_r, SEXP X_r,
           SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
           SEXP family_r, SEXP weights_r,
           SEXP cov_model_r, 
           SEXP theta_r, SEXP beta_r, SEXP w_r, SEXP ws_r, SEXP e_r,
           SEXP is_pp_r, SEXP is_mod_pp_r,
           SEXP n_samples_r, SEXP verbose_r, SEXP n_report_r,
           SEXP n_adapt_r, SEXP target_acc_r, SEXP gamma_r);
           
#endif
