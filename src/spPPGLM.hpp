#ifndef SPPGLM_HPP
#define SPPGLM_HPP

#include <RcppArmadillo.h>

RcppExport SEXP spPPGLM(SEXP Y_r, SEXP X_r,
                        SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
                        SEXP family_r, SEXP weights_r,
                        SEXP is_mod_pp_r,
                        SEXP beta_prior_r, SEXP beta_start_r, SEXP beta_tuning_r, SEXP beta_mu_r, SEXP beta_sd_r,
                        SEXP ws_tuning_r,
                        SEXP cov_model_r,
                        SEXP n_samples_r, SEXP verbose_r, SEXP n_report_r,
                        SEXP n_adapt_r, SEXP target_acc_r, SEXP gamma_r);
            
#endif
