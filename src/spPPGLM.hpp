#ifndef SPPPGLM_HPP
#define SPPPGLM_HPP

#include <RcppArmadillo.h>

RcppExport SEXP spPPGLM(SEXP Y_r, SEXP X_r,
                        SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
                        SEXP family_r, SEXP weights_r,
                        SEXP beta_r, SEXP ws_r, SEXP e_r,
                        SEXP cov_model_r, SEXP is_mod_pp_r,
                        SEXP n_samples_r, SEXP verbose_r, SEXP n_report_r,
                        SEXP n_adapt_r, SEXP target_acc_r, SEXP gamma_r);
           
#endif
