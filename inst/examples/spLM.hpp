#ifndef SPLM_HPP
#define SPLM_HPP

#include <RcppArmadillo.h>

RcppExport
SEXP spLM(SEXP Y_r, SEXP X_r, 
          SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
          SEXP cov_model_r,
          SEXP theta_r, SEXP beta_r,
          SEXP is_pp_r, SEXP is_mod_pp_r,
          SEXP n_samples_r, SEXP verbose_r, SEXP n_report_r);
		
RcppExport
SEXP spLM_gpu(SEXP Y_r, SEXP X_r, 
              SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
              SEXP cov_model_r,
              SEXP theta_r, SEXP beta_r,
              SEXP is_pp_r, SEXP is_mod_pp_r,
              SEXP n_samples_r, SEXP verbose_r, SEXP n_report_r);	
#endif
