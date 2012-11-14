#ifndef SPGLMPREDICT_HPP
#define SPGLMPREDICT_HPP

#include <RcppArmadillo.h>

RcppExport SEXP spGLMPredict(SEXP obj_r, 
                             SEXP pred_X_r, SEXP pred_D_r, SEXP between_D_r, 
                             SEXP verbose_R, SEXP n_report_r);
           
#endif
