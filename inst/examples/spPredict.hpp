#ifndef SPGLMPREDICT_HPP
#define SPGLMPREDICT_HPP

#include <RcppArmadillo.h>

RcppExport SEXP spPredict(SEXP obj_r, 
                          SEXP pred_X_r, SEXP pred_D_r, SEXP between_D_r, 
                          SEXP verbose_R, SEXP n_report_r);
      
RcppExport SEXP spPredict_gpu(SEXP obj_r, 
                              SEXP pred_X_r, SEXP pred_D_r, SEXP between_D_r, 
                              SEXP verbose_R, SEXP n_report_r);

#endif
