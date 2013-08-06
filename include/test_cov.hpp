#ifndef TEST_COV_HPP
#define TEST_COV_HPP

RcppExport SEXP test_calc_cov(SEXP model, SEXP dist, SEXP params);
RcppExport SEXP test_calc_inv_cov(SEXP model, SEXP dist, SEXP params);
RcppExport SEXP test_calc_chol_cov(SEXP model, SEXP dist, SEXP params);
RcppExport SEXP test_calc_cov_gpu(SEXP model, SEXP dist, SEXP params);
RcppExport SEXP test_calc_inv_cov_gpu(SEXP model, SEXP dist, SEXP params);
RcppExport SEXP test_calc_chol_cov_gpu(SEXP model, SEXP dist, SEXP params);

RcppExport SEXP check_gpu_mem();

#endif