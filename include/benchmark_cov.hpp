#ifndef TEST_COV_HPP
#define TEST_COV_HPP

RcppExport SEXP benchmark_calc_cov(SEXP model, SEXP dist, SEXP params, SEXP n_rep);
RcppExport SEXP benchmark_calc_inv_cov(SEXP model, SEXP dist, SEXP params, SEXP n_rep);
RcppExport SEXP benchmark_calc_chol_cov(SEXP model, SEXP dist, SEXP params, SEXP n_rep);
RcppExport SEXP benchmark_calc_cov_gpu(SEXP model, SEXP dist, SEXP params, SEXP n_rep);
RcppExport SEXP benchmark_calc_inv_cov_gpu(SEXP model, SEXP dist, SEXP params, SEXP n_rep);
RcppExport SEXP benchmark_calc_chol_cov_gpu(SEXP model, SEXP dist, SEXP params, SEXP n_rep);

#endif