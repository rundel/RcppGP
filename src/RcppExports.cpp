// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// calc_low_rank
Rcpp::List calc_low_rank(arma::mat cov, int rank, int over_samp, int qr_iter, bool gpu);
RcppExport SEXP RcppGP_calc_low_rank(SEXP covSEXP, SEXP rankSEXP, SEXP over_sampSEXP, SEXP qr_iterSEXP, SEXP gpuSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type cov(covSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type over_samp(over_sampSEXP);
    Rcpp::traits::input_parameter< int >::type qr_iter(qr_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type gpu(gpuSEXP);
    __result = Rcpp::wrap(calc_low_rank(cov, rank, over_samp, qr_iter, gpu));
    return __result;
END_RCPP
}
// calc_cov
arma::mat calc_cov(Rcpp::List model, arma::mat d, arma::vec p, bool gpu);
RcppExport SEXP RcppGP_calc_cov(SEXP modelSEXP, SEXP dSEXP, SEXP pSEXP, SEXP gpuSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Rcpp::List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type d(dSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type p(pSEXP);
    Rcpp::traits::input_parameter< bool >::type gpu(gpuSEXP);
    __result = Rcpp::wrap(calc_cov(model, d, p, gpu));
    return __result;
END_RCPP
}
// calc_inv_cov
Rcpp::List calc_inv_cov(Rcpp::List model, arma::mat d, arma::vec p, arma::vec nug, arma::mat d_btw, arma::mat d_knots, int rank, int over_samp, int qr_iter, bool gpu, bool low_rank, bool pred_proc, bool mod);
RcppExport SEXP RcppGP_calc_inv_cov(SEXP modelSEXP, SEXP dSEXP, SEXP pSEXP, SEXP nugSEXP, SEXP d_btwSEXP, SEXP d_knotsSEXP, SEXP rankSEXP, SEXP over_sampSEXP, SEXP qr_iterSEXP, SEXP gpuSEXP, SEXP low_rankSEXP, SEXP pred_procSEXP, SEXP modSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Rcpp::List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type d(dSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type p(pSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type nug(nugSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type d_btw(d_btwSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type d_knots(d_knotsSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type over_samp(over_sampSEXP);
    Rcpp::traits::input_parameter< int >::type qr_iter(qr_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type gpu(gpuSEXP);
    Rcpp::traits::input_parameter< bool >::type low_rank(low_rankSEXP);
    Rcpp::traits::input_parameter< bool >::type pred_proc(pred_procSEXP);
    Rcpp::traits::input_parameter< bool >::type mod(modSEXP);
    __result = Rcpp::wrap(calc_inv_cov(model, d, p, nug, d_btw, d_knots, rank, over_samp, qr_iter, gpu, low_rank, pred_proc, mod));
    return __result;
END_RCPP
}
// valid_cov_funcs
std::vector<std::string> valid_cov_funcs();
RcppExport SEXP RcppGP_valid_cov_funcs() {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    __result = Rcpp::wrap(valid_cov_funcs());
    return __result;
END_RCPP
}
// valid_nparams
int valid_nparams(std::string func_r);
RcppExport SEXP RcppGP_valid_nparams(SEXP func_rSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< std::string >::type func_r(func_rSEXP);
    __result = Rcpp::wrap(valid_nparams(func_r));
    return __result;
END_RCPP
}
// euclid
arma::mat euclid(arma::mat X, arma::mat Y);
RcppExport SEXP RcppGP_euclid(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    __result = Rcpp::wrap(euclid(X, Y));
    return __result;
END_RCPP
}
// euclid_sym
arma::mat euclid_sym(arma::mat X);
RcppExport SEXP RcppGP_euclid_sym(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    __result = Rcpp::wrap(euclid_sym(X));
    return __result;
END_RCPP
}
// check_gpu_mem
void check_gpu_mem();
RcppExport SEXP RcppGP_check_gpu_mem() {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    check_gpu_mem();
    return R_NilValue;
END_RCPP
}
// init
void init(bool verbose);
RcppExport SEXP RcppGP_init(SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    init(verbose);
    return R_NilValue;
END_RCPP
}
// finalize
void finalize();
RcppExport SEXP RcppGP_finalize() {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    finalize();
    return R_NilValue;
END_RCPP
}
// reset
void reset();
RcppExport SEXP RcppGP_reset() {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    reset();
    return R_NilValue;
END_RCPP
}
// gpu_chol
arma::mat gpu_chol(arma::mat m, char uplo);
RcppExport SEXP RcppGP_gpu_chol(SEXP mSEXP, SEXP uploSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< char >::type uplo(uploSEXP);
    __result = Rcpp::wrap(gpu_chol(m, uplo));
    return __result;
END_RCPP
}
// gpu_diag_add
arma::mat gpu_diag_add(arma::mat m, arma::vec d);
RcppExport SEXP RcppGP_gpu_diag_add(SEXP mSEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d(dSEXP);
    __result = Rcpp::wrap(gpu_diag_add(m, d));
    return __result;
END_RCPP
}
// gpu_scale
arma::mat gpu_scale(arma::mat m, double s);
RcppExport SEXP RcppGP_gpu_scale(SEXP mSEXP, SEXP sSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type s(sSEXP);
    __result = Rcpp::wrap(gpu_scale(m, s));
    return __result;
END_RCPP
}
// gpu_mat_diag_mult
arma::mat gpu_mat_diag_mult(arma::mat m, arma::vec d, char side);
RcppExport SEXP RcppGP_gpu_mat_diag_mult(SEXP mSEXP, SEXP dSEXP, SEXP sideSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d(dSEXP);
    Rcpp::traits::input_parameter< char >::type side(sideSEXP);
    __result = Rcpp::wrap(gpu_mat_diag_mult(m, d, side));
    return __result;
END_RCPP
}
// gpu_mat_mult
arma::mat gpu_mat_mult(arma::mat A, arma::mat B, char opA, char opB);
RcppExport SEXP RcppGP_gpu_mat_mult(SEXP ASEXP, SEXP BSEXP, SEXP opASEXP, SEXP opBSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type B(BSEXP);
    Rcpp::traits::input_parameter< char >::type opA(opASEXP);
    Rcpp::traits::input_parameter< char >::type opB(opBSEXP);
    __result = Rcpp::wrap(gpu_mat_mult(A, B, opA, opB));
    return __result;
END_RCPP
}
// gpu_fill_rnorm
arma::mat gpu_fill_rnorm(arma::mat m, double mu, double sigma);
RcppExport SEXP RcppGP_gpu_fill_rnorm(SEXP mSEXP, SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    __result = Rcpp::wrap(gpu_fill_rnorm(m, mu, sigma));
    return __result;
END_RCPP
}
// gpu_rand_prod
arma::mat gpu_rand_prod(arma::mat m, int l);
RcppExport SEXP RcppGP_gpu_rand_prod(SEXP mSEXP, SEXP lSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type l(lSEXP);
    __result = Rcpp::wrap(gpu_rand_prod(m, l));
    return __result;
END_RCPP
}
// gpu_rand_proj
arma::mat gpu_rand_proj(arma::mat m, int rank, int over_samp, int qr_iter);
RcppExport SEXP RcppGP_gpu_rand_proj(SEXP mSEXP, SEXP rankSEXP, SEXP over_sampSEXP, SEXP qr_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type over_samp(over_sampSEXP);
    Rcpp::traits::input_parameter< int >::type qr_iter(qr_iterSEXP);
    __result = Rcpp::wrap(gpu_rand_proj(m, rank, over_samp, qr_iter));
    return __result;
END_RCPP
}
// gpu_QR_Q
arma::mat gpu_QR_Q(arma::mat m);
RcppExport SEXP RcppGP_gpu_QR_Q(SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    __result = Rcpp::wrap(gpu_QR_Q(m));
    return __result;
END_RCPP
}
// gpu_eig_sym
Rcpp::List gpu_eig_sym(arma::mat m);
RcppExport SEXP RcppGP_gpu_eig_sym(SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    __result = Rcpp::wrap(gpu_eig_sym(m));
    return __result;
END_RCPP
}
// gpu_low_rank_sym
Rcpp::List gpu_low_rank_sym(arma::mat m, int rank, int over_samp, int qr_iter);
RcppExport SEXP RcppGP_gpu_low_rank_sym(SEXP mSEXP, SEXP rankSEXP, SEXP over_sampSEXP, SEXP qr_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type over_samp(over_sampSEXP);
    Rcpp::traits::input_parameter< int >::type qr_iter(qr_iterSEXP);
    __result = Rcpp::wrap(gpu_low_rank_sym(m, rank, over_samp, qr_iter));
    return __result;
END_RCPP
}
// gpu_low_rank_sym_op
Rcpp::List gpu_low_rank_sym_op(arma::mat m, int rank, int over_samp, int qr_iter);
RcppExport SEXP RcppGP_gpu_low_rank_sym_op(SEXP mSEXP, SEXP rankSEXP, SEXP over_sampSEXP, SEXP qr_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type over_samp(over_sampSEXP);
    Rcpp::traits::input_parameter< int >::type qr_iter(qr_iterSEXP);
    __result = Rcpp::wrap(gpu_low_rank_sym_op(m, rank, over_samp, qr_iter));
    return __result;
END_RCPP
}
// gpu_inv_sympd
arma::mat gpu_inv_sympd(arma::mat m);
RcppExport SEXP RcppGP_gpu_inv_sympd(SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    __result = Rcpp::wrap(gpu_inv_sympd(m));
    return __result;
END_RCPP
}
// gpu_inv_lr
arma::mat gpu_inv_lr(arma::mat m, arma::vec tau, int rank, int over_samp, int qr_iter, bool mod);
RcppExport SEXP RcppGP_gpu_inv_lr(SEXP mSEXP, SEXP tauSEXP, SEXP rankSEXP, SEXP over_sampSEXP, SEXP qr_iterSEXP, SEXP modSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type over_samp(over_sampSEXP);
    Rcpp::traits::input_parameter< int >::type qr_iter(qr_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type mod(modSEXP);
    __result = Rcpp::wrap(gpu_inv_lr(m, tau, rank, over_samp, qr_iter, mod));
    return __result;
END_RCPP
}
