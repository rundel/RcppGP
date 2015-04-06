#ifndef GPU_MAT_OP_HPP
#define GPU_MAT_OP_HPP

#include "gpu_mat.hpp"

gpu_mat trans(gpu_mat const& m);

void scale(gpu_mat& m, double const s);
gpu_mat mult_mat(gpu_mat const& X, gpu_mat const& Y, char op_X, char op_Y);
void mult_mat_diag(gpu_mat &m, gpu_mat const& d, char side);

gpu_mat diag(gpu_mat const& m);
void add_diag(gpu_mat& m, gpu_mat const& d);
void add_mat(gpu_mat& X, gpu_mat const& Y);

void fill_rnorm(gpu_mat &m, double mu, double sigma);
gpu_mat rand_prod(gpu_mat const& m, int l);
gpu_mat rand_proj(gpu_mat const& A, int rank, int over_samp, int qr_iter);

gpu_mat low_rank_sympd(gpu_mat const& A, arma::vec& C, int rank, int over_samp, int qr_iter);
gpu_mat low_rank_sympd_op(gpu_mat const& A, arma::vec& C, int rank, int over_samp, int qr_iter);

void solve(gpu_mat& A, gpu_mat& B, char trans);
void solve_sympd(gpu_mat& A, gpu_mat& B);
void eig_sym(gpu_mat& m, arma::vec& vals);
void QR_Q(gpu_mat &m);
void chol(gpu_mat& m, char uplo);
void inv_chol(gpu_mat& m, char uplo);
void inv_sympd(gpu_mat& m);

gpu_mat inv_lr(gpu_mat const& S, arma::vec& A, int rank, int over_samp, int qr_iter, bool mod = false);
gpu_mat inv_pp(gpu_mat const& S, gpu_mat const& U, gpu_mat const& C, arma::vec A, bool mod = false);

#endif