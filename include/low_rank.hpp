#ifndef LOW_RANK_HPP
#define LOW_RANK_HPP

#include <RcppArmadillo.h>

arma::mat rand_proj(arma::mat const& A, int rank, int over_samp, int qr_iter);
void low_rank_sympd(arma::mat const& A, arma::mat& U, arma::vec& C,
                    int rank, int over_samp, int qr_iter);
void low_rank(arma::mat const& A, arma::mat& U, arma::vec& C, arma::mat& V,
              int rank, int over_samp, int qr_iter);
#endif