#ifndef CHOLESKY_UTIL_HPP
#define CHOLESKY_UTIL_HPP

#include <RcppArmadillo.h>

template <typename T> int sign(T val) 
{
    return (val >= T(0)) - (val < T(0));
}

void drotg(double &a, double &b, double &c, double &s);
void chol_update(arma::mat& L, arma::vec const& v);
void chol_downdate(arma::mat& L, arma::vec const& v);

RcppExport SEXP chol_update_test(SEXP L_r, SEXP v_r);
RcppExport SEXP chol_downdate_test(SEXP L_r, SEXP v_r);
#endif