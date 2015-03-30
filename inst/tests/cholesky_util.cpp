#include "adapt_mcmc.hpp"


SEXP chol_update_test(SEXP L_r, SEXP v_r)
{
BEGIN_RCPP
    arma::mat L = Rcpp::as<arma::mat>(L_r);
    arma::vec v = Rcpp::as<arma::vec>(v_r);

    vihola_adapt::chol_update(L,v);

    return Rcpp::wrap(L);
END_RCPP
}

SEXP chol_downdate_test(SEXP L_r, SEXP v_r)
{
BEGIN_RCPP
    arma::mat L = Rcpp::as<arma::mat>(L_r);
    arma::vec v = Rcpp::as<arma::vec>(v_r);

    vihola_adapt::chol_downdate(L,v);

    return Rcpp::wrap(L);
END_RCPP
}