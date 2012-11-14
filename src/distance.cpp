#include "distance.hpp"
#include "assert.hpp"

SEXP euclid(SEXP X_r, SEXP Y_r)
{
BEGIN_RCPP
    arma::mat X = Rcpp::as<arma::mat>(X_r);
    int n = X.n_rows;

    arma::mat Y = Rcpp::as<arma::mat>(Y_r);
    int m = Y.n_rows;

    RT_ASSERT(X.n_cols==Y.n_cols, "Dimension mismatch between coordinates.");

    arma::mat D(n,m);
    for(int i=0; i!=n; ++i)
    {
        for(int j=0; j!=m; ++j)
        {
           D.at(i,j) = sqrt(arma::accu(arma::square(X.row(i)-Y.row(j))));
        }
    }

    return Rcpp::wrap(D);
END_RCPP
}

SEXP euclid_sym(SEXP X_r)
{
BEGIN_RCPP
    arma::mat X = Rcpp::as<arma::mat>(X_r);
    int n = X.n_rows;

    arma::mat D(n,n);
    D.diag() = arma::zeros<arma::vec>(n);

    if (n != 1)
    {
        for(int i=1; i<n; ++i)
        {
            for(int j=0; j<i; ++j)
            {
               D.at(i,j) = sqrt(arma::accu(arma::square(X.row(i)-X.row(j))));
               D.at(j,i) = D.at(i,j);
            }
        }
    }

    return Rcpp::wrap(D);
END_RCPP
}