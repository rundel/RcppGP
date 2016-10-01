#include <RcppArmadillo.h>
#include <boost/assert.hpp>

// [[Rcpp::export]]
arma::mat euclid(arma::mat X, arma::mat Y)
{
    int n = X.n_rows;
    int m = Y.n_rows;

    BOOST_ASSERT_MSG(X.n_cols==Y.n_cols, "Dimension mismatch between coordinates.");

    arma::mat D(n,m);
    for(int i=0; i!=n; ++i)
    {
        for(int j=0; j!=m; ++j)
        {
           D.at(i,j) = sqrt(arma::accu(arma::square(X.row(i)-Y.row(j))));
        }
    }

    return D;
}

// [[Rcpp::export]]
arma::mat euclid_sym(arma::mat X)
{
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

    return D;
}