#include "cholesky_util.hpp"
#include "assert.hpp"

void drotg(double &a, double &b, double &c, double &s)
{
    // Implements BLAS drotg

    double roe = (fabs(a) > fabs(b)) ? a : b;
    double scale = fabs(a) + fabs(b);

    double r = sign(roe) * scale * sqrt(pow(a/scale,2) + pow(b/scale,2));
    
    c = a/r;
    s = b/r;

    double z = 1.0;
    if (fabs(a) > fabs(b)) 
        z = s;
    if (fabs(b) >= fabs(a) && c != 0.0) 
        z = 1.0/c;
    
    if (isnan(r))
        Rcpp::Rcout << a << ", " << b << ", " << roe << ", " << scale << ", " << r << "\n";

    a = r;
    b = z;
}

void chol_update(arma::mat& L, arma::vec const& v)
{
    arma::vec work(v);
    
    int n = L.n_cols;
    RT_ASSERT(n == v.n_elem,"Dimension mismatch between L and v.");

    for (int i=0; i<n; i++) 
    {    
        RT_ASSERT( L.at(i,i) != 0.0 || work[i] != 0.0, "cholesky update failure");
        
        double c, s;
        drotg(L.at(i,i), work[i], c, s);
        
        RT_ASSERT(L.at(i,i) != 0.0, "cholesky update failure");

        if (L.at(i,i) < 0.0) {
            L.at(i,i) = -L.at(i,i);
            c = -c; 
            s = -s;
        } 

        if (i != n-1)
        {
            // Implements BLAS drot for elements below the diagonal
            arma::vec tmp = c * L(arma::span(i+1,n-1),i) + s * work(arma::span(i+1,n-1)) ;
            work(arma::span(i+1,n-1)) = c * work(arma::span(i+1,n-1)) - s * L(arma::span(i+1,n-1),i);
            L(arma::span(i+1,n-1),i) = tmp;
        }   
    }
}

void chol_downdate(arma::mat& L, arma::vec const& v)
{
    int n = L.n_cols;
    RT_ASSERT(n == v.n_elem,"Dimension mismatch between L and v.");

    arma::vec c(n);
    arma::vec s(n);

    arma::vec work = arma::solve(arma::trimatl(L), v);

    // Generate Givens rotations
    double rho_sq = 1.0 - arma::accu(arma::square(work));
    RT_ASSERT(rho_sq > 0.0, "Resulting downdate matrix is not positive definite");

    double rho = sqrt(rho_sq);
    for (int i=n-1; i>=0; i--) 
    {    
        drotg(rho,work[i],c[i],s[i]);
    
        if (rho < 0.0) 
        {
            rho = -rho; 
            c[i] = -c[i]; 
            s[i] = -s[i];
        }
    } // NOTE: rho should be 1 now
    
    work.zeros();
    
    for (int i=n-1; i>=0; i--)
    {
        RT_ASSERT(L.at(i,i) > 0.0, "L diagonal elements must be positive");
    
        arma::vec tmp = c[i] * work(arma::span(i,n-1)) + s[i] * L(arma::span(i,n-1),i);
        L(arma::span(i,n-1),i) = c[i] * L(arma::span(i,n-1),i) - s[i] * work(arma::span(i,n-1));
        work(arma::span(i,n-1)) = tmp;

        RT_ASSERT(L.at(i,i) != 0.0, "Resulting L' diagonal elements should not be 0");

        // If diagonal elem is negative, flip sign of everything below it
        if (L.at(i,i) < 0.0) 
            L(arma::span(i,n-1),i) *= -1.0;
    }
}


SEXP chol_update_test(SEXP L_r, SEXP v_r)
{
BEGIN_RCPP
    arma::mat L = Rcpp::as<arma::mat>(L_r);
    arma::vec v = Rcpp::as<arma::vec>(v_r);

    chol_update(L,v);

    return Rcpp::wrap(L);
END_RCPP
}

SEXP chol_downdate_test(SEXP L_r, SEXP v_r)
{
BEGIN_RCPP
    arma::mat L = Rcpp::as<arma::mat>(L_r);
    arma::vec v = Rcpp::as<arma::vec>(v_r);

    chol_downdate(L,v);

    return Rcpp::wrap(L);
END_RCPP
}