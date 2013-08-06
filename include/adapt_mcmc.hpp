#ifndef ADAPT_MCMC_HPP
#define ADAPT_MCMC_HPP

#include <RcppArmadillo.h>
#include "assert.hpp"

class vihola_adapt
{
private:    
    int n_adapt;
    double target_accept;
    double gamma;
    int n;

    arma::mat S;
    arma::vec U;
    arma::vec jump;

public:
    vihola_adapt(int n_adapt_, double target_accept_, double gamma_, arma::mat tuning_)
      : n_adapt(n_adapt_),
        target_accept(target_accept_),
        gamma(gamma_)
    {
        init(tuning_);
    }

    vihola_adapt(Rcpp::List& settings)
    {
        n_adapt = Rcpp::as<int>(settings["n_adapt"]);
        target_accept = Rcpp::as<double>(settings["target_accept"]);
        gamma = Rcpp::as<double>(settings["gamma"]);
        
        init( Rcpp::as<arma::mat>(settings["tuning"]) );
    }

    void init(arma::mat const& tuning) 
    {
        n = tuning.n_rows;
        if (tuning.n_rows == tuning.n_cols) {
            S = arma::chol(tuning).t();             
        } else {
            if (tuning.n_cols == 1)
                S = arma::diagmat( arma::sqrt(tuning) );
            else 
                throw std::runtime_error("Tuning argument must be a square matrix or vector.");
        }
        
        update_jump();
    }

    arma::mat get_S()
    {
        return S;
    }

    arma::vec get_jump()
    {
        return jump;
    }

    void update_jump()
    {
        U = arma::randn<arma::vec>(n);
        jump = S * U;
    }

    void update(int s, double alpha)
    {
        if(s < n_adapt)
        {
            //arma::mat S_test = arma::chol(S * (arma::eye<arma::mat>(n,n) +  c * U * U.t()) * S.t()).t();
            
            double adapt_rate = std::min(1.0, n * pow(s,-gamma));            
            double c = adapt_rate*(alpha - target_accept) / arma::dot(U,U);

            arma::vec v = sqrt(fabs(c)) * jump;
            if (c < 0.0)
                chol_downdate(S, v);
            else if (c > 0.0)
                chol_update(S, v);
        }

        update_jump();
    }

    template <typename T> static int sign(T val) 
    {
        return (val >= T(0)) - (val < T(0));
    }

    static void drotg(double &a, double &b, double &c, double &s)
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
        
        if (std::isnan(r))
            Rcpp::Rcout << a << ", " << b << ", " << roe << ", " << scale << ", " << r << "\n";

        a = r;
        b = z;
    }

    static void chol_update(arma::mat& L, arma::vec const& v)
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

    static void chol_downdate(arma::mat& L, arma::vec const& v)
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
};

class vihola_ind_adapt
{
private:    
    int n_adapt;
    double target_accept;
    double gamma;
    int n;

    arma::vec S;
    arma::vec U;
    arma::vec jump;

public:
    vihola_ind_adapt(int n_adapt_, double target_accept_, double gamma_, arma::vec tuning_)
      : n_adapt(n_adapt_),
        target_accept(target_accept_),
        gamma(gamma_)
    {
        init(tuning_);
    }

    vihola_ind_adapt(Rcpp::List& settings)
    {
        n_adapt = Rcpp::as<int>(settings["n_adapt"]);
        target_accept = Rcpp::as<double>(settings["target_accept"]);
        gamma = Rcpp::as<double>(settings["gamma"]);
        
        init( Rcpp::as<arma::vec>(settings["tuning"]) );
    }

    void init(arma::vec const& tuning) 
    {
        n = tuning.n_rows;
        S = arma::sqrt(tuning);
        
        update_jump();
    }

    arma::vec get_S()
    {
        return S;
    }

    arma::vec get_jump()
    {
        return jump;
    }

    void update_jump()
    {
        U = arma::randn<arma::vec>(n);
        jump = S % U;
    }

    void update(int s, arma::vec alpha)
    {
        RT_ASSERT(alpha.n_elem == n, "Length mismatch with alpha and tuning.");

        if(s < n_adapt)
        {
            double adapt_rate = std::min(1.0, pow(s,-gamma));
            
            S = S % sqrt(1 + adapt_rate*(alpha - target_accept));
        }

        //Rcpp::Rcout << arma::trans(S.rows(0,5));

        update_jump();
    }
};


#endif