#include "gpu_mat.hpp"

// [[Rcpp::export]]
arma::mat gpu_chol(arma::mat m, char uplo = 'U')
{
    gpu_mat g(m);
    g.chol(uplo);

    return g.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_mat_mult(arma::mat A, arma::mat B, char opA, char opB, bool swap)
{
    gpu_mat gA(A);
    gpu_mat gB(B);

    gA.mat_mult(gB, opA, opB, swap);

    return gA.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_fill_rnorm(arma::mat m, double mu, double sigma)
{
    gpu_mat g(m);

    g.fill_rnorm(mu, sigma);

    return g.get_mat();
}


// [[Rcpp::export]]
arma::mat gpu_rand_prod(arma::mat m, int l)
{
    gpu_mat g(m);
    g.rand_prod(l);

    return g.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_rand_proj(arma::mat m, int rank, int over_samp = 5, int qr_iter = 2)
{
    gpu_mat g(m);

    Rcpp::Rcout << "Rand Proj Init (" << g.get_n_rows() << ", " << g.get_n_cols() << ")\n";

    g.rand_proj(rank, over_samp, qr_iter);

    return g.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_QR_Q(arma::mat m)
{
    gpu_mat g(m);
    g.QR_Q();

    return g.get_mat();
}

// [[Rcpp::export]]
Rcpp::List gpu_eig_sym(arma::mat m)
{
    arma::vec vals;
    gpu_mat g(m);

    g.eig_sym(vals);

    return Rcpp::List::create(Rcpp::Named("C") = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(vals)),
                              Rcpp::Named("U") = g.get_mat());

}

// [[Rcpp::export]]
Rcpp::List gpu_low_rank_sym(arma::mat m, int rank, int over_samp = 5, int qr_iter = 2)
{
    arma::vec vals;
    gpu_mat g(m);

    g.low_rank_sympd(vals, rank, over_samp, qr_iter);

    return Rcpp::List::create(Rcpp::Named("C") = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(vals)),
                              Rcpp::Named("U") = g.get_mat());

}



