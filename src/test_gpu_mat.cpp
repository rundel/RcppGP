#include "gpu_mat.hpp"
#include "gpu_mat_op.hpp"

// [[Rcpp::export]]
arma::mat gpu_chol(arma::mat m, char uplo = 'U')
{
    gpu_mat g(m);
    chol(g,uplo);

    return g.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_diag_add(arma::mat m, arma::vec d)
{
    gpu_mat g(m);

    add_diag(g, gpu_mat(d));

    return g.get_mat();
}


// [[Rcpp::export]]
arma::mat gpu_scale(arma::mat m, double s)
{
    gpu_mat g(m);

    scale(g, s);

    return g.get_mat();
}


// [[Rcpp::export]]
arma::mat gpu_mat_diag_mult(arma::mat m, arma::vec d, char side)
{
    gpu_mat g(m);

    mult_mat_diag(g, gpu_mat(d), side);

    return g.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_mat_mult(arma::mat A, arma::mat B, char opA, char opB)
{
    gpu_mat c = mult_mat(gpu_mat(A), gpu_mat(B), opA, opB);

    return c.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_fill_rnorm(arma::mat m, double mu, double sigma)
{
    gpu_mat g(m);
    fill_rnorm(g, mu, sigma);

    return g.get_mat();
}


// [[Rcpp::export]]
arma::mat gpu_rand_prod(arma::mat m, int l)
{
    gpu_mat g(m);
    rand_prod(g,l);

    return g.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_rand_proj(arma::mat m, int rank, int over_samp = 5, int qr_iter = 2)
{
    gpu_mat g(m);
    rand_proj(g, rank, over_samp, qr_iter);

    return g.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_QR_Q(arma::mat m)
{
    gpu_mat g(m);
    QR_Q(g);

    return g.get_mat();
}

// [[Rcpp::export]]
Rcpp::List gpu_eig_sym(arma::mat m)
{
    arma::vec vals;
    gpu_mat g(m);

    eig_sym(g,vals);

    return Rcpp::List::create(Rcpp::Named("C") = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(vals)),
                              Rcpp::Named("U") = g.get_mat());

}

arma::mat gpu_eig_sym(arma::mat A, arma::mat B)
{
    arma::vec vals;
    gpu_mat M(A);
    gpu_mat res(B);

    solve(M,res,'N');

    return res.get_mat();
}


// [[Rcpp::export]]
Rcpp::List gpu_low_rank_sym(arma::mat m, int rank, int over_samp = 5, int qr_iter = 2)
{
    arma::vec vals;
    gpu_mat g = low_rank_sympd(gpu_mat(m), vals, rank, over_samp, qr_iter);

    return Rcpp::List::create(Rcpp::Named("C") = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(vals)),
                              Rcpp::Named("U") = g.get_mat());
}

// [[Rcpp::export]]
Rcpp::List gpu_low_rank_sym_op(arma::mat m, int rank, int over_samp = 5, int qr_iter = 2)
{
    arma::vec vals;
    gpu_mat g = low_rank_sympd_op(gpu_mat(m), vals, rank, over_samp, qr_iter);

    return Rcpp::List::create(Rcpp::Named("C") = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(vals)),
                              Rcpp::Named("U") = g.get_mat());
}



// [[Rcpp::export]]
arma::mat gpu_inv_sympd(arma::mat m)
{
    gpu_mat g(m);
    inv_sympd(g);

    return g.get_mat();
}

// [[Rcpp::export]]
arma::mat gpu_inv_lr(arma::mat m, arma::vec tau, int rank, int over_samp, int qr_iter, bool mod)
{
    gpu_mat g = inv_lr(gpu_mat(m), tau, rank, over_samp, qr_iter, mod);

    return g.get_mat();
}

//// [[Rcpp::export]]
//arma::mat gpu_inv_pp(arma::mat C_knots,  arma::mat C_btw, arma::vec A, bool mod)
//{
//    gpu_mat g = inv_pp(gpu_mat(C_btw), gpu_mat(C_knots), A, mod);
//
//    return g.get_mat();
//}
