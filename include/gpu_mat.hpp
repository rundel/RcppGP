#ifndef GPU_MAT_HPP
#define GPU_MAT_HPP

#include <RcppArmadillo.h>
#include <boost/utility.hpp>

class gpu_mat : boost::noncopyable 
{
private:
    double *mat;
    bool allocated;
    int n_rows;
    int n_cols;

    void alloc_mat();

public:
    gpu_mat();
    gpu_mat(arma::mat const& m);
    gpu_mat(int r, int c);
    gpu_mat(int r, int c, double init);
    gpu_mat(double *m, int r, int c);

    ~gpu_mat();

    int get_n_rows() const;
    int get_n_cols() const;

    arma::mat get_mat();
    double const* get_const_ptr() const;
    double* get_ptr();
    double* get_gpu_mat();
    double* get_copy();

    void assign(gpu_mat& g);
    void swap_mat(gpu_mat& g);

    bool is_allocated() const;

    void QR_Q();
    void chol(char uplo);
    void inv_chol(char uplo);
    void inv_sympd();
    void rand_proj(int rank, int over_samp, int qr_iter);
    void rand_proj(gpu_mat const& A, int rank, int over_samp, int qr_iter);
    void rand_prod(int l);
    void fill_rnorm(double mu, double sigma);
    void mat_mult(gpu_mat const& Y, char op_X, char op_Y, bool swap);
    void eig_sym(arma::vec& vals);
    void low_rank_sym(arma::vec& C, int rank, int over_samp, int qr_iter);
};

#endif