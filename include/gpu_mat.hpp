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

    // Disable copy constructors
    gpu_mat(const gpu_mat&) = delete;
    gpu_mat& operator=(const gpu_mat&) = delete;

    gpu_mat(gpu_mat&& rhs);
    gpu_mat& operator=(gpu_mat&& rhs);

    gpu_mat();
    gpu_mat(arma::mat const& m);
    gpu_mat(int r, int c);
    gpu_mat(int r, int c, double init);
    gpu_mat(double *m, int r, int c);

    ~gpu_mat();

    void release();

    int get_n_rows() const;
    int get_n_cols() const;

    arma::mat get_mat() const;
    double const* get_const_ptr() const;
    double* get_ptr();

    gpu_mat make_copy() const;

    void assign(gpu_mat& g);
    void swap(gpu_mat& g);

    bool is_allocated() const;
};

#endif