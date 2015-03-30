#ifndef COV_FUNCS_GPU_HPP
#define COV_FUNCS_GPU_HPP

#ifdef USE_GPU

void nugget_cov_gpu(double const* dist, double* cov,
                    const int n, const int m,
                    double nugget, int n_threads);

void constant_cov_gpu(double const* dist, double* cov,
                      const int n, const int m,
                      double sigma2, int n_threads);

void exponential_cov_gpu(double const* dist, double* cov,
                         const int n, const int m,
                         double sigma2, double phi,
                         int n_threads);

void gaussian_cov_gpu(double const* dist, double* cov,
                      const int n, const int m,
                      double sigma2, double phi,
                      int n_threads);

void powered_exponential_cov_gpu(double const* dist, double* cov,
                                 const int n, const int m,
                                 double sigma2, double phi,
                                 double kappa, int n_threads);

void spherical_cov_gpu(double const* dist, double* cov,
                       const int n, const int m,
                       double sigma2, double phi,
                       int n_threads);

void rational_quadratic_cov_gpu(double const* dist, double* cov,
                                const int n, const int m,
                                double sigma2, double phi,
                                double alpha, int n_threads);

void periodic_cov_gpu(double const* dist, double* cov,
                      const int n, const int m,
                      double sigma2, double phi,
                      double gamma, int n_threads);


void exp_periodic_cov_gpu(double const* dist, double* cov,
                          const int n, const int m,
                          double sigma2, double phi1,
                          double gamma, double phi2,
                          int n_threads);

#endif

#endif