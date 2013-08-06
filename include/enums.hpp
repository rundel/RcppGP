#ifndef ENUMS_HPP
#define ENUMS_HPP

enum cov_func 
{
    noop,           // Included for profiling purposes
    nugget_cov, 
    const_cov, 
    exp_cov, 
    gauss_cov, 
    powexp_cov, 
    sphere_cov, 
    matern_cov, 
    rq_cov, 
    periodic_cov, 
    periodic_exp_cov
}; 

enum param_dists 
{
    fixed_dist,
    uniform_dist, 
    invgamma_dist, 
    normal_dist
}; 

#endif