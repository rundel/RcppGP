#ifndef LM_PRIOR_HPP
#define LM_PRIOR_HPP

#include <RcppArmadillo.h>

struct prior_details
{
    std::string type;
    arma::vec mu;
    arma::vec sigma;
};

#endif