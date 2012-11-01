#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include "enum_util.hpp"

enum param_trans {identity_trans, log_trans, logit_trans}; 
typedef enum_map<param_trans> param_trans_map;
RcppExport SEXP valid_param_trans();

inline double logit(double z, double a, double b)
{
    return log((z-a)/(b-z));
}

inline double logitInv(double z, double a, double b)
{
    return b-(b-a)/(1+exp(z));
}

template<int i> double param_update(double cur, double jump, arma::vec hyperparams);
double param_update(int type, double cur, double jump, arma::vec hyperparams);

// http://www.biostat.umn.edu/~sudiptob/ResearchPapers/FBM08_Supplementary_Materials.pdf
template<int i> double jacobian_adj(double cur, arma::vec hyperparams);
double jacobian_adj(int type, double cur, arma::vec hyperparams);


#endif
