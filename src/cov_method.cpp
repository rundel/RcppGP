#include <RcppArmadillo.h>
#include <boost/assign/list_of.hpp>
#include "cov_method.hpp"

template <> std::string cov_method_map::name = "method";
template <> std::map<std::string, int> 
cov_method_map::map = boost::assign::map_list_of("sum",            add_method)
                                                ("addition",       add_method)
                                                ("product",        prod_method)
                                                ("multiplication", prod_method);

SEXP valid_cov_methods()
{
    return Rcpp::wrap(cov_method_map::valid_keys());
}
