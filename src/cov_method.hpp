#ifndef COV_METHOD_HPP
#define COV_METHOD_HPP

#include "enum_util.hpp"

enum cov_method {add_method, prod_method};
typedef enum_map<cov_method> cov_method_map;
RcppExport SEXP valid_cov_methods();

#endif