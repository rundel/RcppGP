#ifndef COV_FUNCS_HPP
#define COV_FUNCS_HPP

enum cov_func {nugget_cov, const_cov, exp_cov, gauss_cov, powexp_cov, sphere_cov, matern_cov, rq_cov, periodic_cov}; 

cov_func cov_func_from_string(std::string const& str) 
{
	if 		(str == "nugget"     		 ) return nugget_cov;
	else if (str == "exponential"     	 ) return exp_cov;
	else if (str == "gaussian"     		 ) return gauss_cov;
	else if (str == "powered_exponential") return powexp_cov;
	else if (str == "spherical"          ) return sphere_cov;
	else if (str == "matern"             ) return matern_cov;
	else if (str == "rational_quadratic" ) return rq_cov;
	else if (str == "periodic"           ) return periodic_cov;
	else throw std::range_error("Unknown covariance function type: " + str + ".");

	return -1;
}


template<int i> arma::mat cov_func(arma::mat const& d, arma::vec const& params)
{
	return d;
}

template<> arma::mat cov_func<nugget_cov>(arma::mat const& d, arma::vec const& params)
{
	if (params.n_elem != 1)
		throw std::range_error("Nugget covariance function expects 1 parameters.")
	
	double tauSq = params(0);

	return tauSq * arma::eye<arma::mat>(d.n_rows, d.n_cols);
}


template<> arma::mat cov_func<exp_cov>(arma::mat const& d, arma::vec const& params)
{
	if (params.n_elem != 2)
		throw std::range_error("Exponential covariance function expects 2 parameters.")
	
	double sigmaSq = params(0);
	double phi = params(1);
	
	return sigmaSq * exp(-phi*d);
}

template<> arma::mat cov_func<gauss_cov>(arma::mat const& d, arma::vec const& params)
{
	if (params.n_elem != 2)
		throw std::range_error("Gaussian covariance function expects 2 parameters.")
	
	double sigmaSq = params(0);
	double phi = params(1);
	
    return sigmaSq * exp(-pow(phi*d,2.0));
}

template<> arma::mat cov_func<powexp_cov>(arma::mat const& d, arma::vec const& params)
{
	if (params.n_elem != 3)
		throw std::range_error("Powered exponential covariance function expects 3 parameters.")
	
	double sigmaSq = params(0);
	double phi = params(1);
	double nu = params(2);
	
	return sigmaSq * exp(-pow(phi*d),nu);
}

template<> arma::mat cov_func<sphere_cov>(arma::mat const& d, arma::vec const& params)
{
	if (params.n_elem != 2)
		throw std::range_error("Spherical covariance function expects 2 parameters.")
	
	double sigmaSq = params(0);
	double phi = params(1);
	
	arma::mat r(d.n_rows, d.n_cols);
    if (d.n_rows == d.n_cols) {
        for(int i=0; i!=r.n_rows; ++i) {
            for(int j=0; j!=i+1; ++j) {
                r(i,j) = (d(i,j) <= 1.0/phi) ? sigmaSq * (1.0 - 1.5*phi*d(i,j) + 0.5*pow(phi*d(i,j),3.0)) : 0; 
                r(j,i) = r(i,j);
            }
        }
    } else {
        for(int i=0; i!=r.n_rows; ++i) {
            for(int j=0; j!=r.n_cols; ++j) {
                r(i,j) = (d(i,j) <= 1.0/phi) ? sigmaSq * (1.0 - 1.5*phi*d(i,j) + 0.5*pow(phi*d(i,j),3.0)) : 0; 
            }
        }    
    }

    return r;  
}

template<> arma::mat cov_func<matern_cov>(arma::mat const& d, arma::vec const& params)
{
	if (params.n_elem != 3)
		throw std::range_error("Matern covariance function expects 3 parameters.")
	
	double sigmaSq = params(0);
	double phi = params(1);
	double nu = params(2);

    arma::mat r(d.n_rows, d.n_cols);
    if (d.n_rows == d.n_cols) {
        for(int i=0; i!=r.n_rows; ++i) {
            for(int j=0; j!=i+1; ++j) {
                r(i,j) = sigmaSq * pow( phi*d(i,j), nu ) * Rf_bessel_k( phi*d(i,j), nu, 1.0) / (pow(2, nu-1) * Rf_gammafn(nu));
                r(j,i) = r(i,j);
            }
        }
    } else {
        for(int i=0; i!=r.n_rows; ++i) {
            for(int j=0; j!=r.n_cols; ++j) {
                r(i,j) = sigmaSq * pow( phi*d(i,j), nu ) * Rf_bessel_k( phi*d(i,j), nu, 1.0) / (pow(2, nu-1) * Rf_gammafn(nu));
            }
        }    
    }  
	
	return r;
}

template<> arma::mat cov_func<rq_cov>(arma::mat const& d, arma::vec const& params)
{
	if (params.n_elem != 3)
		throw std::range_error("Rational quadratic covariance function expects 3 parameters.")
	
	double sigmaSq = params(0);
	double phi = params(1);
	double alpha = params(2);
 
	return sigmaSq * pow(1+pow(phi*d,2)/alpha, -alpha);
}

template<> arma::mat cov_func<periodic_cov>(arma::mat const& d, arma::vec const& params)
{
	if (params.n_elem != 3)
		throw std::range_error("Rational quadratic covariance function expects 3 parameters.")
	
	double sigmaSq = params(0);
	double phi = params(1);
	double gamma = params(2);
 
	return sigmaSq * exp(-2*pow(phi*sin(pi*d/gamma),2));
}