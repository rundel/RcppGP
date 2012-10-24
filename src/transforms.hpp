#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

enum param_tran {identity_trans, log_trans, logit_trans}; 

param_tran trans_from_string(std::string const& str) {
	if (str == "identity") return identity_trans;
	else if (str == "log") return log_trans;
	else if (str == "logit") return logit_trans;
	else throw std::range_error("Unknown transform type: "+str+".")
}

inline double logit(double z, double a, double b)
{
    return log((z-a)/(b-z));
}

inline double logitInv(double z, double a, double b)
{
    return b-(b-a)/(1+exp(z));
}

template<int i> double transform(double cur, double jump, arma::vec hyperparams)
{
	return cur+jump;
}

template<> double transform<log_trans>(double cur, double jump, arma::vec hyperparams)
{
	return cur*exp(jump);
}

template<> double transform<logit_trans>(double cur, double jump, arma::vec hyperparams)
{
	double a = hyperparams(0);
	double b = hyperparams(0);
	return logitInv(logit(cur,a,b) + jump, a, b);
}
