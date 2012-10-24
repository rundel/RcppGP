#ifndef UTIL_HPP
#define UTIL_HPP

struct prior {
  double starting;
  double a;
  double b;
  double tuning;

  prior(SEXP p_r) {
    Rcpp::NumericVector p(p_r);

    starting = p[0];
    a = p[1];
    b = p[2];
    tuning = p[3];
  }
};


void dims(arma::mat const& m)
{
    Rcpp::Rcout << "(" << m.n_rows << "," << m.n_cols << ")\n";
}

inline double logit(double z, double a, double b)
{
    return log((z-a)/(b-z));
}

inline double logitInv(double z, double a, double b)
{
    return b-(b-a)/(1+exp(z));
}

inline double inv_gamma_loglikj(double cur, prior const& p) 
{
	return -1.0*(1.0+p.a)*log(cur)-p.b/cur+log(cur);
}

inline double unif_loglikj(double cur, prior const& p) 
{
	return log(cur - p.a) + log(p.b - cur);   
}

double exp_propose(double cur, prior const& p) 
{
	return exp(log(cur) + Rcpp::rnorm(1,0,p.tuning)[0]);
}

double logit_propose(double cur, prior const& p) 
{
	return logitInv(logit(cur,p.a,p.b) + Rcpp::rnorm(1,0,p.tuning)[0], p.a, p.b);
}

inline double exp_propose(double cur, prior const& p, double fix) 
{
    return exp(log(cur) + fix);
}

inline double logit_propose(double cur, prior const& p, double fix) 
{
    return logitInv(logit(cur,p.a,p.b) + fix, p.a, p.b);
}

#endif