#ifndef REPORT_HPP
#define REPORT_HPP

#include <RcppArmadillo.h>

inline void report_start()
{
    Rcpp::Rcout << "-------------------------------------------------\n"
                << "                   Sampling                      \n"
                << "-------------------------------------------------\n";
}

inline void report_start_predict()
{
    Rcpp::Rcout << "-------------------------------------------------\n"
                << "                  Predicting                     \n"
                << "-------------------------------------------------\n";
}

inline void report_progress(int s, int n_samples, std::string const& type)
{
    Rcpp::Rcout << type << ": " << s << " of " <<  n_samples << " (" << floor(1000*s/n_samples)/10 << "%)\n";                            
}

inline void report_predict(int s, int n_samples)
{
    report_progress(s, n_samples, "Predicted");
}

inline void report_sample(int s, int n_samples)
{
    report_progress(s, n_samples, "Sampled");
}

inline void report_accept(std::string name, int s, int accept, int batch_accept, int n_report)
{
    Rcpp::Rcout << name << " Acceptance rate: " 
                << "Batch - "  << floor(1000.0*batch_accept/n_report)/10 << "%, "
                << "Overall - " << floor(1000.0*accept/s)/10 << "%\n";
}

inline void report_line()
{
    Rcpp::Rcout << "-------------------------------------------------\n";
}

#endif