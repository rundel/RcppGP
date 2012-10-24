#ifndef SPPPLM_HPP
#define SPPPLM_HPP

#include <RcppArmadillo.h>

RcppExport SEXP spmPPLM(SEXP Y_r, SEXP X_r, 
             			SEXP coordsD_r, SEXP knotsD_r, SEXP coordsKnotsD_r,
             			SEXP nugget_r, 
             			SEXP sigmaSqIG_r, SEXP tauSqIG_r, SEXP nuUnif_r, SEXP phiUnif_r,
             			SEXP covModel_r, SEXP nSamples_r, SEXP verbose_r, SEXP nReport_r);
			
#endif
