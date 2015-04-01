library(devtools)
load_all('.') #,recompile=TRUE)

init(TRUE)

n = 15000

x = matrix(runif(2*n), ncol=2)
d = as.matrix( dist(x) )

nugget_cov = list(type = "nugget", params = list( 
                list( name = "tauSq" )
             ))


exponential_cov = list(type = "exponential", params = list(
                        list( name = "sigmaSq" ),
                        list( name = "phi" )
                  ))

#m = cov_model(nugget_cov,exponential_cov)
#Sigma = calc_cov(m, d, c(1,2,3), TRUE)

m = cov_model(exponential_cov)
Sigma = calc_cov(m, d, c(2,3), TRUE)


z=gpu_rand_proj(Sigma, 500, qr_iter = 2)

z=gpu_low_rank_sym(Sigma, 500, qr_iter = 1)


m=z$U %*% diag(c(z$C)) %*% t(z$U)

m[1:5,1:5]
Sigma[1:5,1:5]