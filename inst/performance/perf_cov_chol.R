n = 3000 #10000

x = runif(2*n)
d = as.matrix( dist(x) )

nugget_cov = list(type = "nugget", params = list( 
                list( name = "tauSq" )
             ))


exponential_cov = list(type = "exponential", params = list(
                        list( name = "sigmaSq" ),
                        list( name = "phi" )
                  ))


m = cov_model(nugget_cov,exponential_cov)

r1 = calc_cov(m, d, c(1,2,3), FALSE)
r2 = calc_cov(m, d, c(1,2,3), TRUE)
