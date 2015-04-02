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



z1=gpu_low_rank_sym(Sigma, 1000,5,2)
z2=calc_low_rank_cov(m,d,c(2,3),1000,5,2,gpu=FALSE)

cov1=z1$U %*% diag(c(z1$C)) %*% t(z1$U)
cov2=z2$U %*% diag(c(z2$C)) %*% t(z2$U)

cov1[1:5,1:5]
cov2[1:5,1:5]
Sigma[1:5,1:5]

sum((cov1-Sigma)^2)
sum((cov2-Sigma)^2)