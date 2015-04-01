library(devtools)
load_all('.') #,recompile=TRUE)

n = 300 #10000

x = matrix(runif(2*n), ncol=2)
d = as.matrix( dist(x) )

nugget_cov = list(type = "nugget", params = list( 
                list( name = "tauSq" )
             ))


exponential_cov = list(type = "exponential", params = list(
                        list( name = "sigmaSq" ),
                        list( name = "phi" )
                  ))


m = cov_model(nugget_cov,exponential_cov)


Sigma = calc_cov(m, d, c(1,2,3), TRUE)

q = gpu_QR_Q(Sigma)
q2 = gpu_QR_Q(Sigma[,1:100])

stopifnot(all(dim(q) == c(300,300)))
stopifnot(all(dim(q2) == c(300,100)))
stopifnot(all( diag(t(q) %*% q) == 1))
stopifnot(all( diag(t(q2) %*% q2) == 1))

rp = gpu_rand_prod(diag(1,1000), 100)
mean(rp)
sd(rp)

p = gpu_rand_proj(Sigma, 50)


A = matrix(rnorm(3000*4000),ncol=3000)
B = matrix(rnorm(3000*4000),ncol=4000)
gpu_mat_mult(A,B,'N','N',FALSE)


z=gpu_eig_sym(Sigma)
m=z$U %*% diag(c(z$C)) %*% t(z$U)
m[1:5,1:5]
Sigma[1:5,1:5]

z=gpu_low_rank_sym(Sigma, 100)