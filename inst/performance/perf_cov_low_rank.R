library(devtools)
library(spBayes)
load_all('.') #,recompile=TRUE)

init(TRUE)

n = 15000
n_knots = 900
rank = n_knots

locs = matrix(runif(2*n), ncol=2)


g = seq(0,1,len=sqrt(rank))
g = g[-1] - (g[2]-g[1])/2
knot_locs = expand.grid(x=g,y=g)

d   = iDist(locs)
d22 = iDist(knot_locs)
d12 = iDist(locs,knot_locs)

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
S12 = calc_cov(m, d12, c(2,3), TRUE)
S22 = calc_cov(m, d22, c(2,3), TRUE)

z1 = gpu_low_rank_sym(Sigma,1000,5,1)
z2 = gpu_low_rank_sym_op(Sigma,1000,5,1)

cov1=z1$U %*% diag(c(z1$C)) %*% t(z1$U)
cov2=z2$U %*% diag(c(z2$C)) %*% t(z2$U)

cov1[1:5,1:5]
cov2[1:5,1:5]
Sigma[1:5,1:5]

sum((cov1-Sigma)^2)
sum((cov2-Sigma)^2)

system.time(gpu_low_rank_sym(Sigma,1000,5,1))
system.time(gpu_low_rank_sym_op(Sigma,1000,5,1))


m1 = matrix(1:20,5,4)
m2 = matrix(1:16,4,4)
d = rep(2,4)

gpu_mat_diag_mult(m1,d,"L")
gpu_mat_diag_mult(m1,d,"R")

gpu_mat_diag_mult(m2,d,"L")
gpu_mat_diag_mult(m2,d,"R")


gpu_diag_add(m1,d)
gpu_diag_add(m1,d)

gpu_diag_add(m2,d)
gpu_diag_add(m2,d)


gpu_scale(matrix(1:4,2,2),2)




Q = Sigma[1:20,1:20]
Q_inv = gpu_inv_lr(Q, rep(1.0,20), 20, 0, 3)

sum((Q_inv-solve(diag(1,20)+Q))^2)


#system.time(gpu_inv_lr(Sigma, rep(1.0,n), rank, 5, 2) )
#system.time(gpu_inv_pp(S22, S12, rep(1.0,n)) )
#system.time(gpu_inv_sympd(Sigma + diag(rep(1.0,n))) )

r0=gpu_inv_pp(S22, S12, rep(1.0,n))
r1=gpu_inv_lr(Sigma, rep(1.0,n), rank, 5, 0)
r2=gpu_inv_lr(Sigma, rep(1.0,n), rank, 5, 1)
r3=gpu_inv_lr(Sigma, rep(1.0,n), rank, 5, 2)
r=gpu_inv_sympd(Sigma + diag(rep(1.0,n)))

sum((r-r0)^2)
sum((r-r1)^2)
sum((r-r2)^2)
sum((r-r3)^2)
