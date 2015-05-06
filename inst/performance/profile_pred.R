library(devtools)
library(spBayes)
library(raster)
load_all('.')#,recompile=TRUE)

n_samp = 1000
n = 100

r = raster(nrows=n, ncols=n, xmn=0, xmx=1, ymn=0, ymx=1)

locs = xyFromCell(r,1:n^2)
locs_samp = matrix(runif(n_samp*2),ncol=2)


d_full = iDist(rbind(locs,locs_samp))

d = d_full[1:n^2,1:n^2]
d_btw = d_full[1:n^2,-(1:n^2)]
d_samp = d_full[-(1:n^2),-(1:n^2)]



m = cov_model(
        list(type = "exponential", 
             params = list(
                        list( name = "sigmaSq" ),
                        list( name = "phi" )
                      )
        )
    )

set.seed(310)
ZZ = matrix(rnorm(n^2+n_samp),ncol=1)
Z = matrix(rnorm(n^2),ncol=1)


par(mfrow=c(2,3), mar=c(1,1,1,1), oma=c(1,4,4,1))

params = list( c(1,3), 
               #c(1,6), 
               c(1,12) )



for(i in 1:length(params))
{
    param = params[[i]]

    cov_full = calc_cov(m, d_full, param)
    X_full = t(chol(cov_full)) %*% ZZ
    X_samp = X_full[-(1:n^2),,drop=FALSE]

    r[] = X_full[1:n^2,]

    cov11 = calc_cov(m, d, param)
    cov12 = calc_cov(m, d_btw, param)
    cov22 = calc_cov(m, d_samp, param)

    mean_pred = cov12 %*% solve(cov22, X_samp)
    cov_pred  = cov11 - cov12 %*% solve(cov22, t(cov12))



    r2=r
    r2[] = mean_pred + t(chol(cov_pred)) %*% Z


    lr_pred = calc_low_rank(cov_pred,500)

    r3=r
    r3[] = mean_pred + lr_pred$U %*% diag(c(sqrt(lr_pred$C))) %*% t(lr_pred$U) %*% Z


    lr_pred = calc_low_rank(cov_pred,1000)

    r4=r
    r4[] = mean_pred + lr_pred$U %*% diag(c(sqrt(lr_pred$C))) %*% t(lr_pred$U) %*% Z

    rc = rev(grey.colors(32, start = 0, end = 1, gamma = 1))

    #plot(r,  axes=FALSE,legend=FALSE,col=rc)
    #if(i == 1) mtext("Truth",line=0.5)
    #if(i == 1) mtext("Strong Dep.",side=2,line=0.5)
    #if(i == 2) mtext("Weak Dep.",side=2,line=0.5)

    plot(r2, axes=FALSE,legend=FALSE,col=rc)
    if(i == 1) mtext("Cholesky",line=0.5)
    if(i == 1) mtext("Strong Dep.",side=2,line=0.5)
    if(i == 2) mtext("Weak Dep.",side=2,line=0.5)

    plot(r3, axes=FALSE,legend=FALSE,col=rc)
    if(i == 1) mtext("Rand LR (k=500)",line=0.5)

    plot(r4, axes=FALSE,legend=FALSE,col=rc)
    if(i == 1) mtext("Rand LR (k=1000)",line=0.5)
}