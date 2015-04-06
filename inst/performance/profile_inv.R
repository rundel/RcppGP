library(devtools)
library(spBayes)
load_all('.')#,recompile=TRUE)

init(TRUE)

n = 15000
locs = matrix(runif(2*n), ncol=2)

d = iDist(locs)
m = cov_model(
        list(type = "exponential", 
             params = list(
                        list( name = "sigmaSq" ),
                        list( name = "phi" )
                      )
        )
    )

nug = rep(1,n)
#param = c(1,3)
#param = c(1,6)
param = c(1,12)


cpu = calc_inv_cov(m, d, param,nug, matrix(),matrix())
gpu = calc_inv_cov(m, d, param,nug, matrix(),matrix(), gpu=TRUE)

res = data.frame(rank=c(n,n))

res[1,"method"] = "cpu"
res[1,"time"]   = cpu$time
res[1,"error"]  = sqrt(sum((cpu$C-cpu$C)^2))

res[2,"method"] = "gpu"
res[2,"time"]   = gpu$time
res[2,"error"]  = sqrt(sum((gpu$C-cpu$C)^2))

i = 3
for(s in seq(10,70,by=5))
{
    cat(s,"\n")
    n_knots = s * s
    rank = n_knots

    g = seq(0,1,len=sqrt(rank))
    g = g[-1] - (g[2]-g[1])/2
    knot_locs = expand.grid(x=g,y=g)

    d22 = iDist(knot_locs)
    d12 = iDist(locs,knot_locs)

    tmp = calc_inv_cov(m, d, param, nug, matrix(),matrix(),rank,0,0,gpu=TRUE,low_rank=TRUE)
    res[i,"rank"] = rank
    res[i,"method"] = "lr1"
    res[i,"time"] = tmp$time
    res[i,"error"] = sqrt(sum((tmp$C-cpu$C)^2))
    i=i+1

    tmp = calc_inv_cov(m, d, param, nug, matrix(),matrix(),rank,0,1,gpu=TRUE,low_rank=TRUE)
    res[i,"rank"] = rank
    res[i,"method"] = "lr2"
    res[i,"time"] = tmp$time
    res[i,"error"] = sqrt(sum((tmp$C-cpu$C)^2))
    i=i+1

    tmp = calc_inv_cov(m, d, param, nug, matrix(),matrix(),rank,0,2,gpu=TRUE,low_rank=TRUE)
    res[i,"rank"] = rank
    res[i,"method"] = "lr3"
    res[i,"time"] = tmp$time
    res[i,"error"] = sqrt(sum((tmp$C-cpu$C)^2))
    i=i+1

    tmp = calc_inv_cov(m, d, param, nug, d12, d22,gpu=TRUE,pred_proc=TRUE)
    res[i,"rank"] = rank
    res[i,"method"] = "pp"
    res[i,"time"] = tmp$time
    res[i,"error"] = sqrt(sum((tmp$C-cpu$C)^2))
    i=i+1


    tmp = calc_inv_cov(m, d, param, nug, matrix(),matrix(),rank,0,0,gpu=TRUE,low_rank=TRUE, mod=TRUE)
    res[i,"rank"] = rank
    res[i,"method"] = "lr1 mod"
    res[i,"time"] = tmp$time
    res[i,"error"] = sqrt(sum((tmp$C-cpu$C)^2))
    i=i+1

    tmp = calc_inv_cov(m, d, param, nug, matrix(),matrix(),rank,0,1,gpu=TRUE,low_rank=TRUE, mod=TRUE)
    res[i,"rank"] = rank
    res[i,"method"] = "lr2 mod"
    res[i,"time"] = tmp$time
    res[i,"error"] = sqrt(sum((tmp$C-cpu$C)^2))
    i=i+1

    tmp = calc_inv_cov(m, d, param, nug, matrix(),matrix(),rank,0,2,gpu=TRUE,low_rank=TRUE, mod=TRUE)
    res[i,"rank"] = rank
    res[i,"method"] = "lr3 mod"
    res[i,"time"] = tmp$time
    res[i,"error"] = sqrt(sum((tmp$C-cpu$C)^2))
    i=i+1

    tmp = calc_inv_cov(m, d, param, nug, d12, d22,gpu=TRUE,pred_proc=TRUE, mod=TRUE)
    res[i,"rank"] = rank
    res[i,"method"] = "pp mod"
    res[i,"time"] = tmp$time
    res[i,"error"] = sqrt(sum((tmp$C-cpu$C)^2))
    i=i+1
}

ggplot(res, aes(x=time, y=error, group=method))+geom_line(aes(colour = method, linetype=method))