valid_cov_methods  = function() .Call("valid_cov_methods",PACKAGE="tsBayes")
valid_param_dists  = function() .Call("valid_param_dists",PACKAGE="tsBayes")
valid_param_trans  = function() .Call("valid_param_trans",PACKAGE="tsBayes")
valid_cov_funcs    = function() .Call("valid_cov_funcs",PACKAGE="tsBayes")
valid_nparams      = function(f) .Call("valid_nparams",f,PACKAGE="tsBayes")

block_diag_mat = function(...)
{
    l = list(...)
    if (is.list(l[[1]])) l = l[[1]]

    dims = sapply(l, dim)
    rdim = apply(dims,1,sum)

    r = matrix(0,rdim[1],rdim[2])
    
    s = 0
    for(i in 1:length(l)) {
        cat(s,"\n")
        sub = s + 1:dims[1,i]
        r[sub,sub] = l[[i]]

        s = s+dims[1,i]
    }

    return(r)
}
