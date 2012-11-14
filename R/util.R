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

sp_dist = function(x, y=NULL, method="euclidean", ...)
{
    args = list(...)

    if (!is.na(pmatch(method, "euclidian"))) 
        method = "euclidean"
    METHODS = c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")
    
    method = pmatch(method, METHODS)
    if (is.na(method)) stop("invalid distance method")
    if (method == -1)  stop("ambiguous distance method")
    method = METHODS[method]

    x = as.matrix(x)
    if (!missing(y)) {
        y = as.matrix(y)
        if (ncol(x) != ncol(y)) 
            stop("Dimensionality of x and y mismatch")
    }

    if (method == "euclidean")
    {
        if (missing(y)) return( .Call("euclid_sym", x) )
        else            return( .Call("euclid", x, y) )
    }
    else
    {
        stop("Unsupported distance function.")
    }
}
