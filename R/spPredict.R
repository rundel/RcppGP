spPredict_RcppGP = function(r, pred_coords, pred_X, start=1, end, thin=1, verbose=TRUE, n_report=100, gpu=FALSE)
{    
    if (missing(r)) 
        stop("error: spPredict expects r\n")
    
    if (!class(r) %in% c("spLM_RcppGP"))
        stop("error: requires an output object of class spGGT, bayesGeostatExact, spLM, spMvLM, spGLM, or spMvGLM\n")
    
    if (missing(pred_coords))
        stop("error: pred_coords must be specified\n")

    if (!any(is.data.frame(pred_coords), is.matrix(pred_coords)))
        stop("error: pred_coords must be a data.frame or matrix\n")
    
    if (missing(pred_X)) 
        stop("error: pred_X must be specified\n")

    if (!any(is.data.frame(pred_X), is.matrix(pred_X))) 
        stop("error: pred_X must be a data.frame or matrix\n")

    if (ncol(pred_X) != ncol(r$X))
        stop(paste("error: pred_X must have ",p," columns\n"))

    n_samples = ncol(r$beta)
    
    storage.mode(n_samples) = "integer"
    storage.mode(n_report) = "integer"

    ##thin samples and spatial effects if pre-computed
    if (missing(end)) 
        end = n_samples

    start = as.integer(start)
    end   = as.integer(end)
    thin  = as.integer(thin)

    if (start >= n_samples) stop("error: invalid start")
    if (end   >  n_samples) stop("error: invalid end")
    if (thin  >= n_samples) stop("error: invalid thin")

    thin = seq(start, end, by=as.integer(thin))

    obj_class = class(r)
    if (obj_class == "spLM_RcppGP" | obj_class == "spGLM_RcppGP")
    {
        if (obj_class == "spLM_RcppGP") 
            r$family = "identity"
        
        r$beta = r$beta[,thin, drop=FALSE]
        r$theta = r$theta[,thin, drop=FALSE]
        r$w = r$w[,thin, drop=FALSE]
        
        if (r$is_pp)
        {
            r$w_star = r$w_star[,thin, drop=FALSE]
            r$e = r$e[,thin, drop=FALSE]
        }
        
        pred_D = sp_dist(pred_coords)
        between_D = NULL
        if(r$is_pp) between_D = sp_dist(pred_coords, r$knot_coords)
        else        between_D = sp_dist(pred_coords, r$coords)

        if (gpu)
            return( .Call("spPredict_gpu", r, pred_X, pred_D, between_D, verbose, n_report, PACKAGE="RcppGP") )
        else
            return( .Call("spPredict", r, pred_X, pred_D, between_D, verbose, n_report, PACKAGE="RcppGP") )
    }
    else if (obj_class == "spMvLM")
    {
        #spPredict_MvLM(r, pred_coords, pred_X, start, end, thin, verbose, formal_args, elip_args)   
    }
    else if (obj_class == "spMvGLM")
    {
        #spPredict_MvGLM(r, pred_coords, pred_X, start, end, thin, verbose, formal_args, elip_args) 
    }
    else 
    {
        stop("error: requires an output object of class spLM or spGLM\n")
    }  
}
