
cov_model = function(...) 
{
    valid_funcs   = valid_cov_funcs()
    valid_dists   = valid_param_dists()
    valid_trans   = valid_param_trans()

    covs = list(...)
    nmodels = length(covs)
    
    ######################################
    # Cov model funcs                    #
    ######################################

    model_funcs = tolower(sapply(covs, function(x) ifelse(is.null(x$type), "", x$type)))
    storage.mode(model_funcs) = "character"
    stopifnot(length(model_funcs) == nmodels)

    t = charmatch(model_funcs,valid_funcs, nomatch=0)
    if (any(t==0)) 
        stop("unknown cov model function(s): ",paste0('"',model_funcs[t==0],'"(', which(t==0),')' ,collapse=', '),".")

    ######################################
    # Cov model names                   #
    ######################################

    model_names = sapply(covs, function(x) ifelse(is.null(x$name), "", x$name))
    storage.mode(model_names) = "character"
    stopifnot(length(model_names) == nmodels)

    # Missing names replaced by truncated function name
    model_names[model_names == ""] = strtrim(model_funcs[model_names == ""],3)
    
    # Non-unique names have a numeric suffix added
    t = table(model_names)
    if(length(t) != length(model_names)) {
        for(n in names(t)[t!=1]) model_names[model_names == n] = paste0(n,1:t[n])
    }

    ######################################
    # Number of parameters per model     #
    ######################################

    model_nparams = sapply(covs, function(x) length(x$params))
    storage.mode(model_nparams) = "integer"
    stopifnot(length(model_nparams) == nmodels)

    s = (model_nparams != sapply(model_funcs, valid_nparams))
    if (any(s))
        stop("covariance model(s) ", paste(model_names[s],collapse=", "), " have the incorrect number of parameters.")

    ######################################
    # Model parameters                   #
    ######################################        

    nparams = sum(model_nparams)
    params = unlist(lapply(covs, function(x) x$params), recursive=FALSE)
    param_model = rep(1:nmodels, model_nparams)
    model_params = lapply(1:nmodels, function(x) which(param_model == x))

    ######################################
    # Parameter names                    #
    ######################################        

    param_names = sapply(params, function(x) ifelse(is.null(x$name), "par", x$name))
    storage.mode(param_names) = "character"
    stopifnot(length(param_names) == nparams)

    param_names = paste(rep(model_names, model_nparams),  param_names, sep=".")
    if (length(unique(param_names)) != length(param_names))
        warning("Some parameter names are not unique.")

    ######################################
    # Parameter hyperprior distributions #
    ######################################

    param_dists = tolower(sapply(params, function(x) ifelse(is.null(x$dist), "", x$dist)))
    storage.mode(param_dists) = "character"
    stopifnot(length(param_dists) == nparams)

    if (any(param_dists == ""))
        stop("All parameters must have a hyperprior distribution.")
    
    fix_dist_names = function(y) 
    {
        z = list("ig" = "inverse gamma")

        y[y %in% names(z)] = sapply( y[y %in% names(z)], function(x) z[[x]])
        return(y)   
    }
    
    param_dists = fix_dist_names(param_dists)
    d = charmatch(param_dists,valid_dists, nomatch=0)
    if (any(d==0)) 
        stop("unknown hyperprior distibution(s): ",paste0( unique(param_dists[d==0]),collapse=', '),".")
    param_dists = valid_dists[d]

    ######################################
    # Parameter transformations          #
    ######################################

    param_trans = tolower(sapply(params, function(x) ifelse(is.null(x$trans), "identity", x$trans)))
    storage.mode(param_trans) = "character"
    stopifnot(length(param_trans) == nparams)

    d = charmatch(param_trans,valid_trans, nomatch=0)
    if (any(d==0)) 
        stop("unknown transformation(s): ",paste0( unique(param_trans[d==0]),collapse=', '),".")
    param_trans = valid_trans[d]
    
    #######################################
    # Parameter starting and tuning value #
    #######################################

    param_start = sapply(params, function(x) ifelse(is.null(x$start), NA, x$start))
    storage.mode(param_start) = "double"
    stopifnot(length(param_start) == nparams)

    if (any(is.na(param_start)))
        stop("All parameters must have a starting value.")

    param_tuning = sapply(params, function(x) ifelse(is.null(x$tuning), 0, x$tuning))
    storage.mode(param_tuning) = "double"
    stopifnot(length(param_tuning) == nparams)

    if (any(param_tuning < 0))
        stop("All parameters must have a tuning value > 0.")

    if (any(param_tuning == 0 & param_dists != "fixed"))
        stop("All non-fixed parameters must have a tuning value > 0.")

    #######################################
    # Parameter hyperparameters           #
    #######################################

    param_nhyper = sapply(params, function(x) length(x$hyperparams))
    stopifnot(length(param_nhyper) == nparams)
    
    param_hyper = lapply(params, function(x) as.numeric(x$hyperparams))
    stopifnot(length(param_hyper) == nparams)
    
    #######################################
    # Fixed Parameters                    #
    #######################################

    param_trans[param_trans == "fixed"] = "identity"

    param_nfixed = sum(param_dists == "fixed")
    param_nfree  = sum(param_dists != "fixed")
    param_free_index = which(param_dists != "fixed")

    storage.mode(param_nfixed)     = "integer"
    storage.mode(param_nfree)      = "integer"
    storage.mode(param_free_index) = "integer"



    return( list(
        nmodels = nmodels,                  # 1x1 Integer - m
        nparams = nparams,                  # 1x1 Integer - p
        model_funcs = model_funcs,          # mx1 String
        model_names = model_names,          # mx1 String
        model_nparams = model_nparams,      # mx1 Integer
        model_params = model_params,        # mx1 p(i)x1 Integer
        param_names = param_names,          # px1 String
        param_dists = param_dists,          # px1 String
        param_trans = param_trans,          # px1 String
        param_start = param_start,          # px1 Double
        param_tuning = param_tuning,        # px1 Double
        param_nhyper = param_nhyper,        # px1 Integer
        param_hyper = param_hyper,          # px1 of hp x 1 Double
        param_nfixed = param_nfixed,        # fixed params
        param_nfree = param_nfree,          # free params
        param_free_index = param_free_index # index of free params
    ))
}

calc_cov = function(m,d,p)
{
   .Call("test_calc_cov",m,d,p,"RcppGP") 
}

calc_inv_cov = function(m,d,p)
{
   .Call("test_calc_inv_cov",m,d,p) 
}

calc_chol_cov = function(m,d,p)
{
   .Call("test_calc_chol_cov",m,d,p) 
}

calc_cov_gpu = function(m,d,p)
{
   .Call("test_calc_cov_gpu",m,d,p) 
}

calc_inv_cov_gpu = function(m,d,p)
{
   .Call("test_calc_inv_cov_gpu",m,d,p) 
}

calc_chol_cov_gpu = function(m,d,p)
{
   .Call("test_calc_chol_cov_gpu",m,d,p) 
}

benchmark_cov = function(m,d,p,n)
{
   .Call("benchmark_calc_cov",m,d,p,n,"RcppGP") 
}

benchmark_inv_cov = function(m,d,p,n)
{
   .Call("benchmark_calc_inv_cov",m,d,p,n) 
}

benchmark_chol_cov = function(m,d,p,n)
{
   .Call("benchmark_calc_chol_cov",m,d,p,n) 
}

benchmark_cov_gpu = function(m,d,p,n)
{
   .Call("benchmark_calc_cov_gpu",m,d,p,n) 
}

benchmark_inv_cov_gpu = function(m,d,p,n)
{
   .Call("benchmark_calc_inv_cov_gpu",m,d,p,n) 
}

benchmark_chol_cov_gpu = function(m,d,p,n)
{
   .Call("benchmark_calc_chol_cov_gpu",m,d,p,n) 
}

#cov_model(nugget, exponential, invalid, invalid2)
#cov_model(nugget, exponential)
