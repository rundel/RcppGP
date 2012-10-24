nugget = list(type = "nugget", params = list( 
                list( name = "tauSq",
                      dist = "IG",
                      trans = "logit",
                      start = 1,
                      tuning = 0.01,
                      hyperparams = c(0.1, 0.1)
                    )
              ))

exponential = list(type = "exponential", params = list(
                    list( name = "sigmaSq",
                          dist = "IG",
                          trans = "logit",
                          start = 1,
                          tuning = 0.01,
                          hyperparams = c(0.1, 0.1)
                        ),         
                    list( name = "phi",
                          dist = "Unif",
                          trans = "log",
                          start = 1,
                          tuning = 0.01,
                          hyperparams = c(0.1, 0.1)
                        )            
                ))

invalid = list()
invalid2 = list(type = "exp")

covs = list(nugget, exponential, nugget)
covs2 = list(nugget,exponential, nugget, invalid2)

build_cov_model = function(..., method = "additive") 
{
    valid_methods = c("additive","multiplicative")
    valid_types = c("nugget", "exponential")
    valid_dists = c("uniform","normal","inverse gamma")
    valid_trans = c("log","logit","identity")

    covs = list(...)
    nmodels = length(covs)

    ######################################
    # Cov model aggregation method       #
    ######################################

    if (length(method) != 1) 
        stop("method must be of length 1.")
    m = charmatch(method, valid_methods, nomatch=0)
    method = valid_methods[m]
    if (m==0) 
        stop("unknown method: \"",method,"\"" ) 
    
    ######################################
    # Cov model types                    #
    ######################################

    model_types = tolower(sapply(covs, function(x) ifelse(is.null(x$type), "", x$type)))
    storage.mode(model_types) = "character"
    stopifnot(length(model_types) == nmodels)

    t = charmatch(model_types,valid_types, nomatch=0)
    if (any(t==0)) 
        stop("unknown cov model type(s): ",paste0('"',model_types[t==0],'"(', which(t==0),')' ,collapse=', '),".")

    ######################################
    # Cov model names                   #
    ######################################

    model_names = sapply(covs, function(x) ifelse(is.null(x$name), "", x$name))
    storage.mode(model_names) = "character"
    stopifnot(length(model_names) == nmodels)

    model_names[model_names == ""] = strtrim(model_types[model_names == ""],3)
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

    if (any(model_nparams < 1)) 
        stop("cov model(s) ", paste(model_names[model_nparams < 1],collapse=", "), " must have at least 1 parameter.")

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
    
    fix_dist_names = function(y) {
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
    storage.mode(param_start) = "character"
    stopifnot(length(param_start) == nparams)

    param_tuning = sapply(params, function(x) ifelse(is.null(x$tuning), 0, x$tuning))
    storage.mode(param_tuning) = "character"
    stopifnot(length(param_tuning) == nparams)

    #######################################
    # Parameter hyperparameters           #
    #######################################

    param_nhyper = sapply(params, function(x) length(x$hyperparams))
    stopifnot(length(param_nhyper) == nparams)
    
    param_hyper = lapply(params, function(x) as.numeric(x$hyperparams))
    stopifnot(length(param_hyper) == nparams)
    
    return( list(
        method = method,
        model_types = model_types,
        model_names = model_names,
        model_nparams = model_nparams,
        model_params = model_params,
        param_names = param_names,
        param_dists = param_dists,
        param_trans = param_trans,
        param_start = param_start,
        param_tuning = param_tuning,
        param_nhyper = param_nhyper,
        param_hyper = param_hyper  
    ))
}

#build_cov_model(nugget, exponential, invalid, invalid2, method = "additive")
#build_cov_model(nugget, exponential, method = "additive")
