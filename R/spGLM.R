spGLM = function(formula, family="binomial", weights, data = parent.frame(), 
                  coords, knots=NULL,
                  cov_model,
                  beta, w = list(), w_star = list(), e = list(),
                  modified_pp = TRUE, n_samples, sub_samples, verbose=TRUE, n_report=100,
                  n_adapt=0, target_accept=0.234, gamma=0.5, ...) 
{
  
    ####################################################
    ##Check for unused args
    ####################################################
    
    formal_args = names(formals(sys.function(sys.parent())))
    elip_args = names(list(...))
    for (i in elip_args)
    {
        if ( !(i %in% formal_args) )
            warning("'",i, "' is not an argument")
    }

    ####################################################
    ##formula
    ####################################################
    
    if (missing(formula))
        stop("error: formula must be specified")

    if (class(formula) == "formula") {
        holder = parseFormula(formula, data)
        Y = holder[[1]]
        X = as.matrix(holder[[2]])
        x.names = holder[[3]]
    } else {
        stop("error: formula is misspecified")
    }

    p = ncol(X)
    n = nrow(X)

    storage.mode(Y) = "double"
    storage.mode(X) = "double"
    storage.mode(p) = "integer"
    storage.mode(n) = "integer"

    ####################################################
    ##family
    ####################################################
    
    if ( !(family %in% c("binomial","poisson")) )
        stop("error: family must be binomial or poisson")

    ##default for binomial
    if (family=="binomial") {
        if (missing(weights)) 
            weights = rep(1, n)
        if (length(weights) != n)  
            stop("error: weights vector is misspecified")
    } else {
        weights = 0
    }
    storage.mode(weights) = "integer"

    ####################################################
    ##sampling method
    ####################################################
    
    storage.mode(n_samples)  = "integer"
    storage.mode(n_adapt)    = "integer"
    storage.mode(target_accept) = "double"
    storage.mode(gamma)      = "double"

    ####################################################
    ##Distance matrices
    ####################################################

    ####################
    ##Coords
    ####################
    
    if (missing(coords)) 
        stop("error: coords must be specified")
    
    coords = as.matrix(coords)

    if (nrow(coords) != n)
        stop("error: either the coords have more than two columns or then number of rows is different than data used in the model formula")

    d = ncol(coords)

    if (d < 1)
        stop("error: coords must be at least 1 dimensional.")

    coords_D = sp_dist(coords)
    storage.mode(coords_D) = "double"

    ####################
    ##Knots
    ####################

    is_pp = FALSE
    if (!is.null(knots))
    {
        if (is.vector(knots) && length(knots) %in% c(d,d+1))
        {
            knot_coords = list()   
            for(i in 1:d)
            {
                if (knots[i] > 1)
                {
                    knot_coords[[i]] = seq(min(coords[,i]), max(coords[,i]), length.out=knots[i])
                    
                    if (length(knots) == d)
                        inter = mean(knot_coords[[i]][1:2])
                    else if (length(knots) == d+1)
                        inter = knots[d+1]

                    knot_coords[[i]] = seq(min(knot_coords[[i]])-inter, max(knot_coords[[i]])+inter, length.out=knots[i])
                } 
                else
                {
                    knot_coords[[i]] = (max(coords[,i])-min(coords[,i]))/2
                }
            }

            knot_coords = as.matrix(expand.grid(knot_coords))
            is_pp = TRUE
        }
        else if (is.matrix(knots) && ncol(knots) == d)
        {
            knot_coords = knots
            is_pp = TRUE
        }
        else
        {
            stop("error: knots is misspecified")
        }
    }

    m = 0
    knots_D = matrix()
    coords_knots_D = matrix()

    if (is_pp) 
    {
        knots_D = sp_dist(knot_coords)
        
        m = nrow(knots_D)
        coords_knots_D = sp_dist(knot_coords, coords)
    }

    storage.mode(modified_pp)    = "logical"
    storage.mode(m)              = "integer"
    storage.mode(knots_D)        = "double"
    storage.mode(coords_knots_D) = "double"


    ####################################################
    ##Covariance model
    ####################################################
    
    if (missing(cov_model)) 
        stop("error: cov_model must be specified")

    ####################################################
    ##Other stuff
    #################################################### 

    if (missing(sub_samples)) 
        sub_samples = c(1, n_samples, 1)
    if (length(sub_samples) != 3 || any(sub_samples > n_samples) ) 
        stop("error: sub_samples misspecified")

    storage.mode(n_report) = "integer"
    storage.mode(verbose) = "logical"

    ####################################################
    ## theta settings
    #################################################### 

    theta = list()

    theta$start  = cov_model$param_start
    theta$tuning = as.matrix(cov_model$param_tuning[cov_model$param_free_index])
    
    ####################################################
    ## beta settings
    #################################################### 

    if (missing(beta))
        stop("error: beta settings must be specified")

    if (is.null(beta$prior)) beta$prior = "flat"
    stopifnot(beta$prior %in% c("flat","normal"))
    
    beta$tuning = as.matrix(beta$tuning)
    
    stopifnot(length(beta$start)==p)
    stopifnot( all(dim(beta$tuning) == c(p,1)) | all(dim(beta$tuning) == c(p,p)) )

    if (beta$prior == "normal") {
        if(is.null(beta$hyperparam)) stop("Hyperparameters must be specified if beta has a normal prior.")
        if(length(beta$hyperparam)!=2) stop("Beta must have 2 hyperparameters (mu and sigma) if it has a normal prior.")
        
        beta$hyperparam = lapply(beta$hyperparam,c)
        if(any(sapply(beta$hyperparam, length) != p)) stop("Length of hyperparameter vectors must match the number of betas.")
        if(any(beta$hyperparam[[2]] < 0)) stop("All values of the sigma hyperparameter must be > 0.")
    }

    ####################################################
    ## w settings
    #################################################### 

    if (!is_pp & missing(w))
        stop("error: w settings must be specified for non-PP models")
    
    if (is.null(w$start))  w$start  = rep(0,n)
    if (is.null(w$tuning)) w$tuning = rep(1,n)
    
    if (length(w$start) == 1)  w$start  = rep(w$start,n)
    if (length(w$tuning) == 1) w$tuning = rep(w$tuning,n)

    w$tuning = as.matrix(w$tuning)

    stopifnot(length(w$start)==n)
    stopifnot( all(dim(w$tuning) == c(n,1)) | all(dim(w$tuning) == c(n,n)) )

    ####################################################
    ## ws settings
    #################################################### 

    if (is_pp & missing(w_star))
        stop("error: w star settings must be specified for PP models")
    
    if (is.null(w_star$start))  w_star$start  = rep(0,m)
    if (is.null(w_star$tuning)) w_star$tuning = rep(1,m)
    
    if (length(w_star$start) == 1)  w_star$start  = rep(w_star$start,m)
    if (length(w_star$tuning) == 1) w_star$tuning = rep(w_star$tuning,m)

    w_star$tuning = as.matrix(w_star$tuning)
    
    stopifnot(length(w_star$start)==m)
    stopifnot( all(dim(w_star$tuning) == c(m,1)) | all(dim(w_star$tuning) == c(m,m)) )

    ####################################################
    ## e settings
    #################################################### 

    if (is_pp & modified_pp & missing(e))
        stop("error: e settings must be specified for modified PP models")

    if (is.null(e$start))  e$start  = rep(0,n)
    if (is.null(e$tuning)) e$tuning = rep(1,n)
    
    if (length(e$start) == 1)  e$start  = rep(e$start,n)
    if (length(e$tuning) == 1) e$tuning = rep(e$tuning,n)
            
    e$tuning = as.matrix(e$tuning)
    
    stopifnot(length(e$start)==n)
    stopifnot( all(dim(e$tuning) == c(n,1)) | all(dim(e$tuning) == c(n,n)) )

    ####################################################
    ## Adopt default adaptation params for beta, w, ws, and e 
    ####################################################

    adapt_defaults = function(x)
    {
        if(is.null(x$n_adapt))       x$n_adapt = n_adapt
        if(is.null(x$target_accept)) x$target_accept = target_accept
        if(is.null(x$gamma))         x$gamma = gamma

        storage.mode(x$n_adapt) = "integer"
        storage.mode(x$target_accept) = "double"
        storage.mode(x$gamma) = "double"
        storage.mode(x$start) = "double"
        storage.mode(x$tuning) = "double"

        return(x)
    } 

    theta = adapt_defaults(theta)
    beta = adapt_defaults(beta)
    w = adapt_defaults(w)
    w_star = adapt_defaults(w_star)
    e = adapt_defaults(e)

    ####################################################
    ##Pack it up and off it goes
    ####################################################

    out = .Call("spGLM", Y, X,
                coords_D, knots_D, coords_knots_D,
                family, weights,
                cov_model, 
                theta, beta, w, w_star, e,
                is_pp, modified_pp, 
                n_samples, verbose, n_report,
                n_adapt, target_accept, gamma,
                PACKAGE="RcppGP")  
    
    
    
    out$coords = coords
    out$is_pp = is_pp
    out$modified_pp = modified_pp
    out$verbose = verbose

    out$weights = weights
    out$family = family
    out$Y = Y
    out$X = X
    
    out$coords_D = coords_D
    
    if (is_pp)
    {
        out$knot_coords = knot_coords
        out$knots_D = knots_D
        out$coords_knots_D = coords_knots_D
    }

    out$cov_model = cov_model
    
    out$sub_samples = sub_samples
    

    #subsample 

    start = as.integer(sub_samples[1])
    end   = as.integer(sub_samples[2])
    thin  = as.integer(sub_samples[3])

    thin = seq(start,end,by=thin)

    out$beta  = out$beta[,thin, drop=FALSE]
    out$theta = out$theta[,thin, drop=FALSE]
    out$w     = out$w[,thin, drop=FALSE]
    if (is_pp)
    {
        out$w_star = out$w_star[,thin, drop=FALSE]
        out$e      = out$e[,thin, drop=FALSE]
    }
    
    out$loglik = out$loglik[,thin, drop=FALSE]

    require(coda)

    out$params = mcmc(t(rbind(out$beta, out$theta)))
    colnames(out$params) = c(x.names, cov_model$param_names)

    out$loglik_mcmc = mcmc(t(out$loglik))

    ll_names = c("total","theta","beta","link")
    if (is_pp)
    {
        ll_names = c(ll_names, "ws")
        if (modified_pp)
            ll_names = c(ll_names, "e")
    }
    else
    {
        ll_names = c(ll_names, "w")
    }
    colnames(out$loglik_mcmc) = ll_names

    class(out) = "spGLM"
    return (out)
}

