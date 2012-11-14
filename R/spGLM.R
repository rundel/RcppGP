spGLM2 = function(formula, family="binomial", weights, data = parent.frame(), 
                  coords, knots = c(10,10),
                  beta, w_star = list(), e = list(),
                  cov_model,
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
    if (!is.matrix(coords)) 
        stop("error: coords must n-by-2 matrix of xy-coordinate locations")
    if (ncol(coords) != 2 || nrow(coords) != n)
        stop("error: either the coords have more than two columns or then number of rows is different than data used in the model formula")

    coords_D = as.matrix(dist(coords))
    storage.mode(coords_D) = "double"

    ####################
    ##Knots
    ####################
    is_pp = FALSE

    if (!is.null(knots))
    {
        if (is.vector(knots) && length(knots) %in% c(2,3))
        {
          
            ##allow single knot dim
            if (knots[1] > 1) {
                x.knots = seq(min(coords[,1]), max(coords[,1]), length.out=knots[1])
            } else {
                x.knots = (max(coords[,1])-min(coords[,1]))/2
            }
            
            if (knots[2] > 1) {
                y.knots = seq(min(coords[,2]), max(coords[,2]), length.out=knots[2])
            } else {
                y.knots = (max(coords[,2])-min(coords[,2]))/2
            }
          
            ##if not single knot then adjust out half distance on all sides
            if (length(knots) == 2) {
                if (knots[1] > 1) {
                    x.int = (x.knots[2]-x.knots[1])/2
                    x.knots = seq(min(x.knots)-x.int, max(x.knots)+x.int, length.out=knots[1])
                }
            
                if (knots[2] > 1) {
                    y.int = (y.knots[2]-y.knots[1])/2
                    y.knots = seq(min(y.knots)-y.int, max(y.knots)+y.int, length.out=knots[2])
                }
            
                knot_coords = as.matrix(expand.grid(x.knots, y.knots))
                is_pp = TRUE
            } else {   
                if (knots[1] > 1) {
                    x.int = knots[3]
                    x.knots = seq(min(x.knots)-x.int, max(x.knots)+x.int, length.out=knots[1])
                }
            
                if (knots[2] > 1) {
                    y.int = knots[3]
                    y.knots = seq(min(y.knots)-y.int, max(y.knots)+y.int, length.out=knots[2])
                }
            
                knot_coords = as.matrix(expand.grid(x.knots, y.knots))
                is_pp = TRUE
            }
          
        } else if (is.matrix(knots) && ncol(knots) == 2) {
            knot_coords = knots
            is_pp = TRUE
        } else {
            stop("error: knots is misspecified")
        }
    }

    m = 0
    knots_D = 0
    coords_knots_D = 0

    if (is_pp) {
        knots_D = as.matrix(dist(knot_coords))
        m = nrow(knots_D)
        coords_knots_D = matrix(0, m, n) ##this is for c^t
    
        for (i in 1:n) {
            coords_knots_D[,i] = sqrt((knot_coords[,1]-coords[i,1])^2+
                                      (knot_coords[,2]-coords[i,2])^2)
        }

        storage.mode(modified_pp)    = "logical"
        storage.mode(m)              = "integer"
        storage.mode(knots_D)        = "double"
        storage.mode(coords_knots_D) = "double"
    }

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
    ## beta settings
    #################################################### 

    if (missing(beta))
        stop("error: beta settings must be specified")

    if (is.null(beta$prior))
        beta$prior = "flat"
    stopifnot(beta$prior %in% c("flat","normal"))
    
    stopifnot(length(beta$start)==p)

    beta$tuning = as.matrix(beta$tuning)
    stopifnot( all(dim(beta$tuning) == c(p,1)) | all(dim(beta$tuning) == c(p,p)) )

    if (ncol(beta$tuning) == 1) {
        beta$tuning = rbind(beta$tuning, as.matrix(cov_model$param_tuning))
    } else {
        beta$tuning = block_diag_mat(beta$tuning, diag(cov_model$param_tuning))
    }

    storage.mode(beta$start) = "double"
    storage.mode(beta$tuning) = "double"

    if (beta$prior == "normal") {
        if(is.null(beta$hyperparam)) stop("Hyperparameters must be specified if beta has a normal prior.")
        if(length(beta$hyperparam)!=2) stop("Beta must have 2 hyperparameters (mu and sigma) if it has a normal prior.")
        
        beta$hyperparam = lapply(beta$hyperparam,c)
        if(any(sapply(beta$hyperparam, length) != p)) stop("Length of hyperparameter vectors must match the number of betas.")
        if(any(beta$hyperparam[[2]] < 0)) stop("All values of the sigma hyperparameter must be > 0.")
    }

    ####################################################
    ## ws settings
    #################################################### 


    if (missing(w_star))
        stop("error: w star settings must be specified")
    
    if (is.null(w_star$start))  w_star$start  = rep(0,m)
    if (is.null(w_star$tuning)) w_star$tuning = rep(1,m)
    
    if (length(w_star$start) == 1)  w_star$start  = rep(w_star$start,m)
    if (length(w_star$tuning) == 1) w_star$tuning = rep(w_star$tuning,m)

    stopifnot(length(w_star$start)==m)
    
    w_star$tuning = as.matrix(w_star$tuning)
    stopifnot( all(dim(w_star$tuning) == c(m,1)) | all(dim(w_star$tuning) == c(m,m)) )

    storage.mode(w_star$start) = "double"
    storage.mode(w_star$tuning) = "double"

    ####################################################
    ## e settings
    #################################################### 

    if (modified_pp & missing(e))
        stop("error: e settings must be specified")

    if (is.null(e$start))  e$start  = rep(0,n)
    if (is.null(e$tuning)) e$tuning = rep(1,n)
    
    if (length(e$start) == 1)  e$start  = rep(e$start,n)
    if (length(e$tuning) == 1) e$tuning = rep(e$tuning,n)
            
    stopifnot(length(e$start)==n)
    
    e$tuning = as.matrix(e$tuning)
    stopifnot( all(dim(e$tuning) == c(n,1)) | all(dim(e$tuning) == c(n,n)) )

    storage.mode(e$start) = "double"
    storage.mode(e$tuning) = "double"


    ####################################################
    ## Adopt default adaptation params for beta, ws, and e 
    ####################################################

    adapt_defaults = function(x)
    {
        if(is.null(x$n_adapt))       x$n_adapt = n_adapt
        if(is.null(x$target_accept)) x$target_accept = target_accept
        if(is.null(x$gamma))         x$gamma = gamma

        storage.mode(x$n_adapt) = "integer"
        storage.mode(x$target_accept) = "double"
        storage.mode(x$gamma) = "double"

        return(x)
    } 

    beta = adapt_defaults(beta)
    w_star = adapt_defaults(w_star)
    e = adapt_defaults(e)

    ####################################################
    ##Pack it up and off it goes
    ####################################################

    out = .Call("spPPGLM", Y, X,
                coords_D, knots_D, coords_knots_D,
                family, weights,
                beta, w_star, e,
                cov_model, modified_pp, 
                n_samples, verbose, n_report,
                n_adapt, target_accept, gamma,
                PACKAGE="tsBayes")  
    
    
    
    out$coords = coords
    out$is_pp = is_pp
    out$modified_pp = modified_pp
    out$verbose = verbose

    out$weights = weights
    out$family = family
    out$Y = Y
    out$X = X
    
    out$knot_coords = knot_coords
    out$knots_D = knots_D
    out$coords_D = coords_D
    out$coords_knots_D = coords_knots_D
    
    out$cov_model = cov_model
    
    out$sub_samples = sub_samples
    

    #subsample 

    start = as.integer(sub_samples[1])
    end   = as.integer(sub_samples[2])
    thin  = as.integer(sub_samples[3])

    thin = seq(start,end,by=thin)

    out$w = out$w[,thin, drop=FALSE]
    out$w_star = out$w_star[,thin, drop=FALSE]

    out$beta = out$beta[,thin, drop=FALSE]
    out$theta = out$theta[,thin, drop=FALSE]

    require(coda)

    out$params = mcmc(t(rbind(out$beta, out$theta)))
    out$n_samples = nrow(out$p.samples) ##get adjusted n_samples
    
    colnames(out$params) = c(x.names, cov_model$param_names)

    class(out) = "spGLM"
    out  
}

