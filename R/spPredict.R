spPredict = function(r, pred_coords, pred_X, start=1, end, thin=1, verbose=TRUE, n_report=100) {
  
    ####################################################
    ##Check for unused args
    ####################################################
    
    #formal_args = names(formals(sys.function(sys.parent())))
    #elip_args = names(list(...))
    #
    #for (i in elip_args) 
    #{
    #    if (! i %in% formal_args) 
    #        warning("'",i, "' is not an argument")
    #}
    
    if (missing(r)) 
        stop("error: spPredict expects r\n")
    
    if (!class(r) %in% c("spGGT","bayesGeostatExact","spLM","spMvLM", "spGLM", "spMvGLM"))
        stop("error: requires an output object of class spGGT, bayesGeostatExact, spLM, spMvLM, spGLM, or spMvGLM\n")
    
    if (missing(pred_coords))
        stop("error: pred_coords must be specified\n")

    if (!any(is.data.frame(pred_coords), is.matrix(pred_coords)))
        stop("error: pred_coords must be a data.frame or matrix\n")

    if (!ncol(pred_coords) == 2) 
        stop("error: pred_coords must have two columns (assumed to be X, Y)\n")
    
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

    storage.mode(n_report) = "integer"

    obj_class = class(r)

    ##
    ##prediction
    ##
    if (obj_class == "spLM" | obj_class == "spGLM")
    {
        if (obj_class == "spLM") r$family = "identity"
        spPredict_GLM(r, pred_coords, pred_X, thin, verbose, n_report)
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
        stop("error: requires an output object of class spGGT, bayesGeostatExact, or spLM\n")
    }  
}

spPredict_GLM = function(r, pred_coords, pred_X, thin, verbose, n_report) 
{
    r$beta = r$beta[,thin, drop=FALSE]
    r$theta = r$theta[,thin, drop=FALSE]
    r$w = r$w[,thin, drop=FALSE]
    r$w_star = r$w_star[,thin, drop=FALSE]

    pred_D = sp_dist(pred_coords)
    between_D = NULL
    if(r$is_pp) between_D = sp_dist(pred_coords, r$knot_coords)
    else        between_D = sp_dist(pred_coords, r$coords)

    return( .Call("spGLMPredict", r, pred_X, pred_D, between_D, verbose, n_report, PACKAGE="tsBayes") )
}


spPredict_MvLM = function(r, pred_coords, pred_X, start, end, thin, verbose, formal_args, elip_args) 
{
    is_pp = r$is_pp
    modified.pp = r$modified.pp
    
    if (is_pp)
        knot_coords = r$knot_coords
    
    Y = r$Y
    X = r$X
    n = r$n
    m = r$m
    p = r$p
    q = r$q
    obs.coords = r$coords
    knots_D = r$knots_D
    obs.D = r$coords_D
    obs.knots_D = r$coords_knots_D
    cov.model = r$cov.model
    nugget = r$nugget
    n.samples = r$n.samples
    samples = r$p.samples
    sp.effects = r$recovered.effects

    
    samples = samples[seq(start, end, by=as.integer(thin)),]
    n.samples = nrow(samples)

    w = NULL
    w.str = NULL
    
    if (sp.effects) {##Currently I'm forcing the sp.effects in spMvLM
        w = r$sp.effects[,seq(start, end, by=as.integer(thin))]
        if (is_pp)
            w.str = r$sp.effects.knots[,seq(start, end, by=as.integer(thin))]
    } else {
        w = matrix(0, n, n.samples) #no need to compute w for the pp, but might as well on the way.
        if (is_pp)
            w.str = matrix(0, m, n.samples)
    }
    
    ##get samples
    beta = t(samples[,1:p])##transpose to simply BLAS call
    A = NULL
    L = NULL
    phi = NULL
    nu = NULL

    
    A.chol = function(x, m) {
        A = matrix(0, m, m)
        A[lower.tri(A, diag=TRUE)] = x
        A[upper.tri(A, diag=FALSE)] = t(A)[upper.tri(A, diag=FALSE)]
        t(chol(A))[lower.tri(A, diag=TRUE)]
    }
    
    nltr = m*(m-1)/2+m

    A = samples[,(p+1):(p+nltr)]
    A = t(apply(A, 1, A.chol, m))
    A = t(A)
    
    if (!nugget && cov.model != "matern") {
        phi = t(samples[,(p+nltr+1):(p+nltr+m)])
    } else if (nugget && cov.model != "matern") {
        L = samples[,(p+nltr+1):(p+2*nltr)]; L = t(apply(L, 1, A.chol, m)); L = t(L)
        phi = t(samples[,(p+2*nltr+1):(p+2*nltr+m)])
    } else if (!nugget && cov.model == "matern") {
        phi = t(samples[,(p+nltr+1):(p+nltr+m)])
        nu = t(samples[,(p+nltr+m+1):(p+nltr+2*m)])
    } else {
        L = samples[,(p+nltr+1):(p+2*nltr)]; L = t(apply(L, 1, A.chol, m)); L = t(L)
        phi = t(samples[,(p+2*nltr+1):(p+2*nltr+m)])
        nu = t(samples[,(p+2*nltr+m+1):(p+2*nltr+2*m)])
    }

    pred.D = as.matrix(dist(pred_coords))
    n.pred = nrow(pred.D)
    pred.knots_D = NULL
    pred.obs.D = NULL
    
    if (is_pp) {
        pred.knots_D = matrix(0, n.pred, q)
        
        for (i in 1:n.pred) {
            pred.knots_D[i,] = sqrt((pred_coords[i,1]-knot_coords[,1])^2 + (pred_coords[i,2]-knot_coords[,2])^2)
        } 
    } else {
        pred.obs.D = matrix(0, n.pred, n)
      
        for (i in 1:n.pred) {
            pred.obs.D[i,] = sqrt((pred_coords[i,1]-obs.coords[,1])^2 + (pred_coords[i,2]-obs.coords[,2])^2)
        } 
    }

    ##fix the nugget if needed
    if (is_pp && !nugget) {
        
        if (modified.pp) {
           L = rep(rep(0,m), n.samples)
        } else {
           L = rep(diag(sqrt(0.01))[lower.tri(diag(m), diag=TRUE)], n.samples)
        }
        nugget = TRUE
    }
    
    storage.mode(X) = "double"
    storage.mode(Y) = "double"
    storage.mode(is_pp) = "integer"
    storage.mode(modified.pp) = "integer"
    storage.mode(n) = "integer"
    storage.mode(m) = "integer"
    storage.mode(p) = "integer"
    storage.mode(q) = "integer"
    storage.mode(nugget) = "integer"
    storage.mode(beta) = "double"
    storage.mode(A) = "double"
    storage.mode(L) = "double"
    storage.mode(phi) = "double"
    storage.mode(nu) = "double"
    storage.mode(n.pred) = "integer"
    storage.mode(pred_X) = "double"
    storage.mode(obs.D) = "double"
    storage.mode(pred.D) = "double"
    storage.mode(pred.obs.D) = "double"
    storage.mode(obs.knots_D) = "double"
    storage.mode(knots_D) = "double"
    storage.mode(pred.knots_D) = "double"
    storage.mode(n.samples) = "integer"
    storage.mode(w) = "double"
    storage.mode(w.str) = "double"
    storage.mode(sp.effects) = "integer"
    storage.mode(verbose) = "integer"

    out = .Call("spMvLMPredict",X, Y, is_pp, modified.pp, n, m, p, q, nugget, beta, A, L, phi, nu,
                 n.pred, pred_X, obs.D, pred.D, pred.obs.D, obs.knots_D, knots_D, pred.knots_D,
                 cov.model, n.samples, w, w.str, sp.effects, verbose)
    out
}

spPredict_MvGLM = function(r, pred_coords, pred_X, start, end, thin, verbose, formal_args, elip_args) 
{
    is_pp = r$is_pp
    modified.pp = r$modified.pp

    if (is_pp) knot_coords = r$knot_coords

    family = r$family
    Y = r$Y
    X = r$X
    n = r$n
    m = r$m
    p = r$p
    q = r$q
    obs.coords = r$coords
    knots_D = r$knots_D
    obs.D = r$coords_D
    obs.knots_D = r$coords_knots_D
    cov.model = r$cov.model
    nugget = r$nugget
    n.samples = r$n.samples
    samples = r$p.samples
    sp.effects = r$recovered.effects






    samples = samples[seq(start, end, by=as.integer(thin)),]
    n.samples = nrow(samples)

    w = NULL
    w.str = NULL

    ##Currently I'm forcing the sp.effects in spMvLM
    w = r$sp.effects[,seq(start, end, by=as.integer(thin))]
    if (is_pp)
        w.str = r$sp.effects.knots[,seq(start, end, by=as.integer(thin))]
    
    
    ##get samples
    beta = t(samples[,1:p])##transpose to simply BLAS call
    A = NULL
    L = NULL
    phi = NULL
    nu = NULL

    nltr = m*(m-1)/2+m

    A.chol = function(x, m) {
        A = matrix(0, m, m)
        A[lower.tri(A, diag=TRUE)] = x
        A[upper.tri(A, diag=FALSE)] = t(A)[upper.tri(A, diag=FALSE)]
        t(chol(A))[lower.tri(A, diag=TRUE)]
    }

    if (cov.model != "matern") {
        
        A = samples[,(p+1):(p+nltr)];
        A = t(apply(A, 1, A.chol, m)); 
        A = t(A)
        phi = t(samples[,(p+nltr+1):(p+nltr+m)])

    } else {
        A = samples[,(p+1):(p+nltr)]; 
        A = t(apply(A, 1, A.chol, m)); 
        A = t(A)
        phi = t(samples[,(p+nltr+1):(p+nltr+m)])
        nu = t(samples[,(p+nltr+m+1):(p+nltr+2*m)])
    }

    pred.D = as.matrix(dist(pred_coords))
    n.pred = nrow(pred.D)
    pred.knots_D = NULL
    pred.obs.D = NULL

    if (is_pp) {
        pred.knots_D = matrix(0, n.pred, q)

        for (i in 1:n.pred) {
          pred.knots_D[i,] = sqrt((pred_coords[i,1]-knot_coords[,1])^2 + (pred_coords[i,2]-knot_coords[,2])^2)
        } 
    } else {
        pred.obs.D = matrix(0, n.pred, n)

        for (i in 1:n.pred) {
            pred.obs.D[i,] = sqrt((pred_coords[i,1]-obs.coords[,1])^2 + (pred_coords[i,2]-obs.coords[,2])^2)
        } 
    }

    storage.mode(X) = "double"
    storage.mode(Y) = "double"
    storage.mode(is_pp) = "integer"
    storage.mode(modified.pp) = "integer"
    storage.mode(n) = "integer"
    storage.mode(m) = "integer"
    storage.mode(p) = "integer"
    storage.mode(q) = "integer"
    storage.mode(beta) = "double"
    storage.mode(A) = "double"
    storage.mode(phi) = "double"
    storage.mode(nu) = "double"
    storage.mode(n.pred) = "integer"
    storage.mode(pred_X) = "double"
    storage.mode(obs.D) = "double"
    storage.mode(pred.D) = "double"
    storage.mode(pred.obs.D) = "double"
    storage.mode(obs.knots_D) = "double"
    storage.mode(knots_D) = "double"
    storage.mode(pred.knots_D) = "double"
    storage.mode(n.samples) = "integer"
    storage.mode(w) = "double"
    storage.mode(w.str) = "double"
    storage.mode(sp.effects) = "integer"
    storage.mode(verbose) = "integer"

    out = .Call("spMvGLMPredict", family, X, Y, is_pp, modified.pp, n, m, p, q, beta, A, phi, nu,
               n.pred, pred_X, obs.D, pred.D, pred.obs.D, obs.knots_D, knots_D, pred.knots_D,
               cov.model, n.samples, w, w.str, sp.effects, verbose)
    out
}