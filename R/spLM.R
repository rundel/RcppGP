spLM2 = function(formula, data = parent.frame(), coords, knots,
                 starting, sp.tuning, priors, cov_model, 
                 modified.pp = TRUE, n.samples, sub.samples, verbose=TRUE, n.report=100, ...)
{

  ####################################################
  ##Check for unused args
  ####################################################
  formal.args = names(formals(sys.function(sys.parent())))
  elip.args = names(list(...))
  for(i in elip.args) {
    if (! i %in% formal.args)
      warning("'",i, "' is not an argument")
  }

  ####################################################
  ##formula
  ####################################################
  if (missing(formula)) stop("error: formula must be specified")
  
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
  
  ##make sure storage mode is correct
  storage.mode(Y) = "double"
  storage.mode(X) = "double"
  storage.mode(p) = "integer"
  storage.mode(n) = "integer"

  ####################################################
  ##Distance matrices
  ####################################################
  
  ####################
  ##Coords
  ####################
  if (missing(coords)) {stop("error: coords must be specified")}
  if (!is.matrix(coords)) {stop("error: coords must n-by-2 matrix of xy-coordinate locations")}
  if (ncol(coords) != 2 || nrow(coords) != n) {
    stop("error: either the coords have more than two columns or then number of rows is different than
          data used in the model formula")
  }
  
  coords_D = as.matrix(dist(coords))
  storage.mode(coords_D) = "double"
  
  ####################
  ##Knots
  ####################

  is_pp = TRUE
  modified.pp = TRUE

  if (!missing(knots)) {
    
    if (is.vector(knots) && length(knots) %in% c(2,3)) {
      
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

    for(i in 1:n) {
      coords_knots_D[,i] = sqrt((knot_coords[,1]-coords[i,1])^2+
                                 (knot_coords[,2]-coords[i,2])^2)
    }
    
    storage.mode(modified.pp) = "integer"
    storage.mode(m) = "integer"
    storage.mode(knots_D) = "double"
    storage.mode(coords_knots_D) = "double"
  }

  ####################################################
  ##Covariance model
  ####################################################
  if (missing(cov_model))
    stop("error: cov_model must be specified")
  if (!cov_model%in%c("gaussian","exponential","matern","spherical","powexp"))
    stop("error: specified cov_model '",cov_model,"' is not a valid option; choose, from gaussian, exponential, matern, spherical.")

  ####################################################
  ##Starting values
  ####################################################

  sigmaSq = rep(0,4)
  tauSq = rep(0,4)
  phi = rep(0,4)
  nu = rep(0,4)

  nugget = FALSE
  
  if (missing(starting)) {stop("error: starting value list for the parameters must be specified")}
  
  names(starting) = tolower(names(starting))   

  if (!"sigma.sq" %in% names(starting)) stop("error: sigma.sq must be specified in starting value list")
  if (!"phi" %in% names(starting))      stop("error: phi must be specified in starting value list")

  sigmaSq[1] = starting[["sigma.sq"]][1]
  phi[1] = starting[["phi"]][1]

  if ("tau.sq" %in% names(starting)) {
    tauSq[1] = starting[["tau.sq"]][1]
    nugget = TRUE
  }
  
  if (cov_model %in% c("matern","powexp")) {
    if (!"nu" %in% names(starting)) stop("error: nu must be specified in starting value list")
    nu[1] = starting[["nu"]][1]
  }

  
  ####################################################
  ##Priors
  ####################################################

  if (missing(priors)) stop("error: prior list for the parameters must be specified")
  names(priors) = tolower(names(priors))
  
  prior_list = c("sigma.sq.ig", "phi.unif")
  if(nugget) prior_list[length(prior_list)+1] = "tau.sq.ig"
  if (cov_model %in% c("matern","powexp")) prior_list[length(prior_list)+1] = "nu.unif"

  for(p in prior_list) {
    if (!p %in% names(priors)) 
      stop("error: ",p," must be specified")
    if (!is.vector(priors[[p]]) || length(priors[[p]]) != 2) 
      stop("error: ",p," must be a vector of length 2")
    if (any(priors[[p]] <= 0)) 
      stop("error: ",p," must be a positive vector of length 2")
  }

  sigmaSq[2:3] = priors[["sigma.sq.ig"]]
  phi[2:3] = priors[["phi.unif"]]
  
  if (nugget) tauSq[2:3] = priors[["tau.sq.ig"]]
  if (cov_model %in% c("matern","powexp")) nu[2:3] = priors[["nu.unif"]]

  ####################################################
  ##Tuning values
  ####################################################
  
  if (missing(sp.tuning)) {stop("error: sp.tuning value vector for the spatial parameters must be specified")}
  names(sp.tuning) = tolower(names(sp.tuning))
  
  if (!"sigma.sq" %in% names(sp.tuning)) stop("error: sigma.sq must be specified in tuning value list")
  if (!"phi" %in% names(sp.tuning)) {stop("error: phi must be specified in tuning value list")}
  
  sigmaSq[4] = sqrt(sp.tuning[["sigma.sq"]][1])
  phi[4] = sqrt(sp.tuning[["phi"]][1])
  
  if (nugget) {
    if (!"tau.sq" %in% names(sp.tuning)) stop("error: tau.sq must be specified in tuning value list")
    tauSq[4] = sqrt(sp.tuning[["tau.sq"]][1])
  }
  
    
  if (cov_model == "matern") {
    if (!"nu" %in% names(sp.tuning)) {stop("error: nu must be specified in tuning value list")}
    nu[4] = sqrt(sp.tuning[["nu"]][1])
  }    
  
  
  ####################################################
  ##Other stuff
  ####################################################
  if (missing(n.samples)) {stop("error: n.samples need to be specified")}

  if (missing(sub.samples)) {sub.samples = c(1, n.samples, 1)}
  if (length(sub.samples) != 3 || any(sub.samples > n.samples) ) {stop("error: sub.samples misspecified")}
  
  storage.mode(n.samples) = "integer"
  storage.mode(n.report) = "integer"
  storage.mode(verbose) = "integer"


  ####################################################
  ##Pack it up and off it goes
  ####################################################
              
  out = .Call("spmPPLM",
              Y, X, 
              coords_D, knots_D, coords_knots_D, 
              nugget,               
              sigmaSq, tauSq, nu, phi,
              cov_model, n.samples, verbose, n.report,
              PACKAGE = "tsBayes" )
  
  nParams = 2 + (nugget) + (cov_model %in% c("matern","powexp"))
  out$params = out$params[1:nParams,]

  out$p.samples = rbind(out$beta,out$params)
  #out$beta = NULL
  #out$params = NULL

  out$coords = coords
  out$is_pp = is_pp
  out$modified.pp = modified.pp
  
  if (is_pp)
    out$knot_coords = knot_coords
  
  out$Y = Y
  out$X = X
  out$n = n
  out$m = m
  out$p = p
  out$knots_D = knots_D
  out$coords_D = coords_D
  out$coords_knots_D = coords_knots_D
  out$cov_model = cov_model
  out$nugget = nugget
  out$verbose = verbose
  #out$n.samples = n.samples
  out$sub.samples = sub.samples
  out$recovered.effects = TRUE

  ##subsample
  out$sp.effects = out$sp.effects[,seq(sub.samples[1], sub.samples[2], by=as.integer(sub.samples[3]))]
  if (is_pp) {out$sp.effects.knots = out$sp.effects.knots[,seq(sub.samples[1], sub.samples[2], by=as.integer(sub.samples[3]))]}
  out$p.samples = mcmc(t(out$p.samples[,seq(sub.samples[1], sub.samples[2], by=as.integer(sub.samples[3]))]))
  out$n.samples = nrow(out$p.samples)##get adjusted n.samples
  
  colnames(out$p.samples) = c(x.names, c("sigma.sq", "tau.sq", "phi", "nu")[1:nParams])
  
  class(out) = "spLM"

  return(out)
}

