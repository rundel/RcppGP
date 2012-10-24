library(magic)

mkY <- function(y){
  m <- length(y)
  n <- nrow(y[[1]])
  Y <- matrix(t(do.call('cbind', y)), n*m, 1)
  Y
}

mkX <- function(x){
  m <- length(x)
  q <- do.call(sum, lapply(x, ncol))
  n <- nrow(x[[1]])  
  X <- matrix(0,0,q)

  for(i in 1:n){

    xi <- vector("list", m)
    
    for(j in 1:m){
      xi[[j]] <- array(x[[j]][i,], c(1, length(x[[j]][i,])))
    }
    X <- rbind(X, do.call('adiag', xi))
  }
  X
}


parseFormulaSimple <- function(formula, data, na.action = na.fail){

  m <- model.frame(formula, data, na.action = na.action)##kinda fragile
  Y <- as.matrix(model.response(m))
  X <- as.matrix(model.matrix(formula, m))

  if(any(is.na(X))){stop("error: parseFormulaSimple, NA found in model.matrix")}
  
  xvars <- dimnames(X)[[2]]
  xobs  <- dimnames(X)[[1]]
  list(Y, X, xvars, xobs)
  
}


parseFormula <-  function(formula, data, intercept=TRUE, justX=FALSE){
    
    # extract Y, X, and variable names for model formula and frame
    mt <- terms(formula, data=data)
    if(missing(data)) data <- sys.frame(sys.parent())
    mf <- match.call(expand.dots = FALSE)
    mf$intercept <- mf$justX <- NULL
    mf$drop.unused.levels <- TRUE
    mf[[1]] <- as.name("model.frame")
    mf <- eval(mf, sys.frame(sys.parent()))
    if (!intercept){
      attributes(mt)$intercept <- 0
    }

    # null model support
    X <- if (!is.empty.model(mt)) model.matrix(mt, mf, contrasts)
    X <- as.matrix(X)         # X matrix
    xvars <- dimnames(X)[[2]] # X variable names
    xobs  <- dimnames(X)[[1]] # X observation names
    if (justX){
      Y <- NULL
    }
    else {
      Y <- as.matrix(model.response(mf, "numeric")) # Y matrix
    }
    return(list(Y, X, xvars, xobs))
  }



mkMats <- function(mods, data){

  Y <- vector("list", length(mods))
  X <- vector("list", length(mods))
  X.names <- numeric(0)
  res <- vector("list", 3)
  names(res) <- c("Y", "X", "X.names")
  
  for(i in 1:length(mods)){
    tmp <- parseFormula(mods[[i]], data)
    ##get Y
    Y[[i]] <- tmp[[1]]

    ##get X
    X[[i]] <- tmp[[2]]

    ##get names
    tmp[[3]] <- paste(tmp[[3]], ".mod", i, sep="")
    X.names <- c(X.names,tmp[[3]])
  }

  res[[1]] <- mkY(Y)
  res[[2]] <- mkX(X)
  res[[3]] <- X.names
  
  res
}

mkMvY <- function(y){
  mkY(y)
}

mkMvX <- function(X){
  mkX(X)
}

mkMvFormulaYX <- function(mods, data){
  mkMats(mods, data)
}
