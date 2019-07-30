



#'
#' penMlogistic
#'
#' This function computes a penalized and robust estimator for the Logistic Regression Model.
#' @usage penMlogistic(loss = c("ch", "deviance", "divergence", "ls"), penalty = c("sign","en","scad","mcp"), X, Y, use.weights = F, alpha = NA, estimate.intercept = T, lambda.grid = NULL, nlambda = 12, use.importance.sets = T, lambda.grid.eps = 0.2, nfolds = 10, verbose = F, algorithm.method = c("cd", "hjk"))
#' @param X The design matrix. Columns should be standarized for better algorithmical stability.
#' @param Y The response vector. Each coordinate must be either 0 or 1.
#' @param loss The loss function to be used (see Details for futher information). Defaults to "ch".
#' @param penalty The penalty function to be used (see Details for futher information). Defaults to "sign".
#' @param use.weights If TRUE, the method uses the hard weights described in Bianco et al. (2019). Defaults to FALSE.
#' @param alpha An additional tuning parameter used for certain penalization functions.
#' @param estimate.intercept If TRUE, the method also estimates an intercept parameter. Defaults to TRUE.
#' @param lambda.grid A vector of candidates for the tuning parameter lambda. Robust Cross-Validation will be performed over this grid to select the optimal value for lambda. If no value is provided, the grid is chosen as described in Bianco et al. (2019).
#' @param nlambda The number of elements of the lambda grid to be generated (only if no value is provided for \code{lambda.grid}). Defaults to 12.
#' @param use.importance.sets If TRUE, it implements an heuristic version of the Strong Safe Rules (Tibshirani et al. (2012)) as described in Bianco et al. (2019). Used to improve run time. Defaults to TRUE.
#' @param lambda.grid.eps The ratio between the smallest and largest values of the lambda grid. Defaults to 0.2.
#' @param nfolds The number of Cross Validation folds to be used. Defaults to 10.
#' @param algorithm.method The algorithm to be used (see Details for futher information). Defaults to "cd".
#' @return A list with the following elements:
#' \describe{
#'   \item{all.lambdas}{The used lambda grid}
#'   \item{all.betas}{The matrix whose rows are the estimated coefficient vectors for each value of the lambda grid}
#'   \item{all.betas0}{The estimated intercept for each value of the lambda grid}
#'   \item{all.df}{The number of estimated nonzero coefficients for each value of the lambda grid}
#'   \item{beta}{The estimated coefficient vector (beta) for the chosen value of the lambda grid}
#'   \item{beta0}{The estimated intercept (beta0) for the chosen value of the lambda grid}
#'   \item{lambda}{The chosen value of the lambda grid}
#' }
#' @references Tibshirani, R., Bien, J., Friedman, J., Hastie, T., Simon, N., Taylor, J., & Tibshirani, R. J. (2012). Strong rules for discarding predictors in lasso-type problems. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 74(2), 245-266.
#' @references Chi, E. C., & Scott, D. W. (2014). Robust parametric classification and variable selection by a minimum distance criterion. Journal of Computational and Graphical Statistics, 23, 111-128.
#' @references Croux, C., & Haesbroeck, G. (2003). Implementing the Bianco and Yohai estimator for logistic regression. Computational statistics \& data analysis, 44, 273-295.
#' @references Fan, J., & Li, R. (2001). Variable selection via non--concave penalized likelihood and its oracle properties. Journal of the American Statistical Association, 96, 1348-1360.
#' @references Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society: Series B (Statistical Methodology, 67(2), 301-320.
#' @references Zhang, C. H. (2010). Nearly unbiased variable selection under minimax concave penalty. The Annals of statistics, 38, 894-942.
#' @references Basu, A., Gosh, A., Mandal, A., Martin, N. & Pardo, L. (2017). A Wald--type test statistic for testing linear hypothesis in logistic regression models based on minimum density power divergence estimator. Electronic Journal of Statistics, 11, 2741-2772.
#' @details The loss function must be one value from the vector ("ch", "deviance", "divergence", "ls"). The "ch" function corresponds to the one defined in Croux and Haesbroeck (2003), which guarantees the existence of the unpenalized estimator under overlapping conditions. The "deviance" loss is the classic unbounded function used in the maximum likelihood estimator. The "divergence" loss function is the one defined in Basu et al. (1998). Finally, the "ls" loss functions corresponds to the least squares estimator used in Chi and Scott (2014). 
#' @details The penalty function must be one value from the vector ("sign","en","scad","mcp"). The "sign" penalty is the one defined in Bianco et al. (2019). Moreover, "en" corresponds to the elastic net function defined in Hastie and Zou (2005) and the additional tuning parameter alpha must be defined (and must lie between 0 and 1). When alpha = 1, this corresponds to the Lasso penalty and when alpha = 0, it corresponds to the Ridge penalization. On the other hand, "scad" is the penalty defined in Fan and Li (2001) and the additional tuning parameter alpha must be supplied and greater than 2. Finally, the "mcp" penalty is the one defined in Zhang (2010). In this case, the tuning parameter alpha should also be supplied and must be greater than 1.  
#' @details The algorithm method must be one value from the vector ("cd","hjk"). The "cd" algorithm corresponds to the implementation of the cyclical descent described in Bianco et al. (2019). The "hjk" algorithm uses the Hooke-Jeeves derivative-free minimization algorithm (from the dfoptim package).
#' @import mvtnorm
#' @import Rcpp
#' @import RcppArmadillo
#' @import robustbase
#' @import rrcov
#' @import rrcovHD
#' @import robust
#' @import Gmedian
#' @import glasso
#' @import dfoptim
#' @useDynLib penMlogistic
#' @examples
#' X = scale(matrix(rnorm(2000), 200, 10))
#' beta = c(rep(1,3), rep(0,7))
#' probabilities = 1 / (1 + exp(-X %*% beta))
#' Y = rbinom(n = 200, size = 1, prob = probabilities)
#' penMlogistic(loss = "divergence", penalty = "sign", X = X, Y = Y) 
#' @export
penMlogistic = function(loss = "ch", penalty = "sign", X, Y, use.weights = F, alpha = NA, estimate.intercept = T,
                      lambda.grid = NULL, nlambda = 12, use.importance.sets = T, lambda.grid.eps = 0.2,
                      nfolds = 10, verbose = F, algorithm.method = "cd"){
  n = nrow(X)
  p = ncol(X)
  
  if(penalty == "en" && (is.na(alpha) || alpha < 0 || alpha > 1)){
    stop("The tuning parameter alpha must be defined and lie between 0 and 1 for the Elastic Net Penalty.")
  }
  if(penalty == "mcp" && (is.na(alpha) || alpha <= 1)){
    stop("The tuning parameter alpha must be defined and greater than 1 for the MCP Penalty.")
  }
  if(penalty == "scad" && (is.na(alpha) || alpha <= 2)){
    stop("The tuning parameter alpha must be defined and greater than 2 for the SCAD Penalty.")
  }

  ## Obtain weights
  if(use.weights){
    weights = get.weights.box(X)
    if(verbose){
      print("###################### WEIGHTS ######################")
      print(weights)
      print("#####################################################")
      Sys.sleep(2)
    }
  } else{
    weights = rep(1,n)
  }



  ## Obtain penalty, loss and loss derivative functions
  penalty.fixed.alpha.info = get.penalty.fixed.alpha(penalty, alpha)
  penalty.fixed.alpha = penalty.fixed.alpha.info$pen
  penalty.scale.invariant = penalty.fixed.alpha.info$scale.invariant
  penalty.aditive = penalty.fixed.alpha.info$aditive
  penalty.gradient = penalty.fixed.alpha.info$der.pen

  loss.and.der.loss = get.loss.and.der.loss(loss)
  loss.fun = loss.and.der.loss$loss.fun
  der.loss.fun = loss.and.der.loss$der.loss.fun



  ## Obtain importance set function
  importance.set.function = get.importance.set.function(penalty = penalty, der.loss.fun = der.loss.fun)

  ## Obtain initial estimator
  init.beta.est = suppressWarnings(get.robust.initial.2(X,Y))
  if(verbose){
    print("###################### INITIAL ESTIMATOR ######################")
    print("Beta:")
    print(init.beta.est$beta.init)
    print("Intercept:")
    print(init.beta.est$beta0.init)
    print("###############################################################")
    Sys.sleep(2)
  }

    ## Obtain lambda grid
  if(is.null(lambda.grid)){
    lambda.max = get.lambda.max.lasso(X = X, Y = Y, weights = weights)
    # lambda.max = get.lambda.max.2(X = X, Y = Y, weights = weights, loss.function = loss.fun,
    #                               pen.function = penalty.fixed.alpha,init.beta.est = init.beta.est)

    lambda.grid = 2**seq(log2(lambda.max), log2(lambda.max * lambda.grid.eps), length.out = nlambda)
  }
  if(verbose){
    print("###################### LAMBDA GRID ######################")
    print(lambda.grid)
    print("#########################################################")
    Sys.sleep(2)
  }

  ## Run Cross Validation
  # return(cv(loss.function = loss.fun, der.loss.function = der.loss.fun,
  #           pen.function = penalty.fixed.alpha, der.pen = penalty.gradient, init.beta.est = init.beta.est,
  #           importance.set.function = importance.set.function, X = X, Y = Y,
  #           weights = weights, nfolds = nfolds, lambda.grid = lambda.grid,
  #           estimate.intercept = estimate.intercept, pen.scale.invariant = penalty.scale.invariant,
  #           pen.aditive = penalty.aditive, use.importance.sets = use.importance.sets, verbose = verbose,
  #           algorithm.method = algorithm.method))
  return(cv(loss.function = loss.fun, der.loss.function = der.loss.fun,
            pen.function = penalty.fixed.alpha, init.beta.est = init.beta.est,
            importance.set.function = importance.set.function, X = X, Y = Y,
            weights = weights, nfolds = nfolds, lambda.grid = lambda.grid,
            estimate.intercept = estimate.intercept, pen.scale.invariant = penalty.scale.invariant,
            pen.aditive = penalty.aditive, use.importance.sets = use.importance.sets, verbose = verbose,
            algorithm.method = algorithm.method))
}


##############################
## Initial estimator functions
##############################

get.robust.initial.2 = function(X,Y, prop = 0.1, trim = 0.15){
  n = nrow(X)
  p = ncol(X)
  scores = rep(NA,p)
  p.bar = mean(Y)
  for(j in 1:p){
    scores[j] = p.bar * (1-p.bar) * abs(mean(X[,j] * (Y - p.bar), trim = trim))
  }
  n.active = ceiling(p * prop)
  active = order(-scores)[1:n.active]
  ret = rep(0,p)
  full.fit.coef = try(glmrob(Y~X[,active], family = "binomial", method = "WBY")$coefficients)
  if(class(full.fit.coef) == "try-error" || is.na(full.fit.coef[1])){
    full.fit.coef = c(0,rep(1,n.active))
  }
  ret[active] = full.fit.coef[-1]
  return(list(beta.init = ret, beta0.init = full.fit.coef[1]))
}

####################
## Weights functions
####################

get.rob.hd.cov.estimate = function(X, min.mad = 0.01){
  p = ncol(X)
  kendall.cor = cor(X, method = "kendall")
  mads = apply(X = X, MARGIN = 2, FUN = mad)
  for(j in 1:p){
    mads[j] = max(mads[j], min.mad)
  }

  kendall.cov = matrix(NA, p,p)
  for(j in 1:p){
    for(k in 1:j){
      kendall.cov[j,k] = kendall.cov[k,j] = mads[k] * mads[j] * kendall.cor[j,k]
    }
  }
  return(kendall.cov)
}

get.weights.box = function(X){
  cov = get.rob.hd.cov.estimate(X)
  med = Gmedian(X)
  prec = glasso(cov, rho = 0.1)$w
  dist = rep(NA, nrow(X))
  for(i in 1: nrow(X)){
    dist[i] = sqrt((X[i,] - med) %*% prec %*% t((X[i,] - med)))
  }
  q3 = quantile(dist,0.75)
  q1 = quantile(dist,0.25)
  lbox = q3 - q1
  weights = rep(NA,nrow(X))
  for(i in 1: nrow(X)){
    weights[i] = 1*(dist[i] < q3 + 1.5*lbox)
  }
  return(weights)
}


#######################
## Lambda max functions
#######################

get.lambda.max.lasso = function(X,Y, weights){
  ret = 0
  for(i in 1: ncol(X)){
    ret = max(ret, abs(get.loss.der.for.coord.bypen(X = X, Y = Y, beta = rep(0, ncol(X)),
                                                    beta0 = 0, coord = i, weights = weights)))
  }
  ## TODO: ver si es mejor devolver ret o 2*ret
  return(ret)
}

get.loss.der.for.coord.bypen = function(X,Y,beta,beta0,coord, weights){
  scores = X %*% beta + beta0
  return(mean(X[,coord] * weights * mapply(der_phi_ch_cpp, score = scores, y = Y)))
}

# get.lambda.max.2 = function(X,Y, weights, loss.function, pen.function, init.beta.est, max.multip = 6, factor = 1.5){
#   init.lambda = get.lambda.max.lasso(X,Y, weights)
#   dfs = rep(NA,max.multip)
#   df.init = sum(init.beta.est$beta.init != 0)
#   current.lambda = init.lambda
#   current.df = df.init
#   for(i in 1:max.multip){
#     df = get.df.for.lambda(X,Y,current.lambda, weights, loss.function, pen.function, init.beta.est)
#     print(c("Current lambda: ", current.lambda))
#     print(c("DF: ", df))
#     if(df >= current.df){
#       if(i >= 2){
#         return(current.lambda / factor)
#       } else{
#         return(current.lambda)
#       }
#     }
#     dfs[i] = df
#     current.lambda = factor * current.lambda
#     current.df = df
#   }
#   index = min(which(dfs == min(dfs)))
#   return(init.lambda * (factor ** (index - 1)))
# }
#
# get.df.for.lambda = function(X,Y,lambda, weights, loss.function, pen.function, init.beta.est){
#   p = ncol(X)
#   beta = hjkb(par = rep(0,p+1), fn = f, loss.function = loss.function, weights = weights,
#             pen.function = pen.function, X = X, Y = Y, lambda = lambda, lower = -20, upper = 20, control = list(tol = 1e-2))$par
#   print(beta)
#   return(sum(beta!= 0))
# }



###########################
## Importance set functions
###########################

get.importance.set.function = function(penalty,der.loss.fun){

  importance.set.function = function(X,Y,beta,beta0,lambda, weights){
    if(is.null(beta0)){
      beta0 = 0
    }
    p = ncol(X)
    norm2 = sqrt(sum(beta**2))
    norm1 = sum(abs(beta))
    importance = der.loss.fun(X = X, Y = Y, beta = beta, beta0 = beta0, weights = weights)
    importance[beta!= 0] = Inf

    if(identical(penalty, "sign")){
      importance = importance * norm2
      tol = 1.2
    } else{
      tol = 2
    }

    return(which(abs(importance) > (lambda / tol)))
  }

  return(importance.set.function)
}

##############################
## Error computation functions
##############################

get.errors = function(X,Y,probs,new.X,new.Y,new.probs,beta.est,beta0.est, true.beta){
  scores.oos = cbind(1,new.X) %*% c(beta0.est,beta.est)
  probs.est.oos = inv.logit(scores.oos)
  class.est.oos = 1*(probs.est.oos > (1/2))
  probs.mse.oos = mean((new.probs - probs.est.oos)**2)
  correct.prop.oos = sum(class.est.oos == new.Y) / length(new.Y)

  scores.is = cbind(1,X) %*% c(beta0.est,beta.est)
  probs.est.is = inv.logit(scores.is)
  class.est.is = 1*(probs.est.is > (1/2))
  probs.mse.is = mean((probs - probs.est.is)**2)
  correct.prop.is = sum(class.est.is == Y) / length(Y)

  beta.error = get.beta.error(beta.est = beta.est, true.beta = true.beta)
  return(list(probs.mse.oos = probs.mse.oos, correct.prop.oos = correct.prop.oos, probs.mse.is = probs.mse.is, correct.prop.is = correct.prop.is, beta.se = beta.error$sq.error, tn = beta.error$tn, tp = beta.error$tp))
}

# Returns errors from comparing true beta with estimated beta
get.beta.error = function(beta.est, true.beta){
  p = length(beta.est)
  sq.error = sum((beta.est - true.beta)^2)
  active.set = which(true.beta != 0)
  inactive.set = which(true.beta == 0)
  active.set.est = which(beta.est != 0)
  inactive.set.est = which(beta.est == 0)
  true.pos = length(which(is.element(active.set.est,active.set))) / length(active.set)
  true.neg = length(which(is.element(inactive.set.est,inactive.set))) / length(inactive.set)
  return(list(sq.error = sq.error, tp = true.pos , tn = true.neg))
}

#######
## Optimization algorithm
#######

# Univariate function to minimize
# t is the value of the selected coordinate
# X, Y are the data covariate and responses
# lambda is the penalization parameter
# to.min.index is the coordinate to minimize
# beta, beta0 are current values of the whole vector (all coordinates but to.min.index remain constant with the values of beta)
univ.function.to.min = function(t,loss.function, weights, pen.function,X,Y,lambda,to.min.index,beta, beta0, pen.aditive){
  beta[to.min.index] = t
  if(!pen.aditive){
    return(loss.function(X,Y,beta,beta0,weights) + pen.function(beta = beta, lambda = lambda))
  } else{
    return(loss.function(X,Y,beta,beta0,weights) + pen.function(beta = beta, j = to.min.index, lambda = lambda))
  }
}

# Function defined to minimize the scale of the whole beta vector.
# c is the corresponding scale
scale.function.to.min = function(c,loss.function, weights, pen.function,X,Y,lambda,beta, beta0){
  beta = c*beta
  return(loss.function(X,Y,beta,beta0, weights) + pen.function(beta = beta, lambda = lambda))
}

# The same as scale.funcion.to.min, without the penalization term
scale.function.to.min.nopen = function(c,loss.function, weights, X,Y,beta, beta0){
  beta = c*beta
  return(loss.function(X,Y,beta,beta0,weights))
}

# Evaluates the function to minimize in a given beta and intercept
evaluate = function(loss.function, weights, pen.function,X,Y,lambda,beta, beta0){
  return(scale.function.to.min(c = 1,loss.function = loss.function, weights = weights, pen.function = pen.function,
                               X = X, Y = Y, lambda = lambda, beta = beta, beta0 = beta0))
}


# Performs optimization on a single coordinate
update.one.coord = function(loss.function, der.loss.function, weights, pen.function, der.pen, X, Y, lambda, current.beta,
                            current.beta0, index.to.update, pen.aditive){

  opt = optim(fn = univ.function.to.min, par = current.beta[index.to.update],
              loss.function = loss.function, weights = weights, pen.function = pen.function,
              X = X, Y = Y, lambda = lambda, to.min.index = index.to.update, beta = current.beta,
              beta0 = current.beta0, pen.aditive = pen.aditive)

  opt.value = opt$value
  opt.par = opt$par
  zero.value = univ.function.to.min(t = 0, loss.function = loss.function, weights = weights,
                                    pen.function = pen.function, X = X, Y = Y, lambda = lambda,
                                    beta = current.beta, beta0 = current.beta0, to.min.index = index.to.update,
                                    pen.aditive = pen.aditive)
  if(zero.value > opt.value){
    current.beta[index.to.update] = opt.par
  } else{
    current.beta[index.to.update] = 0
  }
  # print(current.beta)
  return(current.beta)

}


# Optimizes the intercept
optim.intercept = function(loss.function, weights, X,Y,beta){
  fun = function(beta0){
    return(loss.function(X,Y,beta,beta0,weights))
  }
  return(optim(par = 0, fn = fun)$par)
}



# Performs onle cycle of minimizations. Includes:
# - Minimizing (in random order) over all covariates
# - Minimizing intercept
# - Minimizing scale of beta
one.cycle.descent = function(loss.function, weights, der.loss.function, pen.function, der.pen,
                             X,Y, lambda,beta.init,beta0.init, importance.set, estimate.intercept,
                             pen.scale.invariant, pen.aditive){
  p = ncol(X)
  current.beta = beta.init
  beta0 = 0

  if(!is.null(beta0.init)){
    beta0 = beta0.init
  }

  order = sample(1:p)
  coord.old.value = evaluate(loss.function = loss.function, weights = weights,
                             pen.function = pen.function, X = X, Y = Y, lambda = lambda,
                             beta = current.beta, beta0 = beta0)
  for(j in 1:p){
    coord.to.update = order[j]
    if(coord.to.update %in% importance.set){

      current.beta = update.one.coord(loss.function = loss.function, der.loss.function = der.loss.function, weights = weights,
                                      pen.function = pen.function, der.pen = der.pen,
                                      X = X, Y = Y, lambda = lambda, current.beta = current.beta,current.beta0 = beta0,
                                      index.to.update = coord.to.update, pen.aditive = pen.aditive)
    }
  }

  if(estimate.intercept){
    beta0 = optim.intercept(loss.function = loss.function, weights = weights, X = X, Y = Y, beta = current.beta)
  }
  opt.scale = optim.scale(loss.function = loss.function, weights = weights,
                          der.loss.function = der.loss.function, pen.function = pen.function,
                          lambda = lambda, X = X, Y = Y, beta = current.beta,
                          beta0 = beta0, pen.scale.invariant = pen.scale.invariant)
  return(list(beta = opt.scale * current.beta, beta0 = beta0))
}


# Complete minimization method. Stops ciclying when improvement ratio is less than tol.
cyclical.descent = function(loss.function, der.loss.function, pen.function, der.pen, X,Y, weights, lambda,beta.init,importance.set, max.cycles = 100,
                            tol = 1e-2, beta0.init, estimate.intercept, pen.scale.invariant, pen.aditive, verbose = F){
  cycle.index = 1
  current.beta = beta.init
  current.beta0 = beta0.init

  old.opt.value = evaluate(loss.function = loss.function, weights = weights, pen.function = pen.function,
                           beta = current.beta, Y = Y, X = X, lambda = lambda, beta0 = beta0.init)
  stop = F
  while(!stop && cycle.index <= max.cycles){
    cycle.index = cycle.index + 1
    cycle.optim = one.cycle.descent(loss.function = loss.function, weights = weights,
                                    der.loss.function = der.loss.function, pen.function = pen.function,
                                    der.pen = der.pen, X = X, Y = Y, lambda = lambda, beta.init = current.beta,
                                    beta0.init = current.beta0, importance.set = importance.set,
                                    estimate.intercept = estimate.intercept, pen.scale.invariant = pen.scale.invariant,
                                    pen.aditive = pen.aditive)
    current.beta = cycle.optim$beta

    current.beta0 = cycle.optim$beta0

    opt.value = evaluate(loss.function = loss.function, weights = weights, pen.function = pen.function,beta = current.beta, beta0 = current.beta0, Y = Y, X = X, lambda = lambda)

    if(opt.value == 0 || (old.opt.value - opt.value)/ opt.value < tol){
      stop = T
    }
    old.opt.value = opt.value
  }
  if(verbose){
    print(paste("Number of cycles to optimization : ", cycle.index))
  }

  return(list(beta = current.beta, beta0 = current.beta0, value = opt.value))
}

f = function(beta0.beta, loss.function, weights, pen.function, X, Y, lambda, estimate.intercept = T){
  beta0 = beta0.beta[1]
  beta = beta0.beta[-1]
  # print("###################")
  # print(loss.function(X,Y,beta,beta0,weights))
  # print(pen.function(beta = beta, lambda = lambda))
  return(loss.function(X,Y,beta,beta0,weights) + pen.function(beta = beta, lambda = lambda))
}

# Returns a complete list with fitted models for all lambdas and chooses one model with Cross-Validation.
cv = function(loss.function, der.loss.function, pen.function, der.pen, init.beta.est,
              importance.set.function, X,Y, weights, nfolds = 10, lambda.grid, estimate.intercept = T,
              pen.scale.invariant, pen.aditive, use.importance.sets = T, verbose = F, algorithm.method = "cd"){

  n = nrow(X)
  p = ncol(X)
  nlambda = length(lambda.grid)
  folds = separate.folds(n, nfolds = nfolds)
  cv.matrix = matrix(NA, nfolds, nlambda)

  for(iSet in 1:nfolds){
    if(verbose){
      print("###########################################")
      print(paste("Fold : ", iSet))
    }
    X.train = X[-folds[[iSet]],]
    Y.train = Y[-folds[[iSet]]]
    X.test = X[folds[[iSet]],]
    Y.test = Y[folds[[iSet]]]
    weights.train = weights[-folds[[iSet]]]
    weights.test = weights[folds[[iSet]]]

    beta.init = init.beta.est$beta.init
    beta0.init = NULL

    if(estimate.intercept){
      beta0.init = init.beta.est$beta0.init
    }

    importance.set = 1:p

    for(i in 1:nlambda){
      if(verbose){
        print("########################")
        print(paste("Lambda: ", lambda.grid[i]))
      }

      if(identical(algorithm.method, "hjk")){
        beta0.beta.info = hjkb(par = c(beta0.init, beta.init), fn = f, loss.function = loss.function, weights = weights.train,
                               pen.function = pen.function, X = X.train, Y = Y.train, lambda = lambda.grid[i], lower = -20, upper = 20, control = list(tol = 1e-2, maxfeval = 10000))
        beta0.beta = sparse.hjk(beta0.beta = beta0.beta.info$par, loss.function = loss.function, weights = weights,
                                pen.function = pen.function, X = X, Y = Y, lambda = lambda.grid[i])
        # beta0.beta = hjkb(par = rep(0,p+1), fn = f, loss.function = loss$loss.fun, weights = weights.train,
        #                   pen.function = pen$pen, X = X.train, Y = Y.train, lambda = lambda.grid[i], lower = -20, upper = 20, control = list(tol = 1e-2))$par
        lambda.beta = beta0.beta[-1]
        lambda.beta0 = beta0.beta[1]
        cv.matrix[iSet,i] = loss.function(X = X.test, Y = Y.test, beta = lambda.beta, beta0 = lambda.beta0, weights = weights.test)
        if(verbose){

          print(c("DF: ", sum(1*(lambda.beta != 0))))
          print(beta0.beta.info$niter)
          print(beta0.beta.info$feval)
          print(lambda.beta)
        }
      } else{

        lambda.results = cyclical.descent(loss.function = loss.function, der.loss.function = der.loss.function,
                                          pen.function = pen.function, der.pen = der.pen, X = X.train, Y = Y.train, weights = weights.train,
                                          lambda = lambda.grid[i],importance.set = importance.set, beta.init = beta.init,
                                          beta0.init = beta0.init, estimate.intercept = estimate.intercept,
                                          pen.scale.invariant = pen.scale.invariant, pen.aditive = pen.aditive,
                                          verbose = verbose)
        lambda.beta = lambda.results$beta
        lambda.beta0 = lambda.results$beta0
        cv.matrix[iSet,i] = loss.function(X = X.test, Y = Y.test, beta = lambda.beta, beta0 = lambda.beta0, weights = weights.test)
        if(use.importance.sets){
          importance.set = importance.set.function(X = X.train, Y = Y.train, beta = lambda.beta,
                                                   beta0 = lambda.beta0, lambda = lambda.grid[i], weights = weights.train)
        }
        if(verbose){
          print("Estimated beta: ")
          print(lambda.beta)
          print("Estimated intercept: ")
          print(lambda.beta0)
          print("New importance set: ")
          print(importance.set)
        }
      }
    }
  }
  return(get.fit.summary(loss.function, weights = weights, der.loss.function, pen.function, der.pen, init.beta.est,
                         importance.set.function, cv.matrix = cv.matrix, X = X, Y = Y, lambda.grid = lambda.grid,
                         estimate.intercept = estimate.intercept, pen.scale.invariant = pen.scale.invariant,
                         pen.aditive = pen.aditive, use.importance.sets = use.importance.sets, algorithm.method = algorithm.method))
}

sparse.hjk = function(beta0.beta, loss.function, weights, pen.function, X, Y, lambda){
  p = ncol(X)
  value = f(beta0.beta = beta0.beta, loss.function = loss.function, weights = weights,
            pen.function = pen.function,X = X, Y = Y, lambda = lambda)
  sparse.set = c()
  for(j in 1:p){
    if(beta0.beta[j+1] != 0){
      new = beta0.beta
      new[j+1] = 0
      new.value = f(beta0.beta = new, loss.function = loss.function, weights = weights,
                    pen.function = pen.function,X = X, Y = Y, lambda = lambda)
      if(new.value <= value){
        sparse.set = c(sparse.set, j+1)
      }
    }
  }
  ret = beta0.beta
  ret[sparse.set] = 0
  return(ret)
  return()
}


# Given a CV matrix, returns the chosen beta
get.fit.summary = function(loss.function,weights, der.loss.function, pen.function, der.pen, init.beta.est, importance.set.function,
                           cv.matrix, X,Y,lambda.grid, estimate.intercept, pen.scale.invariant, pen.aditive, use.importance.sets,
                           algorithm.method){



  p = ncol(X)
  nlambda = length(lambda.grid)
  means = apply(cv.matrix,2,mean)
  min.index = min(which(means == min(means)))

  if(identical(algorithm.method, "hjk")){
    all.betas = matrix(NA,nlambda,p)
    all.betas0 = rep(NA,nlambda)
    all.df = rep(NA,nlambda)
    for(i in 1:nlambda){
      lambda = lambda.grid[i]
      beta0.beta = hjkb(par = c(init.beta.est$beta0.init, init.beta.est$beta.init), fn = f, loss.function = loss.function,
                        weights = weights, pen.function = pen.function, X = X, Y = Y,
                        lambda = lambda, lower = -20, upper = 20, control = list(tol = 1e-2, maxfeval = 10000))$par
      beta0.beta = sparse.hjk(beta0.beta = beta0.beta, loss.function = loss.function, weights = weights,
                              pen.function = pen.function, X = X, Y = Y, lambda = lambda)
      all.betas[i,] = beta0.beta[-1]
      all.betas0[i] = beta0.beta[1]
      all.df[i] = sum(all.betas[i,] != 0)
    }
    
    chosen.lambda = lambda.grid[min.index]

    return(list(all.lambdas = lambda.grid, all.betas = all.betas, all.betas0 = all.betas0, all.df = all.df,
                beta = beta0.beta[-1], beta0 = beta0.beta[1], lambda = chosen.lambda))

  } else{
    current.beta = init.beta.est$beta.init
    current.beta0 = NULL
    if(estimate.intercept){
      current.beta0 = init.beta.est$beta0.init
    }


    all.betas = matrix(NA, nlambda, p)
    all.betas0 = rep(NA, nlambda)
    all.loss.eval = rep(NA,nlambda)
    all.df = rep(NA,nlambda)


    importance.set = 1:p
    for(i in 1: nlambda){

      opt = cyclical.descent(loss.function = loss.function, weights = weights, der.loss.function = der.loss.function,
                             pen.function = pen.function, der.pen = der.pen, X = X, Y = Y, lambda = lambda.grid[i], importance.set = importance.set,
                             beta.init = current.beta, beta0.init = current.beta0, estimate.intercept = estimate.intercept,
                             pen.scale.invariant = pen.scale.invariant, pen.aditive = pen.aditive)
      opt.beta = opt$beta
      opt.beta0 = opt$beta0

      if(use.importance.sets){
        importance.set = importance.set.function(X = X, Y = Y, beta = opt.beta, lambda = lambda.grid[i], beta0 = opt.beta0, weights = weights)
      }

      all.betas[i,] = opt.beta
      all.betas0[i] = opt.beta0
      all.loss.eval[i] = loss.function(X = X, Y = Y, beta = opt.beta, beta0 = opt.beta0, weights = weights)
      all.df[i] = sum(opt.beta != 0)

    }
    return(list(all.lambdas = lambda.grid, all.betas = all.betas, all.betas0 = all.betas0, all.df = all.df, beta = all.betas[min.index,], beta0 = all.betas0[min.index], lambda = lambda.grid[min.index]))
  }
}


# Returns the scale for beta that minimizes the objetive function
optim.scale = function(loss.function, weights, der.loss.function, pen.function, der.pen, lambda,X,Y,beta,beta0, pen.scale.invariant){
  if(pen.scale.invariant){
    scale.derivative = function(c){
      scores = c * X %*% beta + beta0
      der.phis = mapply(FUN = der.loss.function, score = scores, y = Y, weights = weights)
      return(mean(der.phis * scores))
    }
    return(optim(fn = scale.function.to.min.nopen, par = 1, gr = scale.derivative, loss.function = loss.function, weights = weights, X = X, Y = Y, beta = beta, beta0 = beta0)$par)
  } else{
    return(optim(fn = scale.function.to.min, par = 1, loss.function = loss.function, weights = weights, X = X, Y = Y, beta = beta, beta0 = beta0, pen.function = pen.function, lambda = lambda)$par)
  }
}


########
## Penalties
########


norm.quotient = function(beta, lambda){
  return(norm_quotient_cpp(beta = beta, lambda = lambda))
}


elastic.net = function(beta, alpha, j = NULL, lambda){
  if(is.null(j)){
    return(sum(mapply(FUN = elastic_net_cpp, t = beta, lambda = lambda, gamma = alpha)))
  } else{
    return(elastic_net_cpp(t = beta[j], lambda = lambda, gamma = alpha))
  }
}

scad = function(beta, alpha, j = NULL, lambda){
  if(is.null(j)){
    return(sum(mapply(FUN = scad_cpp, t = beta, lambda = lambda , gamma = alpha)))
  } else{
    return(scad_cpp(t = beta[j], lambda = lambda, gamma = alpha))
  }
}

mcp = function(beta, alpha, j = NULL, lambda){
  if(is.null(j)){
    return(sum(mapply(FUN = mcp_cpp, t = beta, lambda = lambda , gamma = alpha)))
  } else{
    return(mcp_cpp(t = beta[j], lambda = lambda, gamma = alpha))
  }
}

get.penalty.fixed.alpha = function(penalty, alpha){

  ## Penalizacion ELASTIC NET
  if(identical(penalty, "en")){
    if(is.null(alpha)){
      alpha = 1
    }
    ret = function(beta, j = NULL, lambda){
      return(pen = elastic.net(beta, alpha, j, lambda))
    }
    der.pen = function(beta, lambda){
      return((1-alpha)* lambda * beta + alpha * lambda * sign(beta))
    }

    return(list(pen = ret, scale.invariant = F, aditive = T, der.pen = der.pen))
  }

  ## Penalizacion SIGN
  if(identical(penalty, "sign")){
    ret = function(beta, lambda){
      return(norm.quotient(beta, lambda))
    }

    der.pen = function(beta, lambda){
      n1 = sum(abs(beta))
      n2 = sqrt(sum(beta**2))
      return((lambda / (n2**3)) * (sign(beta) * (n2**2) - n1 * beta))
    }
    return(list(pen = ret, scale.invariant = T, aditive = F, der.pen = der.pen))
  }

  ## Penalizacion SCAD
  if(identical(penalty, "scad")){
    if(is.null(alpha)){
      alpha = 3.7
    }
    ret = function(beta, j = NULL, lambda){
      return(scad(beta, alpha, j, lambda))
    }

    der.pen = function(beta, lambda){
      p = length(beta)
      der = rep(NA,p)
      for(j in 1:p){
        betaj = beta[j]
        if(abs(betaj) <= lambda){
          der[j] = lambda * sign(betaj)
        } else if(abs(betaj) <= alpha * lambda){
          der[j] = (alpha * lambda * sign(betaj) - betaj) / (alpha - 1)
        } else{
          der[j] = 0
        }
      }
      return(der)
    }

    return(list(pen = ret, scale.invariant = F, aditive = T, der.pen = der.pen))
  }

  ## Penalizacion MCP
  if(identical(penalty, "mcp")){
    if(is.null(alpha)){
      alpha = 2.7
    }
    ret = function(beta, j = NULL, lambda){
      return(scad(beta, alpha, j, lambda))
    }

    der.pen = function(beta, lambda){
      p = length(beta)
      der = rep(NA,p)
      for(j in 1:p){
        betaj = beta[j]
        if(abs(betaj) <= alpha * lambda){
          der[j] = lambda * sign(betaj) - betaj / alpha
        } else{
          der[j] = 0
        }
      }
      return(der)
    }
    return(list(pen = ret, scale.invariant = F, aditive = T, der.pen = der.pen ))
  }

}

#######
## Losses
#######
get.loss.and.der.loss = function(loss){
  if(identical(loss,"deviance")){
    return(list(loss.fun = eval_loss_function_dev_cpp, der.loss.fun = eval_der_loss_function_dev_cpp))
  }
  if(identical(loss,"ls")){
    return(list(loss.fun = eval_loss_function_lse_cpp, der.loss.fun = eval_der_loss_function_lse_cpp))
  }
  if(identical(loss,"divergence")){
    return(list(loss.fun = eval_loss_function_div_cpp, der.loss.fun = eval_der_loss_function_div_cpp))
  }
  if(identical(loss,"ch")){
    return(list(loss.fun = eval_loss_function_ch_cpp, der.loss.fun = eval_der_loss_function_ch_cpp))
  }
}

#######
## Util
#######
## Function to obtain the design matrix X. Outliers can be generated with one fixed value for X and Y
## or randomly with specified mean and covariance parameters.
##
## Update 3/4/17 : Outilers are added and not replaced in data
##
## n total sample size
## p dimension size
## mu distribution mean
## sigma distribution covariance matrix
## nonzero.beta beta value for nonzero components (the remaining components are assumed to be zero)
## eps outlier proportion
## intercept value for the intercept
## out.X X value for all outliers
## out.Y Y value for all outliers
## out.mu outiler distribution mean
## out.sigma outlier distribution covariance matrix
get.data.norm = function(n,p,mu = NULL,sigma = NULL,true.beta = NULL,eps = 0,
                         intercept = NULL, out.X = NULL, out.Y = NULL,
                         out.mu = NULL, out.sigma = NULL){
  if(is.null(mu)){
    mu = rep(0,p)
  }
  if(is.null(sigma)){
    sigma = diag(p)
  }
  if(is.null(true.beta)){
    true.beta = rep(0,p)
    true.beta[1:5] = rep(1,5)
  }

  out.numb = floor(eps * n)
  X = rmvnorm(n = n, mean = mu, sigma = sigma)
  if(!is.null(intercept)){
    X = cbind(rep(1,n),X)
    true.beta = c(intercept, true.beta)
  }
  X.lc = X %*% true.beta
  probs = inv.logit(X.lc)
  Y = rbinom(n = n, size = 1, prob = probs)

  X.out = NULL
  if(!is.null(out.X)){
    X.out = matrix(data = rep(out.X,out.numb), nrow = out.numb, ncol = p, byrow = T)
    if(!is.null(intercept)){
      X.out = cbind(rep(1,out.numb),X.out)
    }
    X = rbind(X, X.out)
    Y = c(Y, rep(out.Y, out.numb))

  }

  if(!is.null(out.mu) || !is.null(out.sigma)){
    if(is.null(out.mu)){
      out.mu = rep(0,p)
    }
    if(is.null(out.sigma)){
      out.sigma = diag(p)
    }
    X.out = rmvnorm(out.numb, out.mu, out.sigma)
    if(!is.null(intercept)){
      X.out = cbind(rep(1,out.numb),X.out)
    }
    X = rbind(X, X.out)
    Y.out = apply(X.out,1,get.out.Y.for.X, beta = true.beta)
    Y = c(Y, Y.out)
  }

  if(!is.null(X.out)){
    X.out.lc = X.out %*% true.beta
    probs = c(probs, inv.logit(X.out.lc))
  }

  if(!is.null(intercept)){
    X = X[,-1]
  }
  return(list(X = X, Y = Y, probs = probs))
}

get.out.Y.for.X = function(x,beta){
  lc = sum(x*beta)
  return(1*(inv.logit(lc) < 0.5))
}

# Implementation of the function f(s) = (1+exp(-s))^(-1)
inv.logit = function(u){
  return(ifelse(test = (u > 16),yes = 1,no = ifelse(test = (u < -16), yes = 0, no = (1+exp(-u))**(-1))))
}

# Predicts probabilities for given X, beta and intercept
predict.probs = function(beta,X, beta0 = 0){
  scores = X %*% beta + beta0
  return(inv.logit(scores))
}

separate.folds = function(total, nfolds = 10){
  shuffled = sample(1:total,size = total, replace = F)
  return(split(shuffled,cut(1:total,breaks = nfolds)))
}




