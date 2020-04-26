# This file contains the code that needs to be sourced if you want to use the package with caret

glModstepR <- list(type = c("Classification", "Regression"),
                   library = "modstepR",
                   loop = NULL)

prm <- data.frame(parameter = c("arStartOrder", "arLoops"),
                  class = rep("numeric", 2),
                  label = c("startOrder", "loops"))

glModstepR$parameters <- prm

modsteprGrid <- function(x, y, len = NULL, search = "grid") {
  rvGrid = data.frame(arStartOrder = c(rep(1,4), rep(2,4)),
                      arLoops = rep(c(1,2,3,4),2))
  return(rvGrid)
}

glModstepR$grid <- modsteprGrid

modsteprFit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  library(modstepR)

  m <- match.call(expand.dots = TRUE)
  # print("modsteprFit >> match.call")
  # print(m)

  # https://stats.stackexchange.com/questions/89171/help-requested-with-using-custom-model-in-caret-package
  loData = as.data.frame(x)
  loData$.outcome = y
  loData = as.data.frame(loData)

  loFormula = formula(.outcome ~ .)

  mod = model_glm_builder$new(arFormula = loFormula,
                              arData = loData,
                              arStartOrder = param$arStartOrder,
                              arLoops = param$arLoops, ...)

  return(mod)

}

glModstepR$fit <- modsteprFit

modsteprPred <- function(modelFit, newdata, preProc = NULL, submodels = NULL){
  library(modstepR)
  rvPred = modelFit$predict(arNewData = newdata)
  return(rvPred)
}

glModstepR$predict <- modsteprPred


modsteprProb <- function(modelFit, newdata, preProc = NULL, submodels = NULL){
  library(modstepR)
  rvPredProb = modelFit$predict(arNewData = newdata, arProb = TRUE)
  return(rvPredProb)
}

glModstepR$prob <- modsteprProb
