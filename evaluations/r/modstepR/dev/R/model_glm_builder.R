#' R6 model_glm_builder class
#'
#' @export
#' @examples
#'
#' RMSE = function(m, o){ sqrt(mean((m - o)^2)) }   # Define Metric
#'
#' mod_base = lm(mpg~., data = mtcars)     # Base Linear Regression Model
#' pred_base = predict(mod_base, mtcars)            # Predictions with base learner
#' print(RMSE(pred_base,mtcars$mpg))
#'
#' mod = model_glm_builder$new(arFormula = mpg~., arData = mtcars,
#'                             arStartOrder = 2, arLoops = 3)  # LM Builder Model
#' print(mod$get_formula())                                    # Formula built from base features
#' pred = mod$predict(mtcars)                                  # Predictions with LM Builder model
#' print(RMSE(pred,mtcars$mpg))
#'
#' # Passing 'other' arguments to linear model call (e.g. weights)
#' mod = model_glm_builder$new(arFormula = mpg~., arData = mtcars,
#'                             arStartOrder = 2, arLoops = 3, weights = mtcars$mpg)
#'
#' # Logistic Regression problem (including categorical predictors)
#' library(titanic)
#' library(magrittr)
#'
#' data = titanic::titanic_train %>%
#'   dplyr::select(dplyr::one_of("Survived","Age","SibSp","Parch",
#'                               "Fare","Pclass","Sex","Embarked")) %>%
#'   na.omit()
#'
#' mod = model_glm_builder$new(arFormula = Survived~., arData = data, arType = 'class',
#'                             arStartOrder = 2, arFilterThresh = 0.5)
#' pred = mod$predict(data)
#' acc = Metrics::accuracy(data$Survived, pred)
#' print(acc)
#'
#' summary(mod$get_model())
#'
#' # Example with increased iteration limits (to solve convergence issues)
#' mod = model_glm_builder$new(arFormula = Survived~., arData = data, arType = 'class',
#'                             arStartOrder = 2, arLoops = 6, arFilterThresh = 0.1,
#'                             control = list(maxit = 5000))

model_glm_builder= R6::R6Class(
  classname = "model_glm_builder",
  inherit = model_glm_1,
  lock_objects = FALSE,

  #### PUBLIC METHODS ####

  public = list(
    #' @description
    #' Create a new `model_glm_builder` object.
    #' Used to develop the stepwise linear/logistic regression model
    #' @param arFormula
    #' Formula to use. Currently only supports
    #' (1) something like DV ~ .
    #' (2) one DV only
    #' @param arData Dataframe containing the Independent Variables (IV) the Dependent Variable (DV)
    #' @param arType Specify whether you want to build
    #' (1) a regression model ('reg') or
    #' (2) a classification model ('class')
    #' For classification problems, this currently only supports binary classification problems
    #' Also, the levels must be coded 0 and 1 for now.
    #' @param arStartOrder Model developments starts from this order. For example
    #' 1 for 1st order model,
    #' 2 for second order model contained 2 variable interactions, etc.
    #' @param arFilterThresh p-value to use for keeping important variables in the model  #'
    #' @param arLoops Number of loops to go over the variables
    #' @param arMetricFunc Custom Metric Function to be used for model development. Defaults to
    #' Metrics::rmse() for regression problems, and
    #' Metrics::accuracy() for classification problems
    #' @param arMetricMaximize If a custom metric function is specified, should it be maximized?
    #' @param arVerbose Progress Reporting
    #' 0 = No Print
    #' 1 = Minimal printing
    #' 2 = Detailed printing
    #' 3 = Debug Mode only
    #' @param ... Any additional argument that needs to be passed to the underlying glm model call
    #' Examples can include 'weights', 'control', etc.
    #' @return A new `model_glm_builder` object.
    initialize = function(arFormula=NA, arData=NA, arType='reg',
                          arStartOrder=1, arFilterThresh=0.05, arLoops=3,
                          arMetricFunc=NA, arMetricMaximize=NA, arVerbose=0, ...)
    {
      if (private$get_verbose() >= 3){
        print ("glm_builder.initialize >> Initialize parent started")
      }
      super$initialize(arFormula=arFormula, arData=arData, arType=arType, arVerbose=arVerbose)
      if (private$get_verbose() >= 3){
        print ("glm_builder.initialize >> Initialize parent completed")
      }

      private$set_start_order(arStartOrder = arStartOrder)
      private$set_filter_thresh(arFilterThresh = arFilterThresh)
      private$set_loops(arLoops = arLoops)
      private$set_metric_func(arMetricFunc)
      private$set_metric_maximize(arMetricMaximize)
      if (private$get_verbose() >= 3){
        print ("glm_builder.initialize >> Initialize specific vars completed")
        # Print Summary
        self$print()
      }

      if (private$get_verbose() >= 1){
        print ("glm_builder.initialize >> Training started ...")
      }

      private$train(private$get_data_obj(), ...)

      if (private$get_verbose() >= 1){
        print ("glm_builder.initialize >> Training completed")
      }

    }
  ),

  #### PRIVATE METHODS ####

  private = list(
    obStartOrder=NA,
    obFilterThresh=NA,
    obLoops=NA,
    set_start_order = function(arStartOrder){private$obStartOrder = arStartOrder},
    set_filter_thresh = function(arFilterThresh){private$obFilterThresh = arFilterThresh},
    set_loops = function(arLoops){private$obLoops = arLoops},

    get_start_order = function(){return (private$obStartOrder)},
    get_filter_thresh = function(){return (private$obFilterThresh)},
    get_loops = function(){return (private$obLoops)},

    train = function(arDataObj, ...)
    {
      if (private$get_verbose() >= 3){
        print ("glm_builder.train >> Entered train function")
      }
      #rvBestModel = self$obModel
      rvBestModel = self$get_model()
      loData = private$get_data_frame(arDataObj)

      if (private$get_verbose() >= 3){
        print ("glm_builder.train >> Start setting Initial Formula ...")
      }

      if(all(is.na(rvBestModel)))
      {
        if (private$get_start_order() > 1){
          loNewFormula=as.formula(paste(private$get_result_name(),"~(.)^", private$get_start_order(), sep=""), env=environment())
        }
        else{
          loNewFormula=as.formula(paste(private$get_result_name(),"~.", sep=""), env=environment())
        }

        if (private$get_verbose() >= 3){
          print ("glm_builder.train >> Start building initial model.")
        }
        rvBestModel=glm(formula=loNewFormula, family = private$get_family(), data=loData, ...)
        private$set_model(rvBestModel)

        if (private$get_verbose() >= 3){
          print ("glm_builder.train >> Initial Model Built.")
        }
      }

      if (private$get_metric_maximize() == FALSE){
        loBestMetric = 1e23  # Large value so that it can be overwritten by first model created
      }
      else if (private$get_metric_maximize() == TRUE){
        loBestMetric = -1e23 # Small value so that it can be overwritten by first model created
      }

      loTriedNames=c()

      for(i in 1:private$get_loops())
      {
        if (private$get_verbose() >= 1){
          print (paste("Loop: ", i, sep = ""))
        }

        for(lpName in setdiff(arDataObj$get_param_names(),self$obDropParams))
        {
          if (private$get_verbose() >= 2){
            print (paste("    Adding Interactions for Variable: ", lpName, sep = ""))
          }
          loFormula = private$simple_formula(rvBestModel, 1) #.1
          loNewFormula = as.formula(paste(private$get_result_name(),"~",lpName,"*(",loFormula,")",sep=""), env=environment())

          loModel = private$simplify_model(loData,
                                           glm(formula=loNewFormula, family = private$get_family(), data=loData, ...),
                                           private$get_filter_thresh(), ...) #.05
          private$set_model(loModel)

          loMetric = private$get_model_metric(arDataObj, F)

          if (private$get_metric_maximize() == FALSE){
            if(loMetric > loBestMetric)
            {
              private$set_model(rvBestModel)
            }
            else
            {
              rvBestModel=self$get_model()
              loBestMetric=loMetric
            }
          }
          else if (private$get_metric_maximize() == TRUE){
            if(loMetric < loBestMetric)
            {
              private$set_model(rvBestModel)
            }
            else
            {
              rvBestModel=self$get_model()
              loBestMetric=loMetric
            }
          }
        }
      }

      loFormula = private$simple_formula(rvBestModel, 1.0)
      loNewFormula = as.formula(paste(private$get_result_name(),"~",loFormula,sep=""),env=environment())
      private$set_formula(loNewFormula)

    }
  )
)

