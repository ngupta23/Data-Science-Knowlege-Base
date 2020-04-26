#' R6 model (base) class
#'
#' @export
#' @examples
#' loModelObj = model$new(arFormula=formula(mpg~.), arData=mtcars, arType='reg')
#'
#' @importFrom magrittr "%>%"
model = R6::R6Class(
  classname = "model",
  lock_objects=F,
  lock_class=F,

  #### PUBLIC METHODS ####

  public = list(
    #' @description
    #' Create a new `model` object.
    #' This class is intended to be used by developers of this library only.
    #' Used for managing the internal model development process only.
    #' This is the base model class that is inherited by all other model classes.
    #' There should be no need for the users of this library to use this class directly.
    #' @param arFormula
    #' Formula to use. Currently only supports
    #' (1) something like DV ~ .
    #' (2) one DV only
    #' @param arData Dataframe containing the Independent Variables (IV) the Dependent Variable (DV)
    #' @param arType Specify whether you want to build
    #' (1) a regression model ('reg') or
    #' (2) a classification model ('class')
    #' For classification problems, this currently only supports binary classification problems
    #' @param arVerbose Progress Reporting
    #' 0 = No Print
    #' 1 = Minimal printing
    #' 2 = Detailed printing
    #' 3 = Debug Mode only
    #' @return A new `model` object.
    initialize = function(arFormula=NA, arData=NA, arType='reg', arVerbose=0)
    {
      if (class(arFormula) != 'formula'){
        print(paste("Formula: ", arFormula, sep=""))
        print(paste("Class of Formula: ", class(arFormula), sep=""))
        stop("Class: 'model' >> arFormula must be of class 'formula'. Execution will not continue.")
      }
      if (class(arData) != 'data.frame'){
        print(paste("Class of Data: ", class(arData), sep=""))
        stop("Class: 'model' >> arData must be of class 'data.frame'. Execution will not continue.")
      }

      # Check if classification problem output is a factor
      # If not, convert to factor
      loResultName = all.vars(arFormula)[1]
      if (arType == 'class' & !is.factor(arData[[loResultName]])){
        warning("This is a classification problem but the target variable is not a factor. This will be type cast to a factor")
        arData[[loResultName]] = as.factor(arData[[loResultName]])
      }

      private$set_data_obj(data$new(arFormula=arFormula, arData=arData))
      private$set_result_name(all.vars(arFormula)[1])  # picks 1st element in all.vars. Since we only support 1 IV for now, this works.
      private$set_type(arType)
      private$set_family()
      private$set_verbose(arVerbose)

    },

    #' @description Gets the formula used by the model
    #' @return Returns the formula used in the model
    #' Before training this is of the type formula(DV ~ .)
    #' After training, it included terms from model.matrix which can include
    #' expanded categorcial column names (one hot encoded names)
    get_formula             = function(){return(private$obFormula)},

    #' @description (Abstract method) Returns the predicted values for a set of predictors.
    #' @param arNewData
    #' Data Frame containing the predictors
    #' Data should be passed without expansion through model.matrix (for categorical variables)
    #' Expansion of data frame is managed internall (automatically)
    #' @return Predictions
    predict = function(arNewData)
    {
      stop("Class 'model' >> 'predict' method must be implemented in child class")
    },

    #' @description Returns the final model.
    #' @return Final Model
    get_model               = function(){return(private$obModel)},

    #' @description Prints basic information about the class.
    print = function()
    {
      print(paste("Problem Type = ", private$get_type()))
      loFamily = private$get_family()
      print(paste("Model Family Type = ", loFamily[1]))
      print(paste("Model Family Link = ", loFamily[2]))
      print(paste("Target Variable = ", private$get_result_name()))
      print(paste("Model Formula  = "))
      print(self$get_formula())
      loMetric = private$get_metric_func()
      print(paste("Model Metric Used = "))
      print(loMetric)
      # print(substitute(loMetric))
      # print(substitute(loMetric, environment()))
      # print(quote(loMetric))
      # print(enquote(loMetric))
      print(paste("Maximize Metric for training? ", private$get_metric_maximize()))
      print(paste("Model Metric Value =", private$get_total_metric()))
    }

  ),

  #### PRIVATE METHODS ####

  private = list(
    obType="",
    obFamily=NA,
    obVerbose=0,
    obMetricFunc=NA,
    obMetricMaximize=NA,
    obResultName="",
    obModel=NA,
    obDataObj=NA,
    obInteractions=c(),
    obTotalMetric=1e9,
    obR2=0,
    obFormula="",
    obStats=NA,
    obPath="",
    obDropParams=c(),
    obParamOrders=list(),

    set_type         = function(ar){
      loType = tolower(ar)
      if (loType != 'class' & loType != 'reg'){
        stop("Class 'model' >> Argument 'arType' should be either 'class' for classification model or 'reg' for regression model")
      }
      private$obType = loType
    },
    set_family = function(){
      if (private$get_type() == 'reg'){
        private$obFamily = gaussian(link = 'identity')
      }
      else if (private$get_type() == 'class'){
        private$obFamily = binomial(link = 'logit')
      }
    },
    set_verbose = function(ar){
      private$obVerbose = ar
    },
    set_metric_func  = function(ar){
      if (all(is.na(ar))){
        # If metric function is not defined by user, then use preexisting ones based on model type
        if (private$get_type() == 'reg'){
          private$obMetricFunc = Metrics::rmse
        }
        else if(private$get_type() == 'class'){
          private$obMetricFunc = Metrics::accuracy
        }
        else{
          stop("Class 'model' >> type of model is not correct. Should be either 'class' or 'reg'.")
        }
      }
      else{
        # If metric functon is defined by use, then use that one.
        private$obMetricFunc = ar
      }
    },
    set_metric_maximize = function(ar){
      if (all(is.na(ar))){
        # If metric function is not defined by user, then use preexisting ones based on model type
        if (private$get_type() == 'reg'){
          private$obMetricMaximize = FALSE
        }
        else if(private$get_type() == 'class'){
          private$obMetricMaximize = TRUE
        }
        else{
          stop("Class 'model' >> type of model is not correct. Should be either 'class' or 'reg'.")
        }
      }
      else{
        # If metric functon is defined by use, then use that one.
        private$obMetricMaximize = ar
      }
    },
    set_result_name  = function(ar){private$obResultName=ar},
    set_model        = function(ar){private$obModel=ar},
    set_data_obj     = function(ar){private$obDataObj=ar},
    set_interactions = function(ar){private$obInteractions=ar},
    set_total_metric  = function(ar){private$obTotalMetric=ar},
    set_formula      = function(ar){private$obFormula=ar},
    set_stats        = function(ar){private$obStats=ar},
    set_path         = function(ar){private$obPath=ar},
    set_param_orders = function(ar){private$obParamOrders=ar},


    get_model_metric = function(arDataObj, arPlot=FALSE)
    {
      stopifnot(arDataObj$is_result_name(private$get_result_name()))
      stopifnot(!all(is.na(self$get_model())))

      rvError=1e23

      loData = arDataObj$get_param_data()  # Only parameters
      loPredictionAll=self$predict(loData)
      loPredictionAll[which(is.nan(loPredictionAll))] = NA
      loPredictionAll[which(is.infinite(loPredictionAll))] = NA

      loActualAll=arDataObj$get_data_vector(private$get_result_name())

      # Common to regression and classification problems.
      rvTotalMetric = private$get_metric_func()(actual = loActualAll, predicted = loPredictionAll)

      private$set_total_metric(rvTotalMetric)
      return(private$get_total_metric())
    },

    get_valid_params = function(ar)
    {
      return(setdiff(ar,self$obDropParams))
    },

    get_data_frame = function(arDataObj)
    {
      rvData=arDataObj$get_data()[,c(private$get_result_name(),private$get_valid_params(arDataObj$get_param_names()))]
      return(rvData)
    },
    get_type                = function(){return(private$obType)},
    get_family              = function(){return(private$obFamily)},
    get_verbose             = function(){return(private$obVerbose)},
    get_metric_func         = function(){return(private$obMetricFunc)},
    get_metric_maximize     = function(){return(private$obMetricMaximize)},
    get_result_name         = function(){return(private$obResultName)},
    get_data_obj            = function(){return(private$obDataObj)},
    get_interactions        = function(){return(private$obInteractions)},
    get_total_metric         = function(){return(private$obTotalMetric)},
    get_stats               = function(){return(private$obStats)},
    get_path                = function(){return(private$obPath)},
    get_param_orders        = function(){return(private$obParamOrders)},

    model_matrixify = function(arFormula = private$get_formula(), arData, arTrain=TRUE){
      # Converts the original Data Frame into an expanded model matrix.
      # Needed when we have categorical variables and when using in conjunction with simple_model()

      loResultName = private$get_result_name()

      loData = arData

      # If data only has predictors and not the Dependent Variable (DV), add it so that model.matrix can work
      if (!(loResultName %in% colnames(loData))){
        loDummyPredColumn = data.frame(rnorm(nrow(loData)))
        colnames(loDummyPredColumn) = loResultName
        loData = cbind(loData, loDummyPredColumn)
      }

      rvData = model.matrix(arFormula, loData) %>%  # removes the result column
        as.data.frame() %>%
        dplyr::select(-dplyr::one_of('(Intercept)')) # remove intercept column

      # Add DV if in train mode
      if (arTrain == TRUE){
        rvData %>%
          dplyr::mutate(!!loResultName := loData %>% purrr::pluck(loResultName))
      }

      return(rvData)
    },

    prep_data_for_prediction = function(arNewData){
      loFormula = formula(paste(private$get_result_name(), "~.", sep = ""))
      rvData = private$model_matrixify(loFormula, arNewData, arTrain = FALSE)  # Dont need DV in data for prediction, hence set to FALSE
      return(rvData)
    },

    simplify_model = function(arData, arModel=self$get_model(), arThresh=0.1)
    {
      print("simplify_model function must be implemented in child class")
      stop()
    },

    save_model = function(arPath=NA)
    {
      if(is.character(arPath))
      {
        private$set_path(arPath)
      }
      if(is.character(private$get_path()))
      {
        loEnv = attr(self$obFormula, ".Environment")
        parent.env(loEnv) = .GlobalEnv
        rm(list=ls(envir=loEnv), envir=loEnv) # remove all objects from this environment
        #loEnv = attr(self$obModel$terms, ".Environment")
        loEnv = attr(self$get_model()$terms, ".Environment")
        parent.env(loEnv) = .GlobalEnv
        rm(list=ls(envir=loEnv), envir=loEnv)
        ###above code reduces the save rds size
        rvSave=list()
        for(lpName in names(self))
        {
          if(class(self[[lpName]]) != "function" && lpName != ".__enclos_env__")
          {
            if(lpName=="obModel")
            {
              rvSave[[lpName]]=tp_clean_model(self[[lpName]])
            }
            else
            {
              rvSave[[lpName]]=self[[lpName]]
            }
          }
        }
        saveRDS(rvSave,private$get_path())
      }
      else
      {
        print(paste("Can't save model: ",private$get_result_name()," to:\n  ",private$get_path(),sep=""))
      }
    },

    train = function(arData)
    {
      print("train function must be implemented in child class")
      stop()
    }

  )
)




