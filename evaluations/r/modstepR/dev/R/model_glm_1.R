#' R6 model_glm_1 class
#'
#' @export
#' @examples
#' loModelObj = model_glm_1$new(arFormula=formula(mpg~.), arData=mtcars, arType='reg')
model_glm_1= R6::R6Class(
  classname = "model_glm_1",
  inherit = model,

  #### PUBLIC METHODS ####

  public = list(
    #' @description
    #' Create a new `model_glm_1` object.
    #' This class is intended to be used by developers of this library only.
    #' Used for managing the internal model development process only.
    #' This is the base model class that is inherited by all other glm model classes.
    #' There should be no need for the users of this library to use this class directly.
    #' @param arFormula
    #' Formula to use. Currently only supports
    #' (1) something like DV ~ .
    #' (2) one DV only
    #' @param arData Dataframe containing the Independent Variables (IV) the Dependent Variable (DV)
    #' @param arType Specify whether you want to build
    #' (1) a regression model ('reg') or
    #' (2) a classification model ('class')
    #' For classification problems, this currently only supports binary classification problems.
    #' Also, the levels must be coded 0 and 1 for now.
    #' @param arVerbose Progress Reporting
    #' 0 = No Print
    #' 1 = Minimal printing
    #' 2 = Detailed printing
    #' 3 = Debug Mode only
    #' @return A new `model_glm_1` object.
    initialize = function(arFormula=NA, arData=NA, arType='reg', arVerbose=0)
    {
      super$initialize(arFormula=arFormula, arData=arData, arType= arType, arVerbose=arVerbose)
    },

    #' @description
    #' Returns the predicted values for a set of predictors.
    #' @param arNewData
    #' Data Frame containing the predictors
    #' Data should be passed without expansion through model.matrix (for categorical variables)
    #' Expansion of data frame is managed internall (automatically)
    #' @param arProb
    #' Whether to return predicted probabilities for binary classification problems
    #' Ignored for Regression problems
    #' @return Predictions
    predict = function(arNewData, arProb=FALSE)
    {
      loNewData = private$prep_data_for_prediction(arNewData)  # Matrixifies the data based on original formula

      for(lpName in names(self$obParamOrders))
      {
        if(lpName %in% names(loNewData))
        {
          for(lpI in 1:self$obParamOrders[[lpName]])
          {
            if(lpI>1)
            {
              loNewData[[paste0(lpName,"__",lpI)]]=loNewData[[lpName]]^lpI
            }
          }
        }
      }

      if (private$get_type() == 'reg'){
        rvPrediction = predict(self$get_model(), loNewData)
      }
      else if (private$get_type() == 'class'){
        # TODO: Currently output should be of the form 0 and 1 only
        # Need to fix this later
        rvPrediction = predict(self$get_model(), loNewData, type = 'response')  # Returns the probability
        if (arProb == FALSE){
          # Return the prediction not the probability
          rvPrediction = rvPrediction > 0.5
        }
        rvPrediction = as.numeric(rvPrediction)
      }

      return(rvPrediction)
    }
  ),

  #### PRIVATE METHODS ####

  private = list(
    simplify_model = function(arData, arModel = self$get_model(), arThresh = 0.05, ...)
    {
      loFormula  = as.formula(paste(private$get_result_name(),"~", private$simple_formula(arModel, arThresh)), env=environment())
      rvModel  = glm(loFormula, family = private$get_family(), data=arData, model=F, ...)
      if(summary(rvModel)$df[2] == 0)
      {
        return(arModel)
      }

      return(rvModel)
    },

    simple_formula = function(arModel = self$get_model(), arThresh = 0.05)
    {
      loCoefficients = summary(arModel)$coefficients[,4]  ## For categorical variables, this will give a column for each level (dummified)
      loImportantCo  = private$important_params(loCoefficients, arThresh)
      rvFormula=""
      if(!is.null(loImportantCo) && length(loImportantCo)!=0)
      {
        rvFormula=paste("1",paste(loImportantCo,collapse="+"),sep="+")
      }
      else
      {
        rvFormula="1"
      }

      return(rvFormula)
    },

    important_params = function(arCoefficientPvalues, arThresh=0.05)
    {
      reduced_set_interactions = names(which(arCoefficientPvalues < arThresh))
      reduced_set_interactions = reduced_set_interactions[which(reduced_set_interactions!="(Intercept)")]
      return(reduced_set_interactions)

      important_names=c()
      for(interaction_set in reduced_set_interactions)
      {
        if (!interaction_set %in% important_names)
        {
          important_names=c(important_names,interaction_set)
        }
      }
      return(important_names)
    }
  )
)

