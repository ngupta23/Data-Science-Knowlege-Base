#' R6 Data Class (base)
#'
#' @export
#' @examples
#' loDataObj = data$new(arFormula=formula(mpg~.), arData=mtcars)
data = R6::R6Class(
  classname = "data",
  lock_objects=F,
  lock_class=F,

  #### PUBLIC METHODS ####

  public=list(
    #' @description
    #' Create a new `data` object.
    #' This class is intended to be used by developers of this library only.
    #' Used for managing the internal model development process only.
    #' There should be no need for the users of this library to use this class directly.
    #' @param arFormula
    #' Formula to use. Currently only supports
    #' (1) something like DV ~ .
    #' (2) one DV only
    #' @param arData Dataframe containing the Independent Variables (IV) the Dependent Variable (DV)
    #' @return A new `data` object.
    initialize = function(arFormula=NA, arData=NA)
    {
      if(all(is.na(arData)))
      {
        stop("Class: 'data' >> No data has been passed to the data class. Execution will not continue.")
      }
      if(class(arFormula) != 'formula')
      {
        stop("class 'data' >> arFormula is not of class 'formula'. Execution will not continue.")
      }

      # Set the results name
      private$set_result_names(all.vars(arFormula)[1])

      # Convert to model matrix (for categorical variables)
      loData = private$model_matrixify(arFormula, arData)

      # Set the Matrixified Data
      private$set_data(loData)

      private$set_param_names(colnames(loData)[!(colnames(loData) %in% private$get_result_names())])

    },

    # Following getters need to be public since access from model class

    #' @description Gets the underlying data frame
    #' @return Returns the underlying dataframe (in expanded model format for categorical variables)
    get_data              = function(){return(private$obData)},

    #' @description Gets the independent variable names
    #' @return Returns the column names of all the parameters (independent variables) in the model
    #' (in expanded model format for categorical variables)
    get_param_names       = function(){return(private$obParamNames)},

    #' @description Gets the independent variable values
    #' @return Returns the parameter values (independent variables) in the model
    #' (in expanded model format for categorical variables)
    get_param_data        = function(){return(private$obData[,self$get_param_names()])},

    #' @description Gets a particular column from the underlying data
    #' @param ar Column name to fetch
    #' @return Returns a certain column from the underlying data
    get_data_vector       = function(ar){
      stopifnot(ar %in% names(private$obData))
      return(private$obData[[ar]])
    },

    #' @description Checks if a string matches the independent variable name
    #' @param ar Name to check against the underlying target variable name
    #' @return Returns true if the passed argument is the name of the target variable
    is_result_name        = function(ar){return(ar %in% private$obResultNames)}

  ),

  #### PRIVATE METHODS ####

  private = list(
    obData=NA,        #data frame from the combined csv files
    obResultNames=c(),#the result names form the csv
    obParamNames=c(), #the full set of parameter names
    obInteractionNames=c(), #the interaction names
    obDropParams=c(), #additional param names that should not be used in modeling
    obOrders=list(),

    set_data              = function(ar){private$obData=ar},
    set_data_vector       = function(arName,arVec){private$obData[[arName]]=arVec},
    set_result_names      = function(ar){private$obResultNames=ar},
    set_param_names       = function(ar){private$obParamNames=ar},
    set_interaction_names = function(ar){private$obInteractionNames=ar},
    set_drop_params       = function(ar){private$obDropParams=ar},
    set_orders            = function(ar){private$obOrders=ar},

    get_data_names        = function(){return(names(private$obData))},
    get_result_names      = function(){return(private$obResultNames)},
    get_interaction_names = function(){return(private$obInteractionNames)},
    get_drop_params       = function(){return(private$obDropParams)},
    get_orders            = function(){return(private$obOrders)},


    is_param_name         = function(ar){return(ar %in% private$obParamNames)},


    drop_columns = function(arNames)
    {
      private$set_data(private$obData[,(!colnames(private$obData) %in% c(arNames))])
      private$remove_result_names(arNames)
      private$remove_param_names(arNames)
    },

    remove_result_names = function(arDrops)
    {
     private$obResultNames =private$obResultNames[which(!private$obResultNames %in% arDrops)]
    },

    remove_param_names = function(arDrops)
    {
     private$obParamNames =private$obParamNames[which(!private$obParamNames %in% arDrops)]
     private$obInteractionNames =private$obInteractionNames[which(!private$obInteractionNames %in% arDrops)]
    },

    add_interaction = function(arName, arVector)
    {
     private$obData[[arName]]=arVector
      if(!arName %in%private$obInteractionNames)
      {
       private$obInteractionNames = c(private$obInteractionNames,arName)
       private$obParamNames       = c(private$obParamNames,arName)
      }
    },

    increase_order = function(arName){
      if(!arName %in% names(private$obData))
      {
        print(paste("tp_data::increase_order():",arName,"not found"))
        return(arName);
      }
      if(!arName %in% names(private$obOrders))
      {
       private$obOrders[[arName]]=1
      }
      loOrder =private$obOrders[[arName]]+1
      loNewName=paste0(arName,"__",loOrder)
      private$add_interaction(loNewName,private$obData[[arName]]^(loOrder))
     private$obOrders[[arName]] = loOrder
      return(loNewName)
    },

    decrease_order = function(arName){
      if(!arName %in% names(private$obData))
      {
        print(paste("tp_data::decrease_order():",arName,"not found"))
        return();
      }

      if(!arName %in% names(private$obOrders))
      {
       private$obOrders[[arName]]=1
      }
      loOrder =private$obOrders[[arName]]
      if(loOrder == 1)
      {
        return()
      }
      loNewName=paste0(arName,"__",loOrder)
      private$drop_columns(c(loNewName))
     private$obOrders[[arName]] = loOrder-1
    },

    model_matrixify = function(arFormula, arData){
      # Converts the original Data Frame into an expanded model matrix.
      # Needed when we have categorical variables and when using in conjunction with simple_model()

      loResultName = private$get_result_names()

      rvData = model.matrix(arFormula, arData) %>%  # removes the result column
        as.data.frame() %>%
        dplyr::select(-dplyr::one_of('(Intercept)')) %>% # remove intercept column
        dplyr::mutate(!!loResultName := arData %>% purrr::pluck(loResultName))

      return(rvData)
    },

    print = function()
    {
      print(paste("num results:    ",length(private$get_result_names())))
      print(paste(c("    results:  ",private$get_result_names()),collapse=" "))
      print(paste("num params:     ",length(private$get_param_names())))
      print(paste("num rows:       ",nrow(private$get_data())))
    }

  )
)
