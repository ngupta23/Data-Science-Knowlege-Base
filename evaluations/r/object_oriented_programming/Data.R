# https://adv-r.hadley.nz/r6.html#r6-classes

library(R6)

Data = R6Class(
  classname = "Data",
  cloneable = TRUE,
  lock_objects=F,
  lock_class=F,

  #### Public Methods ----
  public=list(
    data=NA,

    #### Constructor ----
    initialize = function(data = NA)
    {
      # Add checks here
      if (all(is.na(data))){ stop("Data has not been set") }
      self$set_data(data = data)
    },

    #### Getters and Setters ----
    get_data = function(){return(self$data)},
    set_data = function(data){self$data = data}

    #### General Public Methods ----


  ),

  #### Private Methods ----
  private = list(


  )

)



dataObj = Data$new(data = "placeholder")
dataObj$data




