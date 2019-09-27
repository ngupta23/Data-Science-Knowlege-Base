# https://adv-r.hadley.nz/r6.html#r6-classes

library(R6)

Model = R6Class(
  classname = "Model",
  cloneable = TRUE,
  lock_objects=F,
  lock_class=F,

  #### Public Methods ----
  public=list(
    final_model=NA,

    #### Constructor ----
    initialize = function()
    {
      # Add checks here

    },

    #### Getters and Setters ----
    get_final_model = function(){return(self$final_model)},
    set_final_model = function(final_model){self$final_model = final_model},

    #### General Public Methods ----

    train = function(){
      stop("You are calling the train function from the base class. You should implement this in the child class")
    },

    predict = function(new_data){
      stop("You are calling the predict function from the base class. You should implement this in the child class")
    }

  ),

  #### Private Methods ----
  private = list(


  )

)



modelObj = Model$new()
modelObj



