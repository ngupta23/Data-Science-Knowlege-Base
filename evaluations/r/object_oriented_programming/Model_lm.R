# https://adv-r.hadley.nz/r6.html#r6-classes

library(R6)

Model_lm = R6Class(
  classname = "Model_lm",
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

    #### Training and Predicting ----
    train = function(formula, data){
      self$final_model = lm(formula = formula, data = data)
    },

    predict = function(new_data){
      stats::predict(self$final_model, new_data = new_data)
    }
  ),

  #### Private Methods ----
  private = list(


  )

)



modelObj = Model_lm$new()
modelObj

