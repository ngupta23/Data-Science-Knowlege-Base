
# modstepR

<!-- badges: start -->
<!-- badges: end -->

The goal of modstepR is to provide a library to model highly non-linear high dimensaional data in an computationally and time efficinet manner using a proprietary modified stepwise variable selection algorithm.

## Installation

You can install the released version of modstepR from [BitBucket](https://bitbucket.itg.ti.com/projects/MOD/repos/modstepr/browse/release). There are 2 ways to install the package

### From **RStudio GUI** 

1. Go to the "Packages" tab and click "Install"
2. Change "Install from:" to "Package Archive File" and point to the tar.gz package file
3. Click "Install"

### From **Command Line** or from a **R Script**

1. Install RTools (if not already installed) from [here](https://cran.r-project.org/bin/windows/Rtools/)
2. Type the following (replacing the file name appropriately)

```r
install.packages("<package_file_name_here>.tar.gz", repos=NULL, type="source")
```


## Documentation
Once the library is installed, you can check out the documentation in RStudio using the following commands.

```r
library(modstepR)
?model_glm_builder
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(modstepR)

RMSE = function(m, o){ sqrt(mean((m - o)^2)) }   # Define Metric

mod_base = lm(mpg~., data = mtcars)     # Base Linear Regression Model
pred_base = predict(mod_base, mtcars)   # Predictions with base learner
print(RMSE(pred_base,mtcars$mpg))                

mod = model_glm_builder$new(arFormula = mpg~., arData = mtcars,
                            arStartOrder = 2, arLoops = 3)  # LM Builder Model
print(mod$get_formula())                                    # Formula built from base features
pred = mod$predict(mtcars)                                  # Predictions with LM Builder model
print(RMSE(pred,mtcars$mpg))                                

# Passing 'other' arguments to linear model call (e.g. weights)
mod = model_glm_builder$new(arFormula = mpg~., arData = mtcars,
                            arStartOrder = 2, arLoops = 3, weights = mtcars$mpg)

# Logistic Regression problem (including categorical predictors)
library(titanic)
library(magrittr)

data = titanic::titanic_train %>%
  dplyr::select(dplyr::one_of("Survived","Age","SibSp","Parch",
                              "Fare","Pclass","Sex","Embarked")) %>%
  na.omit()

mod = model_glm_builder$new(arFormula = Survived~., arData = data, arType = 'class',
                            arStartOrder = 2, arFilterThresh = 0.5)
pred = mod$predict(data)
acc = Metrics::accuracy(data$Survived, pred)
print(acc)

summary(mod$get_model())
```

**For more detailed examples of how to use this model with the `caret` package for resample management (train/test splits, Cross-validation plan, etc.), please refer to the examples directory.** 
