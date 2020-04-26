library(dplyr)
library(caret)
library(mlbench)

source("examples/soucre_caret.R")
fitControl = trainControl(method = "cv", number = 3, verboseIter = TRUE)

#### 1.0 Regression Example ####

print("--------------------------------------------------------------")
print("EXAMPLE 1: Regression problem with Boston Housing Dataset")
print("--------------------------------------------------------------")
cat("\n\n")

data("BostonHousing")

loData = BostonHousing

library(caret)
set.seed(998)
inTraining <- createDataPartition(loData$medv, p = .75, list = FALSE)
training <- loData[ inTraining,]
testing  <- loData[-inTraining,]

set.seed(825)

print("Training using glm_builder and caret")
mdl_builder = train(medv ~ ., data = training,
                    method = glModstepR,
                    trControl = fitControl)

print("Hyperparameter Tuning Results:")
print(mdl_builder$results)

print("Best Hyperparameters:")
print(mdl_builder$bestTune)

test_pred = mdl_builder$finalModel$predict(testing)
print(paste("Test RMSE: ", round(Metrics::rmse(test_pred, testing$medv),2), sep = ""))

cat("\n\n\n")


#### 2.0 Classification Exampe ####

#### 2.1 : Titanic Dataset ####
# Basic example for use with Caret

print("--------------------------------------------------------------")
print("EXAMPLE 2A: Basic Classification problem with Titanic Dataset")
print("--------------------------------------------------------------")
cat("\n\n")

library(titanic)
library(magrittr)

data = titanic::titanic_train %>%
  dplyr::select(dplyr::one_of("Survived","Age","SibSp","Parch",
                              "Fare","Pclass","Sex","Embarked")) %>%
  na.omit()

data$Survived = as.factor(data$Survived)

# 2.1.1 Using plain modeling (without caret) ####
print("Training using glm_builder without caret")
mod = model_glm_builder$new(arFormula = Survived~., arData = data, arType = 'class')
pred = mod$predict(data)
acc = Metrics::accuracy(data$Survived, pred)
print(paste("Train Accuracy: ", round(acc*100,2), "%", sep = ""))
cat("\n\n")

# 2.1.2 Using caret ####
print("Training using glm_builder and caret")
source("examples/soucre_caret.R")
mdl_builder = train(Survived ~ ., data = data,
                    method = glModstepR,
                    trControl = fitControl, arType = 'class')

print("Hyperparameter Tuning Results:")
print(mdl_builder$results)

print("Best Hyperparameters:")
print(mdl_builder$bestTune)

pred = mdl_builder$finalModel$predict(data)
acc = Metrics::accuracy(data$Survived, pred)
print(paste("Train Accuracy: ", round(acc*100,2), "%", sep = ""))

cat("\n\n")

#### 2.2 : Breast Cancer Dataset ####
# Advanced example with Train/Test split showing how to avoid overfitting

print("-------------------------------------------------------------------------")
print("EXAMPLE 2B: Advanced Classification problem with Breast Cancer Dataset")
print("-------------------------------------------------------------------------")
cat("\n\n")

data("BreastCancer")

# Ordered factors cause issued with formula. Converting to normal factor
BreastCancer$Cl.thickness = factor(BreastCancer$Cl.thickness, ordered = FALSE)
BreastCancer$Cell.size = factor(BreastCancer$Cell.size, ordered = FALSE)
BreastCancer$Cell.shape = factor(BreastCancer$Cell.shape, ordered = FALSE)
BreastCancer$Marg.adhesion = factor(BreastCancer$Marg.adhesion, ordered = FALSE)
BreastCancer$Epith.c.size = factor(BreastCancer$Epith.c.size, ordered = FALSE)

BreastCancer$Class = as.factor(as.numeric(BreastCancer$Class) - 1)  # Reseting to 0 and 1

loData = BreastCancer %>%
  dplyr::select(-Id) %>%
  na.omit()

set.seed(998)
inTraining <- createDataPartition(loData$Class, p = .75, list = FALSE)
training <- loData[ inTraining,]
testing  <- loData[-inTraining,]

set.seed(825)

# 2.2.0 Using plain glm model ####
print("Training using stats::glm model")
loFormula = Class ~ .
#loFormula = as.formula("Class ~ (.)^2")  # Does not work, maybe because not enough data points and too many predictors
base_model = glm(formula = loFormula, family = "binomial", data = training, control = list(maxit = 5000))

# Train Accuracy
pred = predict(base_model, training, type = "response")
acc = Metrics::accuracy(training$Class, pred)
print(paste("Train Accuracy: ", round(acc*100,2), "%", sep = ""))

# Test Accuracy
pred = predict(base_model, testing, type = "response")
acc = Metrics::accuracy(testing$Class, pred)
print(paste("Test Accuracy: ", round(acc*100,2), "%", sep = ""))
print("Not a very accurate model...")

cat("\n\n")

# 2.2.1 Using plain modeling (without caret) ####
print("Training using glm_builder without caret")
mod = model_glm_builder$new(arFormula = Class ~ ., arData = training, arType = 'class',
                            arStartOrder = 1, arFilterThresh = 0.5)

# Train Accuracy
pred = mod$predict(training)
acc = Metrics::accuracy(training$Class, pred)
print(paste("Train Accuracy: ", round(acc*100,2), "%", sep = ""))

# Test Accuracy
pred = mod$predict(testing)
acc = Metrics::accuracy(testing$Class, as.numeric(pred))
print(paste("Test Accuracy: ", round(acc*100,2), "%", sep = ""))
print("Model is clearly overfitting (test accuracy << train accuracy)")
cat("\n\n")

# 2.2.2 Using caret ####
print("Training using glm_builder with caret")
# Resetting Grid to only take start order of 1 (else glm model fails)
modsteprGrid <- function(x, y, len = NULL, search = "grid") {
  rvGrid = data.frame(arStartOrder = c(rep(1,4)),
                      arLoops = rep(c(1,2,3,4)))
  return(rvGrid)
}

glModstepR$grid <- modsteprGrid

# Building Model with Caret
mdl_builder = train(Class ~ ., data = training,
                    method = glModstepR,
                    trControl = fitControl, arType = 'class')

print("Hyperparameter Tuning Results:")
print(mdl_builder$results)

print("Best Hyperparameters:")
print(mdl_builder$bestTune)

# Train Accuracy
pred = mdl_builder$finalModel$predict(training)
acc = Metrics::accuracy(training$Class, pred)
print(paste("Train Accuracy: ", round(acc*100,2), "%", sep = ""))

# Test Accuracy
pred = mdl_builder$finalModel$predict(testing)
acc = Metrics::accuracy(testing$Class, as.numeric(pred))
print(paste("Test Accuracy: ", round(acc*100,2), "%", sep = ""))
print("Model is not overfitting anymore (test accuracy =~ train accuracy)")
cat("\n\n")
