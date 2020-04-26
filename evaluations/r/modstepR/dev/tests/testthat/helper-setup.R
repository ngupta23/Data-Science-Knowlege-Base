library(titanic)

# Data for testing
data_with_cat = titanic::titanic_train %>%
  dplyr::select(dplyr::one_of("Survived", "Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Embarked")) %>%
  na.omit()

data_wo_cat = data_with_cat %>%
  dplyr::select(-dplyr::one_of("Sex", "Embarked"))


# Metric Function
RMSE = function(m, o){ sqrt(mean((m - o)^2)) }   # Define Metric
