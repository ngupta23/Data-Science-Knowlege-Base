test_that("[Reg] GLM Builder (Basic Test)", {
  # # Baseline
  # mod = glm(mpg~., data = mtcars)
  # pred = predict(mod, mtcars)
  # metric = Metrics::rmse(mtcars$mpg, pred)  ## 2.15

  mod = model_glm_builder$new(arFormula = mpg~., arData = mtcars, arStartOrder = 2, arLoops = 3)
  pred = mod$predict(mtcars)
  expect_equal(round(RMSE(mtcars$mpg, pred),6), 1.407762)
})

test_that("[Reg] GLM Builder with Weights", {
  # # Baseline
  # mod = glm(mpg~., data = mtcars)
  # pred = predict(mod, mtcars)
  # metric = Metrics::rmse(mtcars$mpg, pred)  ## 2.15

  mod = model_glm_builder$new(arFormula = mpg~., arData = mtcars, arStartOrder = 2, arLoops = 3, weights = mtcars$mpg)
  pred = mod$predict(mtcars)
  expect_equal(round(RMSE(mtcars$mpg, pred),7), 0.8183518)
})

test_that("[Reg] GLM Builder with Weights and Custom Metrics", {
  # # Baseline
  # mod = glm(mpg~., data = mtcars)
  # pred = predict(mod, mtcars)
  # metric = Metrics::rmse(mtcars$mpg, pred)  ## 2.15

  mod = model_glm_builder$new(arFormula = mpg~., arData = mtcars,
                              arStartOrder = 2, arLoops = 3, weights = mtcars$mpg)
  pred = mod$predict(mtcars)
  base_metric = RMSE(mtcars$mpg, pred)

  custom_metric = function(actual, predicted){
    base_metric = Metrics::rmse(actual, predicted)
    return(base_metric/2)
  }

  mod2 = model_glm_builder$new(arFormula = mpg~., arData = mtcars,
                               arStartOrder = 2, arLoops = 3, arMetricFunc = custom_metric, weights = mtcars$mpg)
  pred2 = mod2$predict(mtcars)
  metric2 = custom_metric(pred2, mtcars$mpg)

  expect_equal(round(base_metric,6)/2, round(metric2,6))
})

test_that("[Class] GLM Builder (Basic Test with Continuous vars only)", {
  # # Baseline
  # base = glm(Survived~., family = binomial(link='logit'), data = data_wo_cat)
  # pred = predict(base, data_wo_cat, type = 'response')
  # pred = pred > 0.5
  # acc = Metrics::accuracy(data_wo_cat$Survived, pred)
  # expect_equal(round(acc, 7), 0.7016807)

  mod = model_glm_builder$new(arFormula = Survived~., arData = data_wo_cat, arType = 'class',
                               arStartOrder = 2, arLoops = 3, arFilterThresh = 0.4)
  pred = mod$predict(data_wo_cat)
  pred = pred > 0.5
  acc = Metrics::accuracy(data_wo_cat$Survived, pred)
  expect_equal(round(acc, 7), 0.745098)
})

test_that("[Class] GLM Builder (Basic Test with Continuous and Categorical vars)", {
  # # Baseline
  # base = glm(Survived~., family = binomial(link='logit'), data = data_wo_cat)
  # pred = predict(base, data_wo_cat, type = 'response')
  # pred = pred > 0.5
  # acc = Metrics::accuracy(data_wo_cat$Survived, pred)
  # expect_equal(round(acc, 7), 0.7016807)

  mod = model_glm_builder$new(arFormula = Survived~., arData = data_with_cat, arType = 'class')
  pred = mod$predict(data_with_cat)
  pred = pred > 0.5
  acc = Metrics::accuracy(data_with_cat$Survived, pred)
  expect_equal(round(acc, 5), 0.82913)
})

test_that("[Class] GLM Builder (Model with continuous and categorical variabels and With expanded order, loops and relaxed filtering)", {
  # # Baseline
  # base = glm(Survived~., family = binomial(link='logit'), data = data_wo_cat)
  # pred = predict(base, data_wo_cat, type = 'response')
  # pred = pred > 0.5
  # acc = Metrics::accuracy(data_wo_cat$Survived, pred)
  # expect_equal(round(acc, 7), 0.7016807)

  mod = model_glm_builder$new(arFormula = Survived~., arData = data_with_cat, arType = 'class',
                              arStartOrder = 2, arLoops = 6, arFilterThresh = 0.5)
  pred = mod$predict(data_with_cat)
  pred = pred > 0.5
  acc = Metrics::accuracy(data_with_cat$Survived, pred)
  expect_equal(round(acc, 7), 0.8809524)

})
