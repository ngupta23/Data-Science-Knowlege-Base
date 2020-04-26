# train.formula here: https://github.com/topepo/caret/blob/master/pkg/caret/R/train.default.R

# # Check this snippet to understand "terms" attribute in a model.frame
# mm = model.frame(dist ~ speed, data = cars)
# str(mm)
# attr(mm, "terms")


sum_ng = function(a, b){
  m = match.call()
  print(m$a)
  print(m$b)
  print(m[[1]])  # function name
  print(m[[2]])  # 1st argument
  print(m[[3]])  # 2nd argument
  #print(a+b)
}

sum_ng(1,2)

