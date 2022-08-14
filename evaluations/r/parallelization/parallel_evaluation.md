Parallel Evaluation
================
Nikhil Gupta
2020-12-16 05:15:45

``` r
if (!require("pacman")) install.packages("pacman")
```

    ## Loading required package: pacman

``` r
pacman::p_load(doSNOW, parallel, progress, foreach)
```

``` r
iter_vals = c(1:5000)
```

``` r
add_some_num = function(x, num){
  return(x+num)
}
```

``` r
#init parallel process with max number of cores
cl <- makeCluster(detectCores())
doSNOW::registerDoSNOW(cl)
    
#progress bar for the parallel loop
pb <- progress::progress_bar$new(total = length(iter_vals), format='[:bar] :percent :eta')
progress <- function(n) pb$tick()
```

``` r
# start a parellel loop per each year
# 'num': If some argument is not expected to change, it still has to be repeated the same number of times.
# .export: Used to export some function or variable from global env into foreah env

dataRaw=NULL


# Start the clock!
ptm <- proc.time()

dataRaw = foreach(
  val=iter_vals,
  num=rep(2, length(iter_vals)), 
  .combine=rbind,
  .export=c('add_some_num'),
  .options.snow = list(progress=progress))%dopar%
  {
    return(add_some_num(val, num))
  }
```

    ## Warning in e$fun(obj, substitute(ex), parent.frame(), e$data): already exporting
    ## variable(s): add_some_num

``` r
# Stop the clock
print(proc.time() - ptm)
```

    ##    user  system elapsed 
    ##    2.38    0.26    2.71

``` r
stopCluster(cl)
```

``` r
# Start the clock!
ptm <- proc.time()

dataRaw = foreach(
  val=iter_vals,
  num=rep(2, length(iter_vals)), 
  .combine=rbind,
  .export=c('add_some_num'),
  .options.snow = list(progress=progress))%do%
  {
    return(add_some_num(val, num))
  }

# Stop the clock
print(proc.time() - ptm)
```

    ##    user  system elapsed 
    ##     1.3     0.0     1.3
