library(tidyquant)

start = "1970-12-31"
end = "2019-06-30"

# VOO = ^GPSC
# IVOO = ^RUT
# VIOO = ^RUI
# VEU = 
# VWO = 
# VNQ = ^RMZ
# BND = 

returns_m_components  = c("^GSPC", "^RUT", "^RUI", "VEU", "VWO", "^RMZ", "BND") %>%
  tq_get(get = "stock.prices",
         from = start,
         to = end)

returns_m_components

returns_m_components %>%
  group_by(symbol) %>%
  summarize(n=n()) %>%
  arrange(n)

filter_plot = function(arData, arSymbol){
  stock = arData %>%
    filter(symbol == arSymbol)
  
plot(x = stock$date, y = stock$adjusted, main = arSymbol, xlab = "Date", ylab = "Adjusted Price")
}

filter_plot(returns_m_components, "VEU")
filter_plot(returns_m_components, "^RMZ")


# # Functions to check out
# tq_transmute()
# periodReturn()  # quantmod package
# tq_portfolio()
# tq_performance()
