#-----------------------1. China carbon data--------------------------
# Import data
library(dplyr)
library(tidyverse)
library(readxl)
library(dplyr)
library(tidyverse)
library(readxl)
data1 = read.csv("GCP.csv")
X= data1$log_price
price=data1$price
#===========================================2. European carbon data============================
library(dplyr)
library(tidyverse)
library(readxl)
data1 = read.csv("EUA.csv")[-(1:662),]
X= data1$log_price
price=data1$price

#--------------1.1 ARIMA-----------------------
library(forecast)
time_fit <- system.time({
  # Assume your return series is a vector rtn (length=100)
  rtn_ts <- ts(X)
  # Automatically select the best ARIMA(p,d,q)
  fit <- auto.arima(rtn_ts)
  # In-sample one-step fitted values
  fitted_rn <- fitted(fit)
  sigma2 <- fit$sigma2 
  
  # Timing ARIMA fitting
  
  fit <- auto.arima(rtn_ts)
})
# Print results
print(fit)             # Output fitted model information
print(fitted_rn)   # Output in-sample fitted values


# Recover fitted prices (recursive method)
price_hat <- numeric(length(fitted_rn))
price_hat[1] <- price[1]   # Use the first cleaned price as the initial value
for (i in 2:(length(fitted_rn))) {
  price_hat[i] <- exp(fitted_rn[i] + 0.5 * sigma2) 
}


# Calculate errors
MdAPE = median(abs(price_hat-price)/price)
MdAPE
MSE =  mean((price_hat-price)^2)
MSE
RMSE = sqrt(MSE)
RMSE
MAE = mean(abs(price_hat-price))
MAE
RAE = sum(abs(price_hat-price))/sum(abs(price_hat-mean(price)))
RAE
error = na.omit((exp(price_hat)-exp(price))^2)
RMSE = sqrt(MSE)
RMSE
pred = cbind(rn_date,price,price_hat)


#=====================Interval estimation diagnostic results================================
# Compute interval diagnostic metrics -------------------------------------------------
interval_metrics <- function(y_true, lower, upper) {
  N <- length(y_true)
  ymax <- max(y_true)
  ymin <- min(y_true)
  
  # PICP
  picp <- mean((y_true >= lower) & (y_true <= upper))
  
  # PINAW
  pinaw <- mean(upper - lower) / (ymax - ymin)
  
  # AWD
  delta <- numeric(N)
  for (i in 1:N) {
    if (y_true[i] < lower[i]) {
      delta[i] <- (lower[i] - y_true[i]) / (upper[i] - lower[i])
    } else if (y_true[i] > upper[i]) {
      delta[i] <- (y_true[i] - upper[i]) / (upper[i] - lower[i])
    } else {
      delta[i] <- 0
    }
  }
  awd <- mean(delta)
  
  return(c(PICP = picp, PINAW = pinaw, AWD = awd))
}

#----------------- 1. α = 0.95 ----------------------
# Residual standard deviation (ARIMA model sigma)
sigma <- sqrt(fit$sigma2)

# Get in-sample fitted values
y_hat <- fitted(fit)

# t-distribution critical values for residual confidence intervals (approximately 1.96, 1.44, 1.15)
z_95 <- qnorm(0.975)
z_85 <- qnorm(0.925)
z_75 <- qnorm(0.875)

# Compute lower and upper bounds for different α
L95 <- exp(y_hat - z_95 * sigma)
U95 <- exp(y_hat + z_95 * sigma)

L85 <- exp(y_hat - z_85 * sigma)
U85 <- exp(y_hat + z_85 * sigma)

L75 <- exp(y_hat - z_75 * sigma)
U75 <- exp(y_hat + z_75 * sigma)


metrics_95 <- interval_metrics(price, L95, U95)
print(paste("Alpha=0.95 -> PICP=", round(metrics_95["PICP"],4),
            "PINAW=", round(metrics_95["PINAW"],4),
            "AWD=", round(metrics_95["AWD"],4)))

#----------------- 2. α = 0.85 ----------------------
metrics_85 <- interval_metrics(price, L85, U85)
print(paste("Alpha=0.85 -> PICP=", round(metrics_85["PICP"],4),
            "PINAW=", round(metrics_85["PINAW"],4),
            "AWD=", round(metrics_85["AWD"],4)))

#----------------- 3. α = 0.75 ----------------------
metrics_75 <- interval_metrics(price, L75, U75)
print(paste("Alpha=0.75 -> PICP=", round(metrics_75["PICP"],4),
            "PINAW=", round(metrics_75["PINAW"],4),
            "AWD=", round(metrics_75["AWD"],4)))

# Organize results for output -------------------------------------------------
results <- data.frame(
  Alpha = c(0.95, 0.85, 0.75),
  PICP = c(metrics_95["PICP"], metrics_85["PICP"], metrics_75["PICP"]),
  PINAW = c(metrics_95["PINAW"], metrics_85["PINAW"], metrics_75["PINAW"]),
  AWD = c(metrics_95["AWD"], metrics_85["AWD"], metrics_75["AWD"])
)

print(results)

# Optional: save results
# write.csv(results, "ARIMA_interval_diagnostics.csv", row.names = FALSE)


# =====================Three-step out-of-sample forecasting (price space)=================

# Set training and test data
data1 = read.csv("EUA.csv")
X= data1$log_price
price=data1$price
train <- X[1:(length(X) - 3)]
test <- X[(length(X) - 2):length(X)]   # Last three points for the test set
train_price <- price[1:(length(price) - 3)]
test_price <- price[(length(price) - 2):length(price)]

# Fit ARIMA model (based on training set)
fit_out <- auto.arima(train)

# Make 1-step to 3-step forecasts
forecast_out <- forecast(fit_out, h = 3)

# Extract predicted return means and confidence intervals
pred_rn <- as.numeric(forecast_out$mean)
sigma2_out <- fit_out$sigma2
lower_rn <- as.numeric(forecast_out$lower[,2])  # 95% confidence lower bound
upper_rn <- as.numeric(forecast_out$upper[,2])  # 95% confidence upper bound

# ------------------Transform return forecasts back to price space------------------

# Based on the last true price in the training set
last_price <- tail(train_price, 1)

# Initialize
pred_price <- numeric(3)
lower_price <- numeric(3)
upper_price <- numeric(3)

# Step-by-step recursive price forecasts (price space)
pred_price[1] <- exp(pred_rn[1] + 0.5 * sigma2_out) 
pred_price[2] <- exp(pred_rn[2] + 0.5 * sigma2_out) 
pred_price[3] <- exp(pred_rn[3] + 0.5 * sigma2_out) 

# Interval lower and upper bounds (recursive form)
lower_price[1] <- exp(lower_rn[1]) 
lower_price[2] <- exp(lower_rn[2])
lower_price[3] <- exp(lower_rn[3]) 

upper_price[1] <- exp(upper_rn[1])
upper_price[2] <- exp(upper_rn[2]) 
upper_price[3] <- exp(upper_rn[3]) 

# ------------------Compute error metrics (out-of-sample)------------------

# Ensure consistent lengths
true_price <- test_price[1:3]

# Error metrics
MSE  <- mean((pred_price - true_price)^2)
RMSE <- sqrt(MSE)
MdAPE <- median(abs(pred_price - true_price) / true_price)
MAE  <- mean(abs(pred_price - true_price))
RAE  <- sum(abs(pred_price - true_price)) / sum(abs(true_price - mean(train_price)))

# Output error results
error_metrics <- data.frame(
  Step = 1:3,
  True = true_price,
  Pred = pred_price,
  Lower = lower_price,
  Upper = upper_price
)
print("===== Multi-step forecast results (price space) =====")
print(round(error_metrics, 4))

print("===== Error metrics =====")
cat("MSE =", round(MSE,4),
    " RMSE =", round(RMSE,4),
    " MdAPE =", round(MdAPE,4),
    " MAE =", round(MAE,4),
    " RAE =", round(RAE,4), "\n")
