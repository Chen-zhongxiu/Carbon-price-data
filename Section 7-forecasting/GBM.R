#-----------------------1. China carbon data--------------------------
# Import data
library(dplyr)
library(tidyverse)
library(readxl)
data1 = read.csv("GCP.csv")
X= data1$log_price
price=data1$price

#==========================EU data=====================================
library(dplyr)
library(tidyverse)
library(readxl)
library(dplyr)
library(tidyverse)
library(readxl)
data1 = read.csv("EUA.csv")[-(1:662),]
X= data1$log_price
price=data1$price

#====================================================================
library(yuima)
system.time({
  # Define GBM in log-price form (log Brownian motion)
  gbm_log <- setModel(
    drift = "mu_tilde", 
    diffusion = "sigma", 
    state.var = "x", 
    solve.variable = "x"
  )
  
  # Here assume prices is a price vector, e.g., stock closing prices
  log_prices <- log(data1$price)
  # Note: delta is the time step, e.g., daily = 1/252
  data <- setData(log_prices, delta = 1/252)
  
  yuima_obj <- setYuima(model = gbm_log, data = data)
  
  # 3. Fit (maximum likelihood estimation, MLE)
  fit <- qmle(yuima_obj, start = list(mu_tilde = 0.05, sigma = 0.2))})

# 4. View results
summary(fit)

coef(fit)  # Extract parameters
mu_hat <- coef(fit)["mu_tilde"]
sigma_hat <- coef(fit)["sigma"]

# 4. In-sample one-step prediction (point-by-point)
n <- length(log_prices)
preds <- numeric(n)
lower <- numeric(n)
upper <- numeric(n)

z <-  qnorm(1 - 0.05/2)
dt = 1/252

preds[1] <- log_prices[1]   
lower[1] <- log_prices[1] 
upper[1] <- log_prices[1] 

for (t in 2:n) {
  S_t <- log_prices[t-1]
  # Point forecast = conditional expectation
  preds[t] <-  exp(S_t + mu_hat * dt + 0.5 * sigma_hat^2 * dt)
  
  # Interval forecast
  lower[t] <-  exp((S_t - 0.5 * sigma_hat^2) * dt - z * sigma_hat * sqrt(dt))
  upper[t] <-  exp((S_t - 0.5 * sigma_hat^2) * dt + z * sigma_hat * sqrt(dt))
}

# 5. Error evaluation (RMSE)
actual <- data1$price
rmse <- sqrt(mean((preds - actual)^2))
MdAPE = median(abs(preds - actual)/actual)
MdAPE
MSE =  mean((preds - actual)^2)
MSE
RMSE = sqrt(MSE)
RMSE
MAE = mean(abs(preds - actual))
MAE
RAE = sum(abs(preds - actual))/sum(abs(preds - mean(actual)))
RAE
error = na.omit((exp(preds)-exp(actual))^2)
RMSE = sqrt(MSE)
RMSE
pred = cbind(rn_date,actual,preds)
#write.csv(pred,"GBM_pre_EUA.csv")


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
sigma_hat <- coef(fit)["sigma"]

# t-distribution critical values for residual confidence intervals (approximately 1.96, 1.44, 1.15)
z_95 <- qnorm(0.975)
z_85 <- qnorm(0.925)
z_75 <- qnorm(0.875)

n <- length(log_prices)
L95 <- U95 <- L85 <- U85 <- L75 <- U75 <- numeric(n)

# Compute lower and upper bounds for different α
for (t in 2:n) {
  log_S_last <- log_prices[t-1]  # Previous-period log price
  
  L95[t] <- exp(log_S_last + (mu_hat - 0.5 * sigma_hat^2)*dt - z_95 * sigma_hat * sqrt(dt))
  U95[t] <- exp(log_S_last + (mu_hat - 0.5 * sigma_hat^2)*dt + z_95 * sigma_hat * sqrt(dt))
  
  L85[t] <- exp(log_S_last + (mu_hat - 0.5 * sigma_hat^2)*dt - z_85 * sigma_hat * sqrt(dt))
  U85[t] <- exp(log_S_last + (mu_hat - 0.5 * sigma_hat^2)*dt + z_85 * sigma_hat * sqrt(dt))
  
  L75[t] <- exp(log_S_last + (mu_hat - 0.5 * sigma_hat^2)*dt - z_75 * sigma_hat * sqrt(dt))
  U75[t] <- exp(log_S_last + (mu_hat - 0.5 * sigma_hat^2)*dt + z_75 * sigma_hat * sqrt(dt))
}

metrics_95 <- interval_metrics(price[-1], L95[-1], U95[-1])
print(paste("Alpha=0.95 -> PICP=", round(metrics_95["PICP"],4),
            "PINAW=", round(metrics_95["PINAW"],4),
            "AWD=", round(metrics_95["AWD"],4)))

#----------------- 2. α = 0.85 ----------------------
metrics_85 <- interval_metrics(price[-1], L85[-1], U85[-1])
print(paste("Alpha=0.85 -> PICP=", round(metrics_85["PICP"],4),
            "PINAW=", round(metrics_85["PINAW"],4),
            "AWD=", round(metrics_85["AWD"],4)))

#----------------- 3. α = 0.75 ----------------------
metrics_75 <- interval_metrics(price[-1], L75[-1], U75[-1])
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

# =====================Three-step out-of-sample forecasting (price space)=================
# Here assume prices is a price vector, e.g., stock closing prices
log_prices <- log(data1$收盘价)[1:(nrow(data1)-3)]
# Note: delta is the time step, e.g., daily = 1/252
data <- setData(log_prices, delta = 1/252)

time_fit <- system.time({yuima_obj <- setYuima(model = gbm_log, data = data)

# 3. Fit (maximum likelihood estimation, MLE)
fit <- qmle(yuima_obj, start = list(mu_tilde = 0.05, sigma = 0.2))})

# 4. View results
summary(fit)

coef(fit)  # Extract parameters
mu_hat <- coef(fit)["mu_tilde"]
sigma_hat <- coef(fit)["sigma"]

# Set training and test samples
train_prices <- data1$price[1:(nrow(data1)-3)]
test_prices  <- data1$price[(nrow(data1)-2):nrow(data1)]
log_train <- log(train_prices)
dt <- 1/252
z <- qnorm(0.975)  # z value for 95% confidence interval

# Three-step forecasting based on the last training sample
last_log <- tail(log_train, 1)
last_price <- tail(train_prices, 1)

# Initialize
pred_price <- numeric(3)
lower_price <- numeric(3)
upper_price <- numeric(3)

# Step-by-step recursive forecasting
for (h in 1:3) {
  # Point forecast (with unbiased correction term)
  pred_price[h] <- last_price * exp(mu_hat * dt + 0.5 * sigma_hat^2 * dt)
  
  # Confidence interval (do not add 0.5σ² correction because this is in quantile form)
  lower_price[h] <- last_price * exp((mu_hat - 0.5 * sigma_hat^2) * dt - z * sigma_hat * sqrt(dt))
  upper_price[h] <- last_price * exp((mu_hat - 0.5 * sigma_hat^2) * dt + z * sigma_hat * sqrt(dt))
  
  # Update for the next step
  last_price <- pred_price[h]
}

# True out-of-sample prices
true_price <- test_prices

# ------------------Error metrics------------------
MSE  <- mean((pred_price - true_price)^2)
RMSE <- sqrt(MSE)
MdAPE <- median(abs(pred_price - true_price)/true_price)
MAE  <- mean(abs(pred_price - true_price))
RAE  <- sum(abs(pred_price - true_price)) / sum(abs(true_price - mean(train_prices)))

# Output results
result_out <- data.frame(
  Step = 1:3,
  True = round(true_price, 4),
  Pred = round(pred_price, 4),
  Lower = round(lower_price, 4),
  Upper = round(upper_price, 4)
)

cat("\n===== GBM three-step out-of-sample forecast results (price space) =====\n")
print(result_out)

cat("\n===== Out-of-sample error metrics =====\n")
cat("MSE =", round(MSE,6),
    " RMSE =", round(RMSE,6),
    " MdAPE =", round(MdAPE,6),
    " MAE =", round(MAE,6),
    " RAE =", round(RAE,6), "\n")
