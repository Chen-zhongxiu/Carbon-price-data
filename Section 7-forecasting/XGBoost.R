#-----------------------1. China carbon data--------------------------
# Import data
library(dplyr)
library(tidyverse)
library(readxl)
data1 = read.csv("GCP.csv")
X= data1$log_price
price=data1$price


#=======================EU carbon data==============================
library(dplyr)
library(tidyverse)
library(readxl)
library(dplyr)
library(tidyverse)
library(readxl)
data1 = read.csv("EUA.csv")[-(1:662),]
X= data1$log_price
price=data1$price

# ==========================================================
# Random Forest regression forecasting - with standardization + error metrics
# ==========================================================
library(caret)
library(xgboost)
library(Metrics)

system.time({
  set.seed(123)
  
  # ------------------------------------------
  # 1. Data (can be replaced with your carbon price series)
  # ------------------------------------------
  T <- nrow(data1)
  t <- 1:T
  S <- log(data1$price) 
  
  # Construct lag features (lag = 3)
  lag <- 1
  X <- embed(S, lag + 1)
  y <- X[,1]
  X <- X[,-1, drop = FALSE]
  
  # Train-test split
  n_train <- round(1* nrow(X))
  X_train <- X[1:n_train,]; y_train <- y[1:n_train]
  X_test <- X[nrow(X), , drop = FALSE]; y_test <- y[nrow(X)]
  
  # ------------------------------------------
  # 2. Standardization (based on the training set)
  # ------------------------------------------
  X_train_scaled <- scale(X_train)
  X_test_scaled <- scale(X_test,
                         center = attr(X_train_scaled, "scaled:center"),
                         scale  = attr(X_train_scaled, "scaled:scale"))
  y_mean <- mean(y_train)
  y_sd   <- sd(y_train)
  y_train_scaled <- (y_train - y_mean) / y_sd
  
  # ------------------------------------------
  # 3. Train XGBoost model
  # ------------------------------------------
  dtrain <- xgb.DMatrix(data = X_train_scaled, label = y_train_scaled)
  
  cv <- xgb.cv(
    params = list(objective="reg:squarederror", eta=0.05, max_depth=3),
    data = dtrain,
    nrounds = 300,
    nfold = 5,
    metrics = "rmse",
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  best_nrounds <- cv$best_iteration
  
  xgb_model <- xgb.train(
    params = list(objective="reg:squarederror", eta=0.05, max_depth=3),
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )
  
})
# ------------------------------------------
# 4. In-sample prediction (inverse standardization)
# ------------------------------------------
y_pred_scaled <- predict(xgb_model, X_train_scaled, iteration_range = c(1, best_nrounds))
y_pred <- y_pred_scaled * y_sd + y_mean
resid  <- y_train - y_pred 
sigma2 <- var(resid, na.rm = TRUE)   

# Price series (true & predicted) — mean-unbiased: S_hat = exp(m + 0.5 v)
S_train_true <- exp(y_train)
S_train_hat  <- exp(y_pred + 0.5 * sigma2)       # Unbiased mean forecast (recommended for RMSE/MAE)

calc_metrics <- function(y, yhat){
  mse <- mean((y - yhat)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(y - yhat))
  rae <- sum(abs(y - yhat)) / sum(abs(y - mean(y)))
  mdape <- median(abs((y - yhat) / y))
  return(c(MSE=mse, RMSE=rmse, MAE=mae, RAE=rae, MdAPE=mdape))
}

err_in_price <- calc_metrics(S_train_true, S_train_hat)
err_in_price

#===================== Interval estimation diagnostic results =====================#

# Interval metrics function -------------------------------------------------
interval_metrics <- function(y_true, lower, upper) {
  N <- length(y_true)
  ymax <- max(y_true, na.rm = TRUE)
  ymin <- min(y_true, na.rm = TRUE)
  
  # 1. PICP: coverage probability
  picp <- mean((y_true >= lower) & (y_true <= upper), na.rm = TRUE)
  
  # 2. PINAW: normalized interval width
  pinaw <- mean(upper - lower, na.rm = TRUE) / (ymax - ymin)
  
  # 3. AWD: average width deviation outside interval
  delta <- ifelse(y_true < lower, (lower - y_true)/(upper - lower),
                  ifelse(y_true > upper, (y_true - upper)/(upper - lower), 0))
  awd <- mean(delta, na.rm = TRUE)
  
  return(c(PICP = picp, PINAW = pinaw, AWD = awd))
}

#----------------------------------------------------------
# Residual standard deviation (log space)
sigma_hat <- sqrt(sigma2)

# Normal quantiles (confidence levels α=0.95, 0.85, 0.75)
z_95 <- qnorm(0.975)
z_85 <- qnorm(0.925)
z_75 <- qnorm(0.875)

# Prediction intervals in log space
L95_log <- y_pred - z_95 * sigma_hat
U95_log <- y_pred + z_95 * sigma_hat
L85_log <- y_pred - z_85 * sigma_hat
U85_log <- y_pred + z_85 * sigma_hat
L75_log <- y_pred - z_75 * sigma_hat
U75_log <- y_pred + z_75 * sigma_hat

# Convert to price space
L95 <- exp(L95_log)
U95 <- exp(U95_log)
L85 <- exp(L85_log)
U85 <- exp(U85_log)
L75 <- exp(L75_log)
U75 <- exp(U75_log)

# True prices (in-sample)
S_true <- exp(y_train)

#----------------------------------------------------------
# Compute metrics under three confidence intervals
metrics_95 <- interval_metrics(S_true, L95, U95)
metrics_85 <- interval_metrics(S_true, L85, U85)
metrics_75 <- interval_metrics(S_true, L75, U75)

# Summarize results
results <- data.frame(
  Alpha = c(0.95, 0.85, 0.75),
  PICP  = c(metrics_95["PICP"], metrics_85["PICP"], metrics_75["PICP"]),
  PINAW = c(metrics_95["PINAW"], metrics_85["PINAW"], metrics_75["PINAW"]),
  AWD   = c(metrics_95["AWD"],  metrics_85["AWD"],  metrics_75["AWD"])
)

cat("\n---- Interval estimation diagnostic results ----\n")
print(results)


# ===================== Three-step out-of-sample recursive forecasting (price space) ===================== #

# -------------------------------
# 1️⃣ Prepare data
# -------------------------------
lag <- 1
S <- log(data1$price)
X_full <- embed(S, lag + 1)
y_full <- X_full[, 1]
X_full <- X_full[, -1, drop = FALSE]


# Split training set (first n-3) and test set (last 3)
n_total <- nrow(X_full)
n_train <- n_total - 3
X_train <- X_full[1:n_train, ]
y_train <- y_full[1:n_train]
X_test_init <- X_full[n_train, , drop = FALSE]  # Use the last true point as the starting input
y_test_true <- y_full[(n_train + 1):n_total]     # For comparison

# Standardization (consistent with the training set)
X_train_scaled <- scale(X_train)
y_mean <- mean(y_train)
y_sd   <- sd(y_train)

X_test_scaled <- scale(
  X_test_init,
  center = attr(X_train_scaled, "scaled:center"),
  scale  = attr(X_train_scaled, "scaled:scale")
)
y_train_scaled <- (y_train - y_mean) / y_sd

# Retrain XGBoost model (you can also directly reuse xgb_model above)
dtrain <- xgb.DMatrix(data = X_train_scaled, label = y_train_scaled)

cv <- xgb.cv(
  params = list(objective="reg:squarederror", eta=0.05, max_depth=3),
  data = dtrain,
  nrounds = 300,
  nfold = 5,
  metrics = "rmse",
  early_stopping_rounds = 20,
  verbose = 0
)

best_nrounds <- cv$best_iteration

xgb_model <- xgb.train(
  params = list(objective="reg:squarederror", eta=0.05, max_depth=3),
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

# -------------------------------
# 2️⃣ Three-step recursive forecasting (in log space)
# -------------------------------
steps <- 3
y_pred_scaled <- numeric(steps)
last_input_scaled <- X_test_scaled[1]  # Starting input

for (h in 1:steps) {
  # 1️⃣ Predict next log value (standardized space)
  y_pred_scaled[h] <- predict(xgb_model, as.matrix(last_input_scaled))
  
  # 2️⃣ Inverse standardization back to log space
  y_pred_log <- y_pred_scaled[h] * y_sd + y_mean
  
  # 3️⃣ Inverse standardization of the previous input back to log space
  last_input_unscaled <- as.numeric(last_input_scaled) *
    attr(X_train_scaled, "scaled:scale") +
    attr(X_train_scaled, "scaled:center")
  
  # 4️⃣ Update lags
  n_lags <- ncol(X_train_scaled)
  
  if (n_lags > 1) {
    # If lag>1, drop the oldest lag term
    new_input_log <- c(y_pred_log, last_input_unscaled[1:(n_lags - 1)])
  } else {
    # If lag=1, input only contains the latest prediction
    new_input_log <- c(y_pred_log)
  }
  
  # 5️⃣ Ensure consistent dimension and standardize
  new_input_log <- matrix(new_input_log, nrow = 1, ncol = n_lags)
  new_input_scaled <- (new_input_log - attr(X_train_scaled, "scaled:center")) /
    attr(X_train_scaled, "scaled:scale")
  
  # Update input for the next step
  last_input_scaled <- new_input_scaled
  
  # 6️⃣ Store log-space predictions
  if (h == 1) {
    log_preds <- y_pred_log
  } else {
    log_preds <- c(log_preds, y_pred_log)
  }
}

# -------------------------------
# 3️⃣ Convert back to price space with unbiased correction
# -------------------------------
#sigma2_out <- var(y_train - (y_train_scaled * y_sd + y_mean), na.rm = TRUE)
resid <- y_train - (predict(xgb_model, as.matrix(X_train_scaled))* y_sd + y_mean)
sigma2_out = var(resid)
pred_price <- exp(log_preds + 0.5 * sigma2_out)

# Corresponding true prices (last 3 points in the test set)
true_price <- exp(y_test_true)

# -------------------------------
# 4️⃣ Error metrics
# -------------------------------
MSE  <- mean((pred_price - true_price)^2)
RMSE <- sqrt(MSE)
MdAPE <- median(abs(pred_price - true_price) / true_price)
MAE  <- mean(abs(pred_price - true_price))
RAE  <- sum(abs(pred_price - true_price)) / sum(abs(true_price - mean(exp(y_train))))

# Output results
result_out <- data.frame(
  Step = 1:3,
  True = round(true_price, 4),
  Pred = round(pred_price, 4)
)

cat("\n===== XGBoost three-step out-of-sample recursive forecast (price space) =====\n")
print(result_out)

cat("\n===== Out-of-sample error metrics =====\n")
cat("MSE =", round(MSE,6),
    " RMSE =", round(RMSE,6),
    " MdAPE =", round(MdAPE,6),
    " MAE =", round(MAE,6),
    " RAE =", round(RAE,6), "\n")


# ======================================================
# 🔹 4️⃣ Interval estimation (log space ± z*sigma)
# ======================================================


# Residuals in standardized space
resid <- y_train - (predict(xgb_model, as.matrix(X_train_scaled))* y_sd + y_mean)
sigma_hat <- sd(resid, na.rm = TRUE)


# Normal quantiles (confidence levels)
z_95 <- qnorm(0.975)
z_85 <- qnorm(0.925)
z_75 <- qnorm(0.875)

# Lower/upper bounds at each confidence level (log space)
L95_log <- log_preds - z_95 * sigma_hat
U95_log <- log_preds + z_95 * sigma_hat

L85_log <- log_preds - z_85 * sigma_hat
U85_log <- log_preds + z_85 * sigma_hat

L75_log <- log_preds - z_75 * sigma_hat
U75_log <- log_preds + z_75 * sigma_hat

# Convert to price space (with unbiased correction)
L95 <- exp(L95_log )
U95 <- exp(U95_log )

L85 <- exp(L85_log )
U85 <- exp(U85_log )

L75 <- exp(L75_log )
U75 <- exp(U75_log )
