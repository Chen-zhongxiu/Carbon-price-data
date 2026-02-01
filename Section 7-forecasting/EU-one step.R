normalize_vec <- function(x) {
  s <- sum(x)
  if (!is.finite(s) || s <= 0) rep(1/length(x), length(x)) else x / s
}

# Online one-step filtering update: given alpha_{t-1|t-1}, return alpha_{t|t}
online_filter_update_Q <- function(alpha_prev, dx_t, A, params, N, dt=1) {
  M <- N*N
  # 1-step predictive over z_t
  alpha_pred <- as.numeric(A %*% alpha_prev)   # length M
  
  # emission under Q: f_Q(dx_t | z_t)
  fQ <- emit_mix_density_Q(dx_t, params, N, dt = dt)  # length M
  
  alpha_new <- alpha_pred * fQ
  alpha_new <- normalize_vec(alpha_new)
  alpha_new
}

# Extract fit-like parameter structure from a history snapshot
extract_fit_from_hist <- function(h, N_state) {
  # h can be a list or a named vector
  if (is.list(h)) {
    mu      <- h$mu
    sigma   <- h$sigma
    lambda  <- h$lambda
    mu_beta <- h$mu_beta
    s_beta  <- h$s_beta
    p       <- h$p
  } else {
    v <- h
    nms <- names(v)
    stopifnot(!is.null(nms))
    
    grab_vec <- function(key) {
      idx <- grep(paste0("^", key, "\\b|^", key, "\\["), nms)
      if (length(idx) == 0) idx <- grep(key, nms, fixed = TRUE)
      v[idx]
    }
    mu      <- as.numeric(grab_vec("mu"))[1:N_state]
    sigma   <- as.numeric(grab_vec("sigma"))[1:N_state]
    lambda  <- as.numeric(grab_vec("lambda"))[1:N_state]
    mu_beta <- as.numeric(grab_vec("mu_beta"))[1:N_state]
    s_beta  <- as.numeric(grab_vec("s_beta"))[1:N_state]
    
    # Parsing p: assume history stores p directly (list/array); otherwise you need to store p separately in history
    p <- attr(h, "p")
    if (is.null(p)) stop("Cannot find p in the history snapshot; please make sure history includes p (transition tensor).")
  }
  
  list(mu=mu, sigma=sigma, lambda=lambda, mu_beta=mu_beta, s_beta=s_beta, p=p)
}

# fit-like -> params_hat + A_hat
fit_to_paramsA <- function(fit_like, dt=1) {
  params_hat <- list(
    theta   = (fit_like$mu - 0.5 * fit_like$sigma^2) * dt,
    rho2    = (fit_like$sigma^2) * dt,
    lambda  = fit_like$lambda,
    mu_beta = fit_like$mu_beta,
    s_beta2 = (fit_like$s_beta^2)
  )
  A_hat <- build_A_from_p(fit_like$p)
  list(params=params_hat, A=A_hat)
}

interval_metrics <- function(y_true, lower, upper) {
  N <- length(y_true)
  ymax <- max(y_true); ymin <- min(y_true)
  
  picp <- mean((y_true >= lower) & (y_true <= upper))
  pinaw <- mean(upper - lower) / (ymax - ymin)
  
  delta <- numeric(N)
  for (i in 1:N) {
    if (y_true[i] < lower[i]) {
      delta[i] <- (lower[i] - y_true[i]) / (upper[i] - lower[i])
    } else if (y_true[i] > upper[i]) {
      delta[i] <- (y_true[i] - upper[i]) / (upper[i] - lower[i])
    } else delta[i] <- 0
  }
  awd <- mean(delta)
  c(PICP=picp, PINAW=pinaw, AWD=awd)
}

# Strict online one-step-ahead prediction evaluation
online_eval_one_step <- function(X_full, fit0, N_state,
                                 dt=1,
                                 batch_size=20,   # If you know the number of increments per batch, pass it directly
                                 level=c(0.95,0.85,0.75),
                                 interval_method=c("fast","mc"),
                                 nsim=2000, seed=123) {
  
  interval_method <- match.arg(interval_method)
  level <- sort(level, decreasing = TRUE)
  
  dx <- diff(X_full)
  Tn <- length(dx)
  T_all <- length(X_full)
  S_obs <- exp(X_full)
  
  # --- Decide batch_size: if not given, try to infer from fit0$n_batch ---
  if (is.null(batch_size)) {
    n_batch <- fit0$n_batch
    if (is.null(n_batch)) stop("Please pass batch_size, or ensure fit0$n_batch exists.")
    
    # Common case: n_batch is the number of batches
    if (n_batch <= Tn) batch_size <- ceiling(Tn / n_batch) else batch_size <- n_batch
  }
  
  # Number of history snapshots
  hist_list <- fit0$history
  if (is.null(hist_list) || length(hist_list) == 0) stop("fit0$history is empty.")
  
  # Extended state index
  M <- N_state * N_state
  curr_state <- (((1:M) - 1L) %% N_state) + 1L
  
  # Initial extended-state distribution (strict online: prior only)
  alpha <- rep(1/M, M)
  
  # Output container: predicting t+1, so t=1..Tn yields S_pred[t+1]
  S_pred <- rep(NA_real_, T_all)
  S_pred[1] <- S_obs[1]
  
  L <- matrix(NA_real_, nrow=T_all, ncol=length(level))
  U <- matrix(NA_real_, nrow=T_all, ncol=length(level))
  colnames(L) <- paste0("L", level)
  colnames(U) <- paste0("U", level)
  L[1,] <- U[1,] <- S_obs[1]
  
  z <- qnorm((1 + level)/2)
  
  set.seed(seed)
  t_start <- Sys.time()
  
  # Helper: use current alpha (representing z_t|X_1:t) to predict at t+1
  predict_next <- function(alpha_t, A, params, S_t) {
    # z_{t+1|t}
    pred_ext <- as.numeric(A %*% alpha_t)
    pred_ext <- normalize_vec(pred_ext)
    
    # y_{t+1} weights
    w <- numeric(N_state)
    for (a in 1:N_state) w[a] <- sum(pred_ext[curr_state == a])
    
    theta  <- params$theta
    rho2   <- params$rho2
    lambda <- params$lambda
    mu_b   <- params$mu_beta
    sb2    <- params$s_beta2
    
    # Regime-conditional mean/variance of ΔX (first moments under your model)
    m_a <- theta + lambda * mu_b
    v_a <- rho2  + lambda * sb2
    
    
    mean_dx <- sum(w * m_a)
    var_dx  <- sum(w * (v_a + m_a^2)) - mean_dx^2
    var_dx  <- max(var_dx, 1e-12)
    sd_dx   <- sqrt(var_dx)
    
    # Point forecast: E[S_{t+1}|info] = S_t * E[e^{ΔX}|info]
    c_a <- (1 - lambda) * exp(theta + 0.5*rho2) +
      lambda * exp(theta + mu_b + 0.5*(rho2 + sb2))
    point <- S_t * sum(w * c_a)
    
    list(point=point, mean_log = log(S_t) + mean_dx, sd_log = sd_dx, w = w)
  }
  
  # Main loop: t=1..Tn
  for (t in 1:Tn) {
    # --- Which batch does the current time index belong to (1-based) ---
    b <- ceiling(t / batch_size)
    b <- min(b, length(hist_list))  # Avoid out-of-range
    
    fit_like <- extract_fit_from_hist(hist_list[[b]], N_state)
    pa <- fit_to_paramsA(fit_like, dt=dt)
    params <- pa$params
    A <- pa$A
    
    # 1) After observing dx_t, update alpha_t = P(z_t | X_1:t)
    alpha <- online_filter_update_Q(alpha_prev=alpha, dx_t=dx[t],
                                    A=A, params=params, N=N_state, dt=dt)
    
    # 2) Use alpha_t to predict next S_{t+1}
    pred <- predict_next(alpha_t = alpha, A=A, params=params, S_t=S_obs[t+0])
    
    S_pred[t+1] <- pred$point
    
    # 3) Interval
    if (interval_method == "fast") {
      # Lognormal approximation: X_{t+1} ~ N(mean_log, sd_log^2)
      for (j in seq_along(level)) {
        L[t+1,j] <- exp(pred$mean_log - z[j] * pred$sd_log)
        U[t+1,j] <- exp(pred$mean_log + z[j] * pred$sd_log)
      }
    } else {
      # MC: sample regime via pred$w, then sample jump, then sample normal
      theta  <- params$theta; rho2 <- params$rho2
      lambda <- params$lambda; mu_b <- params$mu_beta; sb2 <- params$s_beta2
      
      sim <- numeric(nsim)
      for (m in 1:nsim) {
        a <- sample(1:N_state, 1, prob = pred$w)
        J <- rbinom(1, 1, lambda[a])
        mean_dx <- theta[a] + ifelse(J==1, mu_b[a], 0)
        var_dx  <- rho2[a]  + ifelse(J==1, sb2[a], 0)
        sim[m]  <- S_obs[t] * exp(rnorm(1, mean_dx, sqrt(var_dx)))
      }
      for (j in seq_along(level)) {
        alpha_j <- (1 - level[j]) / 2
        L[t+1,j] <- as.numeric(quantile(sim, probs = alpha_j))
        U[t+1,j] <- as.numeric(quantile(sim, probs = 1 - alpha_j))
      }
    }
  }
  
  run_time <- as.numeric(difftime(Sys.time(), t_start, units="secs"))
  
  # ---- Point forecast errors (strict online: predictions from t=2..T_all) ----
  pred_vec <- S_pred[2:T_all]
  true_vec <- S_obs[2:T_all]
  
  MSE  <- mean((pred_vec - true_vec)^2, na.rm=TRUE)
  RMSE <- sqrt(MSE)
  MAE  <- mean(abs(pred_vec - true_vec), na.rm=TRUE)
  MdAPE <- median(abs(pred_vec - true_vec) / true_vec, na.rm=TRUE)
  
  # RAE denominator uses the mean of the same true segment (strictly consistent)
  RAE <- sum(abs(pred_vec - true_vec), na.rm=TRUE) /
    sum(abs(true_vec - mean(true_vec, na.rm=TRUE)), na.rm=TRUE)
  
  perf_point <- data.frame(MSE=MSE, RMSE=RMSE, MAE=MAE, MdAPE=MdAPE, RAE=RAE, run_time=run_time)
  
  # ---- Interval metrics ----
  # ---- Interval metrics ----
  perf_interval <- data.frame(Alpha=level, PICP=NA, PINAW=NA, AWD=NA)
  for (j in seq_along(level)) {
    met <- interval_metrics(true_vec, L[2:T_all, j], U[2:T_all, j])
    perf_interval$PICP[j]  <- met["PICP"]
    perf_interval$PINAW[j] <- met["PINAW"]
    perf_interval$AWD[j]   <- met["AWD"]
  }
  
  # ---- Prediction output table: expand each L/U column into plain columns ----
  pred_df <- data.frame(
    t      = 1:T_all,
    S_obs  = S_obs,
    S_pred = S_pred
  )
  for (j in seq_along(level)) {
    a <- level[j]
    pred_df[[paste0("L", a)]] <- L[, j]
    pred_df[[paste0("U", a)]] <- U[, j]
  }
  
  list(
    pred = pred_df,
    perf_point = perf_point,
    perf_interval = perf_interval,
    settings = list(batch_size=batch_size, interval_method=interval_method, nsim=nsim)
  )
  
}

#=======================Data==========================================
data = read_xlsx("欧盟排放配额(EUA).xlsx")


#跑数据
library(dplyr)
library(tidyverse)
library(readxl)
data1 = read.csv("EUA.csv")[-(1:662),]
X= data1$log_price
price=data1$price

X_full = X
T_all = length(X)
T_test  <- 3
T_train <- T_all - T_test

X_train      <- X_full[1:T_train]
X_test       <- X_full[(T_train+1):T_all]


#==============================N=1=========================================
N=1
mu_init      <- c(bench$par[1])
sigma_init   <- c(bench$par[2])
lambda_init  <- c(bench$par[3])
mu_beta_init <- c(bench$par[4])
s_beta_init  <- c(bench$par[5])

p_init <- array(1, dim = c(1, 1, 1)) 




# Initial transition probabilities: add a small perturbation around the true value
p_init <- array(0, dim = c(N, N, N))

set.seed(2025)
dt <- 1
N=1
time_fit <- system.time({
  fit0 <- em_fit_jump_mixture_rn_batch_global(
    X = X_train, dt = dt, N = N,
    p_init = p_init, mu_init = mu_init, sigma_init = sigma_init,
    lambda_init = lambda_init, mu_beta_init = mu_beta_init, s_beta_init = s_beta_init,
    n_batch = 210, esscher_theta = 0, verbose = FALSE
  )
})
time_fit

# Strict online evaluation (use fast intervals; for higher accuracy set interval_method="mc")
out_online <- online_eval_one_step(
  X_full  = X_train,
  fit0   = fit0,
  N_state = N,
  dt = dt,
  batch_size = 20,          # If unsure, let it infer automatically; if you know increments per batch, it's better to set it manually
  interval_method = "fast"
)

print(out_online$perf_point)
print(out_online$perf_interval)

plot(out_online$pred$S_obs)
#==============================N=2=========================================
N=2

bench$par
mu_init      <- c(0.00123,0.00245)
sigma_init   <- c(0.01755 , 0.029456)
lambda_init  <- c(0.1, 0.2)
mu_beta_init <- c(-0.0052, -0.007)
s_beta_init  <- c(0.046435, 0.05342)

# Initial transition probabilities: add a small perturbation around the true value
p_init <- array(0, dim = c(N, N, N))
p_init[,1,1] <- c(0.80, 0.20)
p_init[,2,1] <- c(0.55, 0.45)
p_init[,1,2] <- c(0.35, 0.65)
p_init[,2,2] <- c(0.15, 0.85)


set.seed(2025)
dt <- 1
time_fit <- system.time({fit0 <- em_fit_jump_mixture_rn_batch_global(
  X = X_train, dt = dt, N = N,
  p_init = p_init,
  mu_init = mu_init, sigma_init = sigma_init,
  lambda_init = lambda_init,
  mu_beta_init = mu_beta_init,
  s_beta_init  = s_beta_init,
  n_batch = 243,
  esscher_theta = 0,
  verbose = FALSE
)
})

# Strict online evaluation (use fast intervals; for higher accuracy set interval_method="mc")
out_online <- online_eval_one_step(
  X_full  = X_train,
  fit0   = fit0,
  N_state = N,
  dt = dt,
  batch_size = 20,          # If unsure, let it infer automatically; if you know increments per batch, it's better to set it manually
  interval_method = "fast"
)

print(out_online$perf_point)
print(out_online$perf_interval)

plot(out_online$pred$S_obs)
plot(out_online$pred$S_pred)
#==============================N=3=========================================
N=3
bench$par

mu_init      <- c(0.001,0.00245,0.00345)
sigma_init   <- c(0.01755 , 0.02462,0.03456)
lambda_init  <- c(0.1, 0.2,0.3)
mu_beta_init <- c(-0.0052, -0.007,-0.008)
s_beta_init  <- c(0.046435, 0.05342,0.06634)


# Initial transition probabilities: add a small perturbation around the true value
p_init <- array(0, dim = c(N, N, N))
# prev = 1
p_init[,1,1] <- c(0.50, 0.30, 0.20)  # curr=1
p_init[,2,1] <- c(0.3, 0.2, 0.5)  # curr=2
p_init[,3,1] <- c(0.2, 0.4, 0.4)  # curr=3

# prev = 2
p_init[,1,2] <- c(0.2, 0.3, 0.5)  # curr=1
p_init[,2,2] <- c(0.3, 0.5, 0.2)  # curr=2
p_init[,3,2] <- c(0.5, 0.1, 0.4)  # curr=3

# prev = 3
p_init[,1,3] <- c(0.4, 0.4, 0.2)  # curr=1
p_init[,2,3] <- c(0.3, 0.3, 0.4)  # curr=2
p_init[,3,3] <- c(0.3, 0.2, 0.5)  # curr=3


set.seed(2025)
dt <- 1
N=3
time_fit <- system.time({fit0 <- em_fit_jump_mixture_rn_batch_global(
  X = X_train, dt = dt, N = N,
  p_init = p_init,
  mu_init = mu_init, sigma_init = sigma_init,
  lambda_init = lambda_init,
  mu_beta_init = mu_beta_init,
  s_beta_init  = s_beta_init,
  n_batch = 210,
  esscher_theta = 0,
  verbose = FALSE
)
})
time_fit
# Strict online evaluation (use fast intervals; for higher accuracy set interval_method="mc")
out_online <- online_eval_one_step(
  X_full  = X_train,
  fit0   = fit0,
  N_state = N,
  dt = dt,
  batch_size = 20,          # If unsure, let it infer automatically; if you know increments per batch, it's better to set it manually
  interval_method = "fast"
)

print(out_online$perf_point)
print(out_online$perf_interval)

plot(out_online$pred$S_obs)
