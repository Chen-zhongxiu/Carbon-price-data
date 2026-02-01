###########################################################
## A) Extract params_hat + A_hat from an offline fit
###########################################################
fit_to_paramsA_offline <- function(fit, dt = 1, N_state = 2) {
  params_hat <- list(
    theta   = (fit$mu - 0.5 * fit$sigma^2) * dt,
    rho2    = (fit$sigma^2) * dt,
    lambda  = fit$lambda,
    mu_beta = fit$mu_beta,
    s_beta2 = fit$s_beta^2
  )
  A_hat <- build_A_from_p(fit$p)
  M <- N_state * N_state
  curr_state <- (((1:M) - 1L) %% N_state) + 1L
  list(params = params_hat, A = A_hat, M = M, curr_state = curr_state)
}

###########################################################
## B) Given a starting point t0, run forward using history 1:t0
##    to obtain the predictive distribution of z_{t0}
##    Here z_t = (y_{t-1}, y_t), and dx_t is driven by y_t (consistent with your current code)
###########################################################
get_pred_z_t0 <- function(X_full_log, t0, A_hat, params_hat, N_state, dt = 1) {
  # t0 is the index of the price series (1..T_all)
  # Need history up to t0: dx_hist = diff(X[1:t0]), length t0-1
  stopifnot(t0 >= 2)
  
  dx_hist <- diff(X_full_log[1:t0])
  Tn_hist <- length(dx_hist)   # = t0-1
  M <- N_state * N_state
  
  # forward filter using history only
  ff <- forward_filter_mixture_rn(
    dx = dx_hist,
    A = A_hat,
    params = params_hat,
    N = N_state,
    dt = dt,
    esscher_theta = 0
  )
  gamma_filt <- ff$gamma_filt  # M x Tn_hist, corresponds to z_1..z_{t0-1}
  
  # Prior prediction to z_{t0}:  gamma_pred0 = A * P(z_{t0-1}|hist)
  gamma_last <- gamma_filt[, Tn_hist]
  gamma_pred0 <- as.numeric(A_hat %*% gamma_last)
  gamma_pred0 <- gamma_pred0 / sum(gamma_pred0)
  
  gamma_pred0
}

###########################################################
## C) Multi-step forecasting (analytic point forecast + MC intervals)
###########################################################
forecast_multi_step_offline <- function(X_full_log, fit_offline, N_state,
                                        k = 3, H = 3, dt = 1,
                                        level = c(0.95, 0.85, 0.75),
                                        nsim = 20000, seed = 123) {
  
  T_all <- length(X_full_log)
  stopifnot(k >= 1, H >= 1, k <= T_all - H)
  
  obj <- fit_to_paramsA_offline(fit_offline, dt = dt, N_state = N_state)
  params_hat <- obj$params
  A_hat      <- obj$A
  M          <- obj$M
  curr_state <- obj$curr_state
  
  theta  <- params_hat$theta
  rho2   <- params_hat$rho2
  lambda <- params_hat$lambda
  mu_b   <- params_hat$mu_beta
  sb2    <- params_hat$s_beta2
  
  # Regime growth factor: c_a = E[exp(dx_t) | y_t=a]
  c_a <- (1 - lambda) * exp(theta + 0.5 * rho2) +
    lambda * exp(theta + mu_b + 0.5 * (rho2 + sb2))
  
  # Extended state j growth factor g_j = c_{curr_state(j)}
  g_j <- c_a[curr_state]  # length M
  Dg  <- diag(g_j, nrow = M, ncol = M)
  
  # For analytic point prediction: u_{t+1} = A * D(g) * u_t
  B <- A_hat %*% Dg
  
  level <- sort(level, decreasing = TRUE)
  alpha_q <- (1 - level) / 2
  
  S_full <- exp(X_full_log)
  
  set.seed(seed)
  res_list <- list()
  idx <- 1
  
  # Origin set: last k origins, ensuring each has H-step truth available
  for (t0 in (T_all - H - k + 1):(T_all - H)) {
    
    # 1) Obtain P(z_{t0} | info up to t0)
    gamma_pred0 <- get_pred_z_t0(
      X_full_log = X_full_log,
      t0 = t0,
      A_hat = A_hat,
      params_hat = params_hat,
      N_state = N_state,
      dt = dt
    )
    
    S_t0 <- S_full[t0]
    
    # ======================================================
    # 2) Point forecast (analytic)
    #    u0 = S_t0 * P(z_t0)
    #    u_h = (A Dg)^h u0
    #    E[S_{t0+h}] = 1' u_h
    # ======================================================
    u <- S_t0 * gamma_pred0
    point_h <- numeric(H)
    for (h in 1:H) {
      u <- as.numeric(B %*% u)
      point_h[h] <- sum(u)
    }
    
    # ======================================================
    # 3) Interval (MC), simulate starting from z_{t0} ~ gamma_pred0
    # ======================================================
    sim_S <- matrix(NA_real_, nrow = nsim, ncol = H)
    
    for (m in 1:nsim) {
      z_idx <- sample.int(M, size = 1, prob = gamma_pred0)
      S_curr <- S_t0
      
      for (h in 1:H) {
        a <- curr_state[z_idx]
        
        J <- rbinom(1, 1, lambda[a])
        mean_dx <- theta[a] + ifelse(J == 1, mu_b[a], 0)
        var_dx  <- rho2[a]  + ifelse(J == 1, sb2[a], 0)
        dx_draw <- rnorm(1, mean_dx, sqrt(max(var_dx, 1e-12)))
        
        S_curr <- S_curr * exp(dx_draw)
        sim_S[m, h] <- S_curr
        
        if (h < H) {
          z_idx <- sample.int(M, size = 1, prob = A_hat[, z_idx])
        }
      }
    }
    
    # 4) Extract intervals for each horizon
    for (h in 1:H) {
      S_true_h <- S_full[t0 + h]
      S_sim_h  <- sim_S[, h]
      
      out_row <- data.frame(
        origin_index = t0,
        horizon = h,
        target_index = t0 + h,
        S_true = S_true_h,
        S_point = point_h[h]
      )
      
      for (j in seq_along(level)) {
        out_row[[paste0("L", level[j])]] <- as.numeric(quantile(S_sim_h, probs = alpha_q[j]))
        out_row[[paste0("U", level[j])]] <- as.numeric(quantile(S_sim_h, probs = 1 - alpha_q[j]))
      }
      
      res_list[[idx]] <- out_row
      idx <- idx + 1
    }
    
    message("done t0 = ", t0)
  }
  
  do.call(rbind, res_list)
}


em_fit_jump_mixture_rn <- function(
    X, dt, N,
    p_init,
    mu_init, sigma_init, lambda_init, mu_beta_init, s_beta_init,
    max_iter = 1000, tol = 1e-5, verbose = TRUE,
    esscher_theta = 0      # <--- New parameter: Esscher tilting intensity
) {
  dx <- diff(X)
  Tn <- length(dx)
  
  theta_init <- (mu_init - 0.5 * sigma_init^2) * dt
  rho2_init  <- (sigma_init^2) * dt
  
  params <- list(
    theta   = theta_init,
    rho2    = rho2_init,
    lambda  = lambda_init,
    mu_beta = mu_beta_init,
    s_beta2 = s_beta_init^2
  )
  
  p  <- p_init
  ll_old <- -Inf
  
  for (it in 1:max_iter) {
    A  <- build_A_from_p(p)
    
    fb <- fb_augmented_mixture_rn(
      dx, A, params, N,
      dt = dt,
      esscher_theta = esscher_theta   # <--- Pass through
    )
    gamma  <- fb$gamma
    xi_sum <- fb$xi_sum
    
    stats   <- jump_posterior(dx, params, N, gamma)
    par_new <- mstep_params(dx, stats, params, N)
    p_new   <- mstep_transition(xi_sum, N)
    
    params <- par_new
    p      <- p_new
    
    # if (verbose) {
    #  cat(sprintf("Iter %3d | loglik_P=%.6f | theta_E=%.3f | lambda=%s\n",
    #             it, fb$loglik_P, esscher_theta,
    #            paste(round(params$lambda, 4), collapse=",")))
    #}
    
    if (abs(fb$loglik_P - ll_old) < tol * (1 + abs(ll_old))) break
    ll_old <- fb$loglik_P
  }
  
  mu_hat    <- params$theta / dt + 0.5 * (params$rho2 / dt)
  sigma_hat <- sqrt(params$rho2 / dt)
  
  list(
    mu      = as.numeric(mu_hat),
    sigma   = as.numeric(sigma_hat),
    lambda  = as.numeric(params$lambda),
    mu_beta = as.numeric(params$mu_beta),
    s_beta  = as.numeric(sqrt(params$s_beta2)),
    p       = p,
    loglik_P      = ll_old,
    esscher_theta = esscher_theta
  )
}
#=======================Data==========================================
library(dplyr)
library(tidyverse)
library(readxl)
data1 = read.csv("EUA.csv")
X= data1$log_price
price=data1$price

X_full = X
T_all = length(X)
T_test  <- 3
T_train <- T_all - T_test

X_train      <- X_full[1:T_train]
X_test       <- X_full[(T_train+1):T_all]

X= data1$log_price
bench <- fit_benchmark_optim(X_train, dt = 1)
bench$par
#==============================N=1=========================================
N=1
mu_init      <- c(bench$par[1])
sigma_init   <- c(bench$par[2])
lambda_init  <- c(bench$par[3])
mu_beta_init <- c(bench$par[4])
s_beta_init  <- c(bench$par[5])

p_init <- array(1, dim = c(1, 1, 1)) 


#==============================N=2=========================================
dx_train <- diff(X_train)
N        <- 2
dt       <- 1

# Initial values
mu_init      <- c(mean(dx_train) + 0.02, mean(dx_train))
sigma_init   <- rep(sd(dx_train) / 2, N)
lambda_init  <- c(0.2, 0.6)
mu_beta_init <- c(0.1, 0.3)
s_beta_init  <- c(0.2, 0.3)

# Initial transition probabilities: add a small perturbation around the true value
p_init <- array(0, dim = c(N, N, N))
p_init[,1,1] <- c(0.80, 0.20)
p_init[,2,1] <- c(0.55, 0.45)
p_init[,1,2] <- c(0.35, 0.65)
p_init[,2,2] <- c(0.15, 0.85)


#==============================N=3=========================================
N=3

mu_init      <- c(0.000321,0.0004253,0.0006234)
sigma_init   <- c(0.008455 ,0.009352, 0.01456)
lambda_init  <- c(0.4, 0.5,0.6)
mu_beta_init <- c(-0.0006,-0.0007, -0.00080)
s_beta_init  <- c(0.0452,0.05297, 0.0634)

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

#=======================================================================
fit_train <- em_fit_jump_mixture_rn(
  X = X_train, dt = dt, N = N,
  p_init = p_init,
  mu_init = mu_init, sigma_init = sigma_init,
  lambda_init = lambda_init,
  mu_beta_init = mu_beta_init,
  s_beta_init  = s_beta_init,
  esscher_theta = 0,
  verbose = TRUE
)

# Multi-step forecast (last k origins, each forecast H steps)
res_ms <- forecast_multi_step_offline(
  X_full_log  = X_full,     # Note: pass log(price); your X_full above is already log(price)
  fit_offline = fit_train,
  N_state = N,
  k = 1,
  H = 3,
  dt = 1,
  level = c(0.95),
  nsim = 20000,
  seed = 123
)

print(res_ms)
