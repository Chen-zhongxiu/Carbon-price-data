###########################################################
## 0. Tools: Augmented state indexing (2nd-order HMM -> N^2-dimensional 1st-order HMM)
###########################################################


pair_index <- function(prev, curr, N) {
  (prev - 1L) * N + curr
}

pair_decode <- function(idx, N) {
  prev <- (idx - 1L) %/% N + 1L
  curr <- (idx - 1L) %%  N + 1L
  c(prev = prev, curr = curr)
}

###########################################################
## 1. Build the augmented transition matrix A from p[a,b,c] (column-stochastic)
###########################################################


build_A_from_p <- function(p_tensor) {
  N <- dim(p_tensor)[1]
  M <- N * N
  A <- matrix(0, nrow = M, ncol = M)
  
  for (prev in 1:N) {
    for (curr in 1:N) {
      j <- pair_index(prev, curr, N)
      for (nxt in 1:N) {
        i <- pair_index(curr, nxt, N)
        A[i, j] <- p_tensor[nxt, curr, prev]
      }
      colsum <- sum(A[, j])
      if (colsum > 0) A[, j] <- A[, j] / colsum
    }
  }
  A
}

###########################################################
## 2. Observation density f_a(Δx) (two-component Gaussian mixture, under Q)
###########################################################


emit_mix_density_Q <- function(dx_t, params, N, dt = 1) {
  M <- N * N
  curr_state <- ((1:M - 1L) %% N) + 1L
  
  theta   <- params$theta[curr_state]   # ϑ_a * dt
  rho2    <- pmax(params$rho2[curr_state], 1e-8)   # ϱ_a^2 * dt
  lambda  <- params$lambda[curr_state]
  mu_b    <- params$mu_beta[curr_state]
  sb2     <- pmax(params$s_beta2[curr_state], 1e-8)
  
  # 权重裁剪到 (0,1)
  w1 <- pmin(pmax(lambda * dt, 0), 0.999)
  w0 <- pmax(1 - w1, 1e-6)
  
  m0 <- theta
  v0 <- rho2
  m1 <- theta + mu_b
  v1 <- rho2 + sb2
  
  sd0 <- sqrt(v0)
  sd1 <- sqrt(v1)
  
  dens0 <- dnorm(dx_t, mean = m0, sd = sd0)
  dens1 <- dnorm(dx_t, mean = m1, sd = sd1)
  
  f <- w0 * dens0 + w1 * dens1
  pmax(f, 1e-300)
}

###########################################################
## 3. Local RN increment φ_k(a) = f_a(Δx_k) / g(Δx_k)
##    Here g is the standard normal N(0, 1) density
###########################################################

emit_logrn_mixture <- function(dx_t, params, N,
                               dt = 1, base_sd = 1,
                               esscher_theta = esscher_theta) {
  M <- N * N
  curr_state <- ((1:M - 1L) %% N) + 1L
  
  # Q 下的混合密度 f_a(Δx)
  f_Q <- emit_mix_density_Q(dx_t, params, N, dt = dt)
  log_f_Q <- log(pmax(f_Q, 1e-300))
  
  # 参考测度 P 的简单密度 g(Δx)（这里取 N(0, base_sd^2)）
  log_g <- dnorm(dx_t, mean = 0, sd = base_sd, log = TRUE)
  
  # 没有 Esscher 时的局部 RN：log φ_k(a) = log f_a - log g
  log_phi <- log_f_Q - as.numeric(log_g)
  
  # ====== Esscher 倾斜：φ_k^(θ)(a) = e^{θΔx}/M_a(θ) * φ_k(a) ======
  if (esscher_theta != 0) {
    th  <- esscher_theta
    
    theta_a <- params$theta[curr_state]              # ϑ_a * dt
    rho2_a  <- pmax(params$rho2[curr_state], 1e-8)   # ϱ_a^2 * dt
    lambda_a<- params$lambda[curr_state]
    mu_b_a  <- params$mu_beta[curr_state]
    sb2_a   <- pmax(params$s_beta2[curr_state], 1e-8)
    
    # 对应当前时间步的 0/1 跳权重 w0, w1 (注意 λ_a dt)
    w1 <- pmin(pmax(lambda_a * dt, 0), 0.999)
    w0 <- pmax(1 - w1, 1e-6)
    
    # log M_a(θ) = log( w0 e^{θϑ_a + 0.5 θ^2 ϱ_a^2}
    #                 + w1 e^{θ(ϑ_a+μ_βa)+0.5 θ^2(ϱ_a^2+s_βa^2)} )
    log_part0 <- log(w0) + th * theta_a + 0.5 * th^2 * rho2_a
    log_part1 <- log(w1) + th * (theta_a + mu_b_a) +
      0.5 * th^2 * (rho2_a + sb2_a)
    m  <- pmax(log_part0, log_part1)
    logM <- m + log(exp(log_part0 - m) + exp(log_part1 - m))
    
    # 加 Esscher 因子： log φ^(θ) = log φ + θΔx - logM
    log_phi <- log_phi + th * dx_t - logM
  }
  # =====================================================
  
  list(
    log_phi = log_phi,          # 长度 M: log φ_k^(θ)(i)
    log_g   = as.numeric(log_g) # 标量: log g(Δx_k)
  )
}


###########################################################
## 4. Forward–backward algorithm (with Radon–Nikodym derivative)
##
##    Corresponds to Proposition 2: v_{k+1} = A D_{k+1} v_k,
##    implemented in log space for numerical stability.
###########################################################


fb_augmented_mixture_rn <- function(dx, A, params, N,
                                    dt = 1, base_sd = 1,
                                    esscher_theta = esscher_theta,
                                    init_logpi = NULL) {
  Tn <- length(dx)
  M  <- N * N
  logA <- log(pmax(A, 1e-300))
  
  logB   <- matrix(NA_real_, M, Tn)
  log_gv <- numeric(Tn)
  for (t in 1:Tn) {
    tmp <- emit_logrn_mixture(dx[t], params, N,
                              dt = dt, base_sd = base_sd,
                              esscher_theta = esscher_theta)
    logB[, t] <- tmp$log_phi
    log_gv[t] <- tmp$log_g
  }
  
  log_alpha <- matrix(-Inf, M, Tn)
  log_beta  <- matrix(0,    M, Tn)
  cvec      <- numeric(Tn)
  
  # init
  if (is.null(init_logpi)) {
    log_alpha[, 1] <- -log(M) + logB[, 1]
  } else {
    log_alpha[, 1] <- init_logpi + logB[, 1]
  }
  m1 <- max(log_alpha[, 1]); s1 <- sum(exp(log_alpha[, 1] - m1))
  cvec[1] <- m1 + log(s1)
  log_alpha[, 1] <- log_alpha[, 1] - cvec[1]  # normalize
  
  # forward
  for (t in 2:Tn) {
    tmp <- matrix(log_alpha[, t-1], nrow = M, ncol = M, byrow = TRUE) + logA
    logsum <- apply(tmp, 1, function(x) {
      m <- max(x); m + log(sum(exp(x - m)))
    })
    log_alpha[, t] <- logB[, t] + logsum
    mt <- max(log_alpha[, t]); st <- sum(exp(log_alpha[, t] - mt))
    cvec[t] <- mt + log(st)
    log_alpha[, t] <- log_alpha[, t] - cvec[t] # normalize
  }
  
  # backward (for smoothing)
  for (t in (Tn-1):1) {
    add <- logB[, t+1] + log_beta[, t+1]
    tmp <- matrix(add, nrow = M, ncol = M, byrow = TRUE) + t(logA)
    log_beta[, t] <- apply(tmp, 1, function(x) {
      m <- max(x); m + log(sum(exp(x - m)))
    }) - cvec[t+1]
  }
  
  # gamma (smoothing)
  log_gamma <- log_alpha + log_beta
  gamma <- matrix(NA_real_, M, Tn)
  for (t in 1:Tn) {
    mt <- max(log_gamma[, t]); st <- sum(exp(log_gamma[, t] - mt))
    log_gamma[, t] <- log_gamma[, t] - (mt + log(st))
    gamma[, t] <- exp(log_gamma[, t])
  }
  
  # alpha (filtering): already normalized
  alpha <- exp(log_alpha)  # M x Tn
  
  # xi_sum (smoothing-based; keep as is)
  xi_sum <- matrix(0, M, M)
  for (t in 2:Tn) {
    log_xi <- matrix(NA_real_, M, M)
    for (j in 1:M) {
      log_xi[, j] <- log_alpha[j, t-1] + logA[, j] + logB[, t] + log_beta[, t]
    }
    mt <- max(log_xi); st <- sum(exp(log_xi - mt))
    log_xi <- log_xi - (mt + log(st))
    xi_sum <- xi_sum + exp(log_xi)
  }
  
  loglik_P      <- sum(cvec)
  loglik_Qtheta <- loglik_P + sum(log_gv)
  
  # ====== 新增：把扩展状态边缘化到 y_t（curr state）======
  curr_state <- ((1:M - 1L) %% N) + 1L
  
  a_prob_filter <- matrix(0, N, Tn)  # from alpha
  a_prob_smooth <- matrix(0, N, Tn)  # from gamma
  for (t in 1:Tn) {
    for (j in 1:M) {
      a <- curr_state[j]
      a_prob_filter[a, t] <- a_prob_filter[a, t] + alpha[j, t]
      a_prob_smooth[a, t] <- a_prob_smooth[a, t] + gamma[j, t]
    }
  }
  
  list(
    # smoothing objects (for parameter estimation)
    gamma = gamma,
    xi_sum = xi_sum,
    
    # filtering objects (for state estimate)
    alpha = alpha,
    a_prob_filter = a_prob_filter,
    a_prob_smooth = a_prob_smooth,
    
    loglik_P = loglik_P,
    loglik_Qtheta = loglik_Qtheta
  )
}



###########################################################
## 5. Jump posterior r[a,t] and the state marginal probability γ_k(a)
###########################################################


jump_posterior <- function(dx, params, N, gamma) {
  Tn <- length(dx)
  M  <- N * N
  
  # 每个扩展状态对应的当前 regime a
  curr_state <- ((1:M - 1L) %% N) + 1L
  
  # a_prob[a,t] = P(y_t=a | data)
  a_prob <- matrix(0, N, Tn)
  for (t in 1:Tn) {
    for (j in 1:M) {
      a <- curr_state[j]
      a_prob[a, t] <- a_prob[a, t] + gamma[j, t]
    }
  }
  
  theta  <- params$theta
  rho2   <- pmax(params$rho2,  1e-8)
  lambda <- pmax(params$lambda,1e-8)
  mu_b   <- params$mu_beta
  sb2    <- pmax(params$s_beta2,1e-8)
  
  r <- matrix(0, N, Tn)
  logN <- function(x,m,v) -0.5*(log(2*pi) + log(v) + (x-m)^2/v)
  
  for (a in 1:N) {
    m0 <- theta[a]
    v0 <- rho2[a]
    m1 <- theta[a] + mu_b[a]
    v1 <- rho2[a] + sb2[a]
    
    logphi0 <- logN(dx, m0, v0)
    logphi1 <- logN(dx, m1, v1)
    log_num <- log(lambda[a]) + logphi1
    log_den <- log(exp(log(1 - lambda[a]) + logphi0) + exp(log_num))
    r[a, ]  <- exp(log_num - log_den)
  }
  
  list(r = r, a_prob = a_prob)
}

###########################################################
## 6. M-step: Update (ϑ_a, ϱ_a^2, λ_a, μ_{βa}, s_{βa}^2)
##    This part corresponds with Proposition 3.
###########################################################

mstep_params <- function(dx, stats, params, N) {
  r      <- stats$r
  a_prob <- stats$a_prob
  Tn     <- length(dx)
  
  theta_new  <- numeric(N)
  rho2_new   <- numeric(N)
  lambda_new <- numeric(N)
  mu_b_new   <- numeric(N)
  sb2_new    <- numeric(N)
  
  for (a in 1:N) {
    w_a <- a_prob[a, ]            # γ_k(a)
    w0  <- w_a * (1 - r[a, ])     # 无跳权重
    w1  <- w_a * r[a, ]           # 有跳权重
    
    # λ_a = 有跳相对频率
    if (sum(w_a) > 0) {
      lambda_new[a] <- sum(w1) / sum(w_a)
    } else {
      lambda_new[a] <- params$lambda[a]
    }
    
    # ϑ_a (只用无跳部分的条件均值)
    if (sum(w0) > 0) {
      theta_new[a] <- sum(w0 * dx) / sum(w0)
    } else {
      theta_new[a] <- params$theta[a]
    }
    
    # μ_{βa} (有跳样本均值减去 ϑ_a)
    if (sum(w1) > 0) {
      mu_b_new[a] <- sum(w1 * dx) / sum(w1) - theta_new[a]
    } else {
      mu_b_new[a] <- params$mu_beta[a]
    }
    
    # ϱ_a^2 : 用无跳部分的二阶矩
    if (sum(w0) > 0) {
      rho2_new[a] <- sum(w0 * (dx - theta_new[a])^2) / sum(w0)
    } else {
      rho2_new[a] <- params$rho2[a]
    }
    rho2_new[a] <- max(rho2_new[a], 1e-8)
    
    # s_{βa}^2 : 用有跳部分的二阶矩，扣掉扩散方差
    if (sum(w1) > 0) {
      sb2_new[a] <- sum(w1 * (dx - theta_new[a] - mu_b_new[a])^2) / sum(w1) - rho2_new[a]
    } else {
      sb2_new[a] <- params$s_beta2[a]
    }
    sb2_new[a] <- max(sb2_new[a], 1e-8)
  }
  
  list(
    theta   = theta_new,
    rho2    = rho2_new,
    lambda  = lambda_new,
    mu_beta = mu_b_new,
    s_beta2 = sb2_new
  )
}

###########################################################
## 7. M-step: Update the ternary transition probabilities p[a,b,c]
##This part corresponds with Proposition 3.
###########################################################


mstep_transition <- function(xi_sum, N, smooth_eta = 1e-3) {
  M <- N * N
  p_new <- array(0, dim = c(N, N, N))
  
  for (prev in 1:N) {
    for (curr in 1:N) {
      j <- pair_index(prev, curr, N)
      num <- numeric(N)
      for (a in 1:N) {
        i <- pair_index(curr, a, N)
        num[a] <- xi_sum[i, j]
      }
      s <- sum(num)
      if (s <= 0 || !is.finite(s)) {
        p_new[, curr, prev] <- rep(1/N, N)
      } else {
        p_new[, curr, prev] <- (num + smooth_eta) / (s + N * smooth_eta)
      }
    }
  }
  p_new
}

###########################################################
## 8. Main EM routine
###########################################################

###########################################################
## Batch-processing EM with global sufficient statistics
##
##  - X: log-price path
##  - n_batch: split diff(X) evenly into several batches
##  - Returns: final parameters + per-batch parameter path (history)
###########################################################

em_fit_jump_mixture_rn_batch_global <- function(
    X, dt, N,
    p_init,
    mu_init, sigma_init, lambda_init, mu_beta_init, s_beta_init,
    n_batch,
    esscher_theta = esscher_theta,
    base_sd = 1,
    verbose = TRUE
) {
  dx <- diff(X)
  Tn <- length(dx)
  M  <- N * N
  
  idx_start <- floor((0:(n_batch-1)) * Tn / n_batch) + 1
  idx_end   <- floor((1:n_batch)   * Tn / n_batch)
  
  theta_init <- (mu_init - 0.5 * sigma_init^2) * dt
  rho2_init  <- (sigma_init^2) * dt
  
  params <- list(
    theta   = theta_init,
    rho2    = rho2_init,
    lambda  = lambda_init,
    mu_beta = mu_beta_init,
    s_beta2 = s_beta_init^2
  )
  p <- p_init
  
  # global stats (for parameter estimation)
  W      <- numeric(N)
  W0     <- numeric(N)
  W1     <- numeric(N)
  S0_dx  <- numeric(N)
  S0_dx2 <- numeric(N)
  S1_dx  <- numeric(N)
  S1_dx2 <- numeric(N)
  Xi_global <- matrix(0, M, M)
  
  init_logpi <- rep(-log(M), M)
  
  param_history <- vector("list", n_batch)
  
  # ====== 新增：保存每批的“滤波”状态边缘概率，用来做 state estimate ======
  a_prob_filter_list <- vector("list", n_batch)  # each: N x T_b
  
  for (b in 1:n_batch) {
    s_idx <- idx_start[b]
    e_idx <- idx_end[b]
    dx_b  <- dx[s_idx:e_idx]
    T_b   <- length(dx_b)
    
    if (verbose) {
      cat("==== Batch", b, "/", n_batch,
          "| t =", s_idx, ":", e_idx, "====\n")
    }
    
    A  <- build_A_from_p(p)
    fb <- fb_augmented_mixture_rn(
      dx_b, A, params, N,
      dt = dt, base_sd = base_sd,
      esscher_theta = esscher_theta,
      init_logpi = init_logpi
    )
    
    # ===== smoothing objects for parameter estimation =====
    gamma_b  <- fb$gamma
    xi_sum_b <- fb$xi_sum
    Xi_global <- Xi_global + xi_sum_b
    
    # ===== filtering marginal for state estimate (save it) =====
    a_prob_filter_list[[b]] <- fb$a_prob_filter  # N x T_b
    
    # parameter E-step weights (use smoothing gamma as before)
    stats_b  <- jump_posterior(dx_b, params, N, gamma_b)
    r_b      <- stats_b$r
    a_prob_b <- stats_b$a_prob   # this is smoothing-based
    
    # accumulate global sufficient stats
    dx2_b <- dx_b^2
    for (a in 1:N) {
      w_a_t <- a_prob_b[a, ]
      w0_t  <- w_a_t * (1 - r_b[a, ])
      w1_t  <- w_a_t * r_b[a, ]
      
      W[a]  <- W[a]  + sum(w_a_t)
      W0[a] <- W0[a] + sum(w0_t)
      W1[a] <- W1[a] + sum(w1_t)
      
      S0_dx[a]  <- S0_dx[a]  + sum(w0_t * dx_b)
      S0_dx2[a] <- S0_dx2[a] + sum(w0_t * dx2_b)
      S1_dx[a]  <- S1_dx[a]  + sum(w1_t * dx_b)
      S1_dx2[a] <- S1_dx2[a] + sum(w1_t * dx2_b)
    }
    
    # M-step (unchanged)
    theta_new  <- params$theta
    rho2_new   <- params$rho2
    lambda_new <- params$lambda
    mu_b_new   <- params$mu_beta
    sb2_new    <- params$s_beta2
    
    for (a in 1:N) {
      if (W[a]  > 0) lambda_new[a] <- W1[a] / W[a]
      if (W0[a] > 0) theta_new[a]  <- S0_dx[a] / W0[a]
      if (W1[a] > 0) mu_b_new[a]   <- S1_dx[a] / W1[a] - theta_new[a]
      
      if (W0[a] > 0) {
        num_rho <- S0_dx2[a] - 2 * theta_new[a] * S0_dx[a] +
          theta_new[a]^2 * W0[a]
        rho2_new[a] <- num_rho / W0[a]
      }
      rho2_new[a] <- max(rho2_new[a], 1e-8)
      
      if (W1[a] > 0) {
        num_sb <- S1_dx2[a] - 2 * (theta_new[a] + mu_b_new[a]) * S1_dx[a] +
          (theta_new[a] + mu_b_new[a])^2 * W1[a]
        sb2_new[a] <- num_sb / W1[a] - rho2_new[a]
      }
      sb2_new[a] <- max(sb2_new[a], 1e-8)
    }
    
    params <- list(
      theta   = theta_new,
      rho2    = rho2_new,
      lambda  = lambda_new,
      mu_beta = mu_b_new,
      s_beta2 = sb2_new
    )
    
    p <- mstep_transition(Xi_global, N)
    
    mu_hat_b    <- params$theta / dt + 0.5 * (params$rho2 / dt)
    sigma_hat_b <- sqrt(params$rho2 / dt)
    
    param_history[[b]] <- list(
      batch   = b,
      mu      = as.numeric(mu_hat_b),
      sigma   = as.numeric(sigma_hat_b),
      lambda  = as.numeric(params$lambda),
      mu_beta = as.numeric(params$mu_beta),
      s_beta  = as.numeric(sqrt(params$s_beta2)),
      p       = p
    )
    
    if (verbose) {
      cat(sprintf("  -> After batch %d: mu=%s | lambda=%s\n",
                  b,
                  paste(round(mu_hat_b, 4), collapse = ","),
                  paste(round(params$lambda, 4), collapse = ",")))
    }
    
    # next batch init: keep using smoothing end (as you already do)
    gamma_end <- gamma_b[, T_b]
    gamma_end <- pmax(gamma_end, 1e-300)
    gamma_end <- gamma_end / sum(gamma_end)
    init_logpi <- log(gamma_end)
  }
  
  mu_final    <- params$theta / dt + 0.5 * (params$rho2 / dt)
  sigma_final <- sqrt(params$rho2 / dt)
  
  # ===== 新增：拼出全样本“滤波”state estimate =====
  state_prob_filter <- do.call(cbind, a_prob_filter_list)  # N x Tn
  if (ncol(state_prob_filter) != Tn) {
    stop(sprintf("state_prob_filter length mismatch: got %d, expected %d",
                 ncol(state_prob_filter), Tn))
  }
  state_hat_filter <- apply(state_prob_filter, 2, which.max)
  
  list(
    mu      = as.numeric(mu_final),
    sigma   = as.numeric(sigma_final),
    lambda  = as.numeric(params$lambda),
    mu_beta = as.numeric(params$mu_beta),
    s_beta  = as.numeric(sqrt(params$s_beta2)),
    p       = p,
    history = param_history,
    
    # ===== 额外输出：state estimate 用 filtering =====
    state_prob_filter = state_prob_filter,  # P(y_t | X_{1:t})
    state_hat_filter  = state_hat_filter,
    
    # 若你还想看 smoothing 的边缘，也可以在 fb 里取 a_prob_smooth 另外存
    stats   = list(
      W = W, W0 = W0, W1 = W1,
      S0_dx = S0_dx, S0_dx2 = S0_dx2,
      S1_dx = S1_dx, S1_dx2 = S1_dx2,
      Xi_global = Xi_global
    ),
    esscher_theta = esscher_theta
  )
}

#================================= Import data =================================
library(dplyr)
library(tidyverse)
library(readxl)

data1 = read.csv("GCP.csv")
n_batch <- 96
date <- as.Date(data1$date)   # 如果本来就是Date可省略
Tn <- length(date) - 1            # dx 的长度
idx_start <- floor((0:(n_batch-1)) * Tn / n_batch) + 1
idx_end   <- floor((1:n_batch)   * Tn / n_batch)
# 每块最后一个 dx 对应的是 X 的 e_idx+1，所以日期取 e_idx+1
end_dates <- date[idx_end + 1]
end_dates

#================================ Set initial values ===================================
# ---- log-sum-exp helper ----
logsumexp2 <- function(a, b) {
  m <- pmax(a, b)
  m + log(exp(a - m) + exp(b - m))
}

# par = c(mu, log_sigma, logit_lambda, mu_beta, log_s_beta)
nll_benchmark <- function(par, X, dt = 1) {
  dx <- diff(X)
  
  mu        <- par[1]
  sigma     <- exp(par[2])                      # >0
  lambda    <- plogis(par[3])                   # in (0,1)
  mu_beta   <- par[4]
  s_beta    <- exp(par[5])                      # >0
  
  theta <- (mu - 0.5 * sigma^2) * dt            # ϑ
  rho2  <- (sigma^2) * dt                       # ρ^2
  sb2   <- (s_beta^2)                           # s_β^2
  
  # 两个分量的 log density
  log0 <- dnorm(dx, mean = theta,            sd = sqrt(rho2),       log = TRUE)
  log1 <- dnorm(dx, mean = theta + mu_beta,  sd = sqrt(rho2 + sb2), log = TRUE)
  
  # log mix density: log((1-lam)*exp(log0) + lam*exp(log1))
  ll <- logsumexp2(log(1 - lambda) + log0, log(lambda) + log1)
  
  # 负对数似然
  -sum(ll)
}

# ---- 给一个稳一点的起点（可以按你数据调整）----
make_start <- function(X, dt = 1) {
  dx <- diff(X)
  m  <- mean(dx)
  s  <- sd(dx)
  
  mu0      <- m / dt + 0.5 * (s^2 / dt)   # 粗略把 theta = (mu-0.5sigma^2)dt 反推
  sigma0   <- max(s / sqrt(dt), 1e-3)
  lambda0  <- 0.2
  mu_beta0 <- 0
  s_beta0  <- max(0.5 * sigma0, 1e-3)
  
  c(mu0,
    log(sigma0),
    qlogis(lambda0),
    mu_beta0,
    log(s_beta0))
}

# ---- 主调用：用 optim 拟合基准参数 ----
fit_benchmark_optim <- function(X, dt = 1) {
  par0 <- make_start(X, dt)
  
  opt <- optim(
    par = par0,
    fn  = nll_benchmark,
    X   = X,
    dt  = dt,
    method = "BFGS",
    control = list(maxit = 2000, reltol = 1e-10)
  )
  
  # 还原到原始参数
  mu_hat      <- opt$par[1]
  sigma_hat   <- exp(opt$par[2])
  lambda_hat  <- plogis(opt$par[3])
  mu_beta_hat <- opt$par[4]
  s_beta_hat  <- exp(opt$par[5])
  
  list(
    par = c(mu = mu_hat,
            sigma = sigma_hat,
            lambda = lambda_hat,
            mu_beta = mu_beta_hat,
            s_beta = s_beta_hat),
    value = opt$value,
    convergence = opt$convergence,
    message = opt$message
  )
}

X= data1$log_price
bench <- fit_benchmark_optim(X, dt = 1)
bench$par

#====================================N=1======================================
mu_init      <- c(bench$par[1],bench$par[1])
sigma_init   <- c(bench$par[2],bench$par[2])
lambda_init  <- c(bench$par[3],bench$par[3])
mu_beta_init <- c(bench$par[4],bench$par[4])
s_beta_init  <- c(bench$par[5],bench$par[5])


N=1
# 转移概率初值：真值附近稍微扰动一下
p_init <- array(0, dim = c(N, N, N))


#====================================run======================================
set.seed(2025)
dt=1
fit0  <- em_fit_jump_mixture_rn_batch_global(
  X = X, dt = dt, N = N,
  p_init = p_init,
  mu_init = mu_init, sigma_init = sigma_init,
  lambda_init = lambda_init,
  mu_beta_init = mu_beta_init,
  s_beta_init  = s_beta_init,
  n_batch = 96,
  esscher_theta = 0.,   # 可以改成 0.5, 1 看稳健性
  verbose = FALSE
)

# 假设对象叫 p，里面有 p$history
lst <- fit0$history          # list 长度 100，每个元素长度 7

# 把每个元素变成一行，再按行拼起来
df <- do.call(
  rbind,
  lapply(lst, function(x) {
    as.data.frame(t(unlist(x)))
  })
)

#=============================plot====================================================
#参数图
date = end_dates
state = df[,2]
df_a = data.frame(date,state)
a =  ggplot(data = df_a,aes(x=date,y=df_a[,2]))+
  geom_line(linetype = "solid",size=0.5,color = "black")+
  geom_point(aes(x=date,y=df_a[,2]),color = "black",size=1.2)+
  labs(x = NULL)+
  ylab(expression(mu))+
  theme_bw()  +
  theme(legend.position ="right",legend.title = element_blank(),legend.text = element_text(size = 8),
        legend.background = element_rect(fill = "transparent"),
        legend.box.background = element_rect(color = "#333333", linewidth = 0.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "transparent",colour = NA),
        axis.text.x = element_text(size = 12,angle = 30,vjust = 0.6),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 15),
        #plot.title = element_text(size=6,hjust=0.5)
        plot.title = element_blank())+
  scale_x_date(date_breaks="1 year",date_labels="%Y-%m",expand = c(0, 0))
a
#ggsave("D:/博士/论文1/审稿意见/图片/Figure 11.png",width =10,height=5,dpi = 500)

#2
date = as.Date(end_dates)
state = df[,4]
df_b = data.frame(date,state)
b = ggplot(data = df_b,aes(x=date,y=df_b[,2]))+
  geom_line(linetype = "solid",size=0.5,color = "black")+
  geom_point(aes(x=date,y=df_b[,2]),color = "black",size=1.2)+
  labs(x = NULL)+
  ylab(expression(sigma))+
  theme_bw()  +
  theme(legend.position ="right",legend.title = element_blank(),legend.text = element_text(size = 8),
        legend.background = element_rect(fill = "transparent"),
        legend.box.background = element_rect(color = "#333333", linewidth = 0.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "transparent",colour = NA),
        axis.text.x = element_text(size = 12,angle = 30,vjust = 0.6),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 15),
        #plot.title = element_text(size=6,hjust=0.5)
        plot.title = element_blank())+
  scale_x_date(date_breaks="1 year",date_labels="%Y-%m",expand = c(0, 0))
b
#ggsave("D:/博士/论文1/审稿意见/图片/Figure 7.png",width =10,height=5,dpi = 500)

#3
date = end_dates
state = df[,6]
df_c = data.frame(date,state)
c = ggplot(data = df_c,aes(x=date,y=df_c[,2]))+
  geom_line(linetype = "solid",size=0.5,color = "black")+
  geom_point(aes(x=date,y=df_c[,2]),color = "black",size=1.2)+
  labs(x = NULL)+
  ylab(expression(lambda))+
  theme_bw()  +
  theme(legend.position ="right",legend.title = element_blank(),legend.text = element_text(size = 8),
        legend.background = element_rect(fill = "transparent"),
        legend.box.background = element_rect(color = "#333333", linewidth = 0.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "transparent",colour = NA),
        axis.text.x = element_text(size = 12,angle = 30,vjust = 0.6),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 15),
        #plot.title = element_text(size=6,hjust=0.5)
        plot.title = element_blank())+
  scale_x_date(date_breaks="1 year",date_labels="%Y-%m",expand = c(0, 0))

#ggsave("D:/博士/论文1/审稿意见/图片/Figure 7.png",width =10,height=5,dpi = 500)
c

#4
date = end_dates
state = df[,8]
df_d = data.frame(date,state)
d = ggplot(data = df_d,aes(x=date,y=df_d[,2]))+
  geom_line(linetype = "solid",size=0.5,color = "black")+
  geom_point(aes(x=date,y=df_d[,2]),color = "black",size=1.2)+
  labs(x = NULL)+
  ylab(expression(mu[beta]))+
  theme_bw()  +
  theme(legend.position ="right",legend.title = element_blank(),legend.text = element_text(size = 8),
        legend.background = element_rect(fill = "transparent"),
        legend.box.background = element_rect(color = "#333333", linewidth = 0.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "transparent",colour = NA),
        axis.text.x = element_text(size = 12,angle = 30,vjust = 0.6),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 15),
        #plot.title = element_text(size=6,hjust=0.5)
        plot.title = element_blank())+
  scale_x_date(date_breaks="1 year",date_labels="%Y-%m",expand = c(0, 0))
d

#4
date = end_dates
state = df[,10]
df_e = data.frame(date,state)
e = ggplot(data = df_e,aes(x=date,y=df_e[,2]))+
  geom_line(linetype = "solid",size=0.5,color = "black")+
  geom_point(aes(x=date,y=df_e[,2]),color = "black",size=1.2)+
  labs(x = NULL)+
  ylab(expression(s[beta]))+
  theme_bw()  +
  theme(legend.position ="right",legend.title = element_blank(),legend.text = element_text(size = 8),
        legend.background = element_rect(fill = "transparent"),
        legend.box.background = element_rect(color = "#333333", linewidth = 0.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "transparent",colour = NA),
        axis.text.x = element_text(size = 12,angle = 30,vjust = 0.6),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 15),
        #plot.title = element_text(size=6,hjust=0.5)
        plot.title = element_blank())+
  scale_x_date(date_breaks="1 year",date_labels="%Y-%m",expand = c(0, 0))
e

library(ggpubr)
# Create a text grob
text= "China Carbon Index"
tgrob <- text_grob(text,size = 15)
# Draw the text
plot_0 <- as_ggplot(tgrob) + theme(plot.margin = margin(0,0,0,0, "cm"))

ggarrange(plot_0,a,b,c,d,e,
          ncol = 1,nrow = 6,heights = c(1, 5, 5,5,5,5,5), 
          labels = c("","A", "B","C","D","E"),label.y = 1.1)

#ggsave("GC_1_state_par_1.png",width =7,height=15,dpi = 500)
