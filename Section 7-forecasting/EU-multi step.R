###########################################################
## 0. Tools: extended state indexing (2nd-order HMM -> N^2-dim 1st-order HMM)
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
## 1. Build extended transition matrix A from p[a,b,c] (column-stochastic)
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
  
  # Clip weights to (0,1)
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
##    Here g uses the standard normal N(0, 1) density
###########################################################

emit_logrn_mixture <- function(dx_t, params, N,
                               dt = 1, base_sd = 1,
                               esscher_theta = 0) {
  M <- N * N
  curr_state <- ((1:M - 1L) %% N) + 1L
  
  # Mixture density under Q: f_a(Δx)
  f_Q <- emit_mix_density_Q(dx_t, params, N, dt = dt)
  log_f_Q <- log(pmax(f_Q, 1e-300))
  
  # Simple reference density under measure P: g(Δx) (here take N(0, base_sd^2))
  log_g <- dnorm(dx_t, mean = 0, sd = base_sd, log = TRUE)
  
  # Local RN without Esscher: log φ_k(a) = log f_a - log g
  log_phi <- log_f_Q - as.numeric(log_g)
  
  # ====== Esscher tilting: φ_k^(θ)(a) = e^{θΔx}/M_a(θ) * φ_k(a) ======
  if (esscher_theta != 0) {
    th  <- esscher_theta
    
    theta_a <- params$theta[curr_state]              # ϑ_a * dt
    rho2_a  <- pmax(params$rho2[curr_state], 1e-8)   # ϱ_a^2 * dt
    lambda_a<- params$lambda[curr_state]
    mu_b_a  <- params$mu_beta[curr_state]
    sb2_a   <- pmax(params$s_beta2[curr_state], 1e-8)
    
    # 0/1 jump weights w0, w1 for the current time step (note λ_a dt)
    w1 <- pmin(pmax(lambda_a * dt, 0), 0.999)
    w0 <- pmax(1 - w1, 1e-6)
    
    # log M_a(θ) = log( w0 e^{θϑ_a + 0.5 θ^2 ϱ_a^2}
    #                 + w1 e^{θ(ϑ_a+μ_βa)+0.5 θ^2(ϱ_a^2+s_βa^2)} )
    log_part0 <- log(w0) + th * theta_a + 0.5 * th^2 * rho2_a
    log_part1 <- log(w1) + th * (theta_a + mu_b_a) +
      0.5 * th^2 * (rho2_a + sb2_a)
    m  <- pmax(log_part0, log_part1)
    logM <- m + log(exp(log_part0 - m) + exp(log_part1 - m))
    
    # Add Esscher factor: log φ^(θ) = log φ + θΔx - logM
    log_phi <- log_phi + th * dx_t - logM
  }
  # =====================================================
  
  list(
    log_phi = log_phi,          # length M: log φ_k^(θ)(i)
    log_g   = as.numeric(log_g) # scalar: log g(Δx_k)
  )
}


###########################################################
## 4. Forward-backward algorithm (RN-derivative version)
##
##    Corresponds to Proposition 2: v_{k+1} = A D_{k+1} v_k,
##    implemented in log space for numerical stability.
###########################################################

fb_augmented_mixture_rn <- function(dx, A, params, N,
                                    dt = 1, base_sd = 1,
                                    esscher_theta = 0,
                                    init_logpi = NULL) {
  Tn <- length(dx)
  M  <- N * N
  logA <- log(pmax(A, 1e-300))
  
  logB   <- matrix(NA_real_, M, Tn)  # each column is log φ_k^(θ)(i)
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
  
  ## Initial step: if init_logpi is not provided, use a uniform prior
  if (is.null(init_logpi)) {
    log_alpha[, 1] <- -log(M) + logB[, 1]
  } else {
    log_alpha[, 1] <- init_logpi + logB[, 1]
  }
  m1 <- max(log_alpha[, 1])
  s1 <- sum(exp(log_alpha[, 1] - m1))
  cvec[1] <- m1 + log(s1)
  log_alpha[, 1] <- log_alpha[, 1] - cvec[1]
  
  for (t in 2:Tn) {
    tmp <- matrix(log_alpha[, t-1], nrow = M, ncol = M, byrow = TRUE) + logA
    logsum <- apply(tmp, 1, function(x) {
      m <- max(x); m + log(sum(exp(x - m)))
    })
    log_alpha[, t] <- logB[, t] + logsum
    mt <- max(log_alpha[, t])
    st <- sum(exp(log_alpha[, t] - mt))
    cvec[t] <- mt + log(st)
    log_alpha[, t] <- log_alpha[, t] - cvec[t]
  }
  
  for (t in (Tn-1):1) {
    add <- logB[, t+1] + log_beta[, t+1]
    tmp <- matrix(add, nrow = M, ncol = M, byrow = TRUE) + t(logA)
    log_beta[, t] <- apply(tmp, 1, function(x) {
      m <- max(x); m + log(sum(exp(x - m)))
    }) - cvec[t+1]
  }
  
  log_gamma <- log_alpha + log_beta
  gamma <- matrix(NA_real_, M, Tn)
  for (t in 1:Tn) {
    mt <- max(log_gamma[, t])
    st <- sum(exp(log_gamma[, t] - mt))
    log_gamma[, t] <- log_gamma[, t] - (mt + log(st))
    gamma[, t] <- exp(log_gamma[, t])
  }
  
  xi_sum <- matrix(0, M, M)
  for (t in 2:Tn) {
    log_xi <- matrix(NA_real_, M, M)
    for (j in 1:M) {
      log_xi[, j] <- log_alpha[j, t-1] + logA[, j] + logB[, t] + log_beta[, t]
    }
    mt <- max(log_xi); st <- sum(exp(log_xi - mt))
    log_xi <- log_xi - (mt + log(st))
    xi <- exp(log_xi)
    xi_sum <- xi_sum + xi
  }
  
  loglik_P     <- sum(cvec)
  loglik_Qtheta<- loglik_P + sum(log_gv)
  
  list(
    gamma = gamma,
    xi_sum = xi_sum,
    loglik_P = loglik_P,
    loglik_Qtheta = loglik_Qtheta
  )
}



###########################################################
## 5. Jump posterior r[a,t] and state marginal probabilities γ_k(a)
###########################################################

jump_posterior <- function(dx, params, N, gamma) {
  Tn <- length(dx)
  M  <- N * N
  
  # Each extended state corresponds to the current regime a
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
  lambda <- pmin(pmax(params$lambda, 1e-8), 1 - 1e-6)
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
    
    log_lambda      <- log(lambda[a])
    log_one_minus   <- log1p(-lambda[a])  # more stable than log(1 - lambda)
    
    log_num <- log_lambda + logphi1
    
    # Stable computation of log_den
    z0 <- log_one_minus + logphi0
    z1 <- log_num
    m  <- pmax(z0, z1)
    log_den <- m + log(exp(z0 - m) + exp(z1 - m))
    
    r[a, ] <- exp(log_num - log_den)
    
    # Prevent r from going outside [0,1] or becoming Inf/NaN due to extreme values
    r[a, ][!is.finite(r[a, ])] <- 0
    r[a, ] <- pmin(pmax(r[a, ], 0), 1)
  }
  
  
  list(r = r, a_prob = a_prob)
}

###########################################################
## 6. M-step: update (ϑ_a, ϱ_a^2, λ_a, μ_{βa}, s_{βa}^2)
##    This corresponds one-to-one with your notes 4.1–4.3.
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
    w0  <- w_a * (1 - r[a, ])     # no-jump weights
    w1  <- w_a * r[a, ]           # jump weights
    
    # λ_a = relative frequency of jumps
    if (sum(w_a) > 0) {
      lambda_new[a] <- sum(w1) / sum(w_a)
    } else {
      lambda_new[a] <- params$lambda[a]
    }
    
    # ϑ_a (conditional mean using only the no-jump part)
    if (sum(w0) > 0) {
      theta_new[a] <- sum(w0 * dx) / sum(w0)
    } else {
      theta_new[a] <- params$theta[a]
    }
    
    # μ_{βa} (mean of jump samples minus ϑ_a)
    if (sum(w1) > 0) {
      mu_b_new[a] <- sum(w1 * dx) / sum(w1) - theta_new[a]
    } else {
      mu_b_new[a] <- params$mu_beta[a]
    }
    
    # ϱ_a^2 : second moment using the no-jump part
    if (sum(w0) > 0) {
      rho2_new[a] <- sum(w0 * (dx - theta_new[a])^2) / sum(w0)
    } else {
      rho2_new[a] <- params$rho2[a]
    }
    rho2_new[a] <- max(rho2_new[a], 1e-8)
    
    # s_{βa}^2 : second moment using the jump part, subtracting diffusion variance
    if (sum(w1) > 0) {
      sb2_new[a] <- sum(w1 * (dx - theta_new[a] - mu_b_new[a])^2) / sum(w1) - rho2_new[a]
    } else {
      sb2_new[a] <- params$s_beta2[a]
    }
    sb2_new[a] <- max(sb2_new[a], 1e-8)
  }
  lambda_new <- pmin(pmax(lambda_new, 1e-8), 1 - 1e-6)
  
  
  list(
    theta   = theta_new,
    rho2    = rho2_new,
    lambda  = lambda_new,
    mu_beta = mu_b_new,
    s_beta2 = sb2_new
  )
}

###########################################################
## 7. M-step: update ternary transition probabilities p[a,b,c]
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

forward_filter_mixture_rn <- function(dx, A, params, N,
                                      dt = 1, base_sd = 1,
                                      esscher_theta = 0,
                                      init_logpi = NULL) {
  Tn <- length(dx)
  M  <- N * N
  logA <- log(pmax(A, 1e-300))
  
  # log φ_k(i) of observation increments under each extended state
  logB   <- matrix(NA_real_, M, Tn)
  log_gv <- numeric(Tn)
  for (t in 1:Tn) {
    tmp <- emit_logrn_mixture(dx[t], params, N,
                              dt = dt, base_sd = base_sd,
                              esscher_theta = esscher_theta)
    logB[, t] <- tmp$log_phi
    log_gv[t] <- tmp$log_g
  }
  
  # forward recursion
  log_alpha <- matrix(-Inf, M, Tn)
  cvec      <- numeric(Tn)
  
  if (is.null(init_logpi)) {
    log_alpha[, 1] <- -log(M) + logB[, 1]
  } else {
    log_alpha[, 1] <- init_logpi + logB[, 1]
  }
  m1 <- max(log_alpha[, 1])
  s1 <- sum(exp(log_alpha[, 1] - m1))
  cvec[1] <- m1 + log(s1)
  log_alpha[, 1] <- log_alpha[, 1] - cvec[1]
  
  for (t in 2:Tn) {
    tmp <- matrix(log_alpha[, t-1], nrow = M, ncol = M, byrow = TRUE) + logA
    logsum <- apply(tmp, 1, function(x) {
      m <- max(x); m + log(sum(exp(x - m)))
    })
    log_alpha[, t] <- logB[, t] + logsum
    mt <- max(log_alpha[, t])
    st <- sum(exp(log_alpha[, t] - mt))
    cvec[t] <- mt + log(st)
    log_alpha[, t] <- log_alpha[, t] - cvec[t]
  }
  
  # Filtering posterior: gamma_filt[,t] = P(z_t | x_1,...,x_t)
  gamma_filt <- exp(log_alpha)    # each column already normalized
  
  list(
    gamma_filt = gamma_filt,      # M x Tn, extended-state filtering posterior
    loglik_P   = sum(cvec)
  )
}

###########################################################
## 8. EM main function (RN-derivative version)
###########################################################

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
    
    #  if (verbose) {
    #   cat(sprintf("Iter %3d | loglik_P=%.6f | theta_E=%.3f | lambda=%s\n",
    #              it, fb$loglik_P, esscher_theta,
    #             paste(round(params$lambda, 4), collapse=",")))
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
mu_init      <- c(0.00031,0.000445,0.000692)
sigma_init   <- c(0.0155 , 0.02562,0.03456)
lambda_init  <- c(0.0132, 0.02356,0.03556)
mu_beta_init <- c(0.002, 0.003642,0.004)
s_beta_init  <- c(0.6435, 0.7345, 0.8342)

# 转移概率初值：真值附近稍微扰动一下
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

