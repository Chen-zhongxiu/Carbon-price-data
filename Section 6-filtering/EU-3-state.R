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
##    Here g is the standard normal N(0, 1) density
###########################################################

emit_logrn_mixture <- function(dx_t, params, N,
                               dt = 1, base_sd = 1,
                               esscher_theta = esscher_theta) {
  M <- N * N
  curr_state <- ((1:M - 1L) %% N) + 1L
  
  # Mixture density f_a(Δx) under Q
  f_Q <- emit_mix_density_Q(dx_t, params, N, dt = dt)
  log_f_Q <- log(pmax(f_Q, 1e-300))
  
  # Simple reference density g(Δx) under P (here N(0, base_sd^2))
  log_g <- dnorm(dx_t, mean = 0, sd = base_sd, log = TRUE)
  
  # Local RN without Esscher: log φ_k(a) = log f_a - log g
  log_phi <- log_f_Q - as.numeric(log_g)
  
  # ====== Esscher tilt: φ_k^(θ)(a) = e^{θΔx}/M_a(θ) * φ_k(a) ======
  if (esscher_theta != 0) {
    th  <- esscher_theta
    
    theta_a <- params$theta[curr_state]              # ϑ_a * dt
    rho2_a  <- pmax(params$rho2[curr_state], 1e-8)   # ϱ_a^2 * dt
    lambda_a<- params$lambda[curr_state]
    mu_b_a  <- params$mu_beta[curr_state]
    sb2_a   <- pmax(params$s_beta2[curr_state], 1e-8)
    
    # 0/1 jump weights at this step (note λ_a dt)
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
## 4. Forward-backward algorithm (with RN derivative)
##
##    Corresponds to Proposition 2: v_{k+1} = A D_{k+1} v_k,
##    implemented in log space for numerical stability.
###########################################################

###########################################################
## 4*. Forward-backward algorithm (with RN derivative + custom initialization)
##
##    init_logpi: length-M vector of log initial probabilities
##    - NULL reduces to a uniform prior (-log M)
##    - In batch mode, use gamma[,T] from the previous batch as init
###########################################################

fb_augmented_mixture_rn <- function(dx, A, params, N,
                                    dt = 1, base_sd = 1,
                                    esscher_theta = 0,
                                    init_logpi = NULL) {
  Tn <- length(dx)
  M  <- N * N
  
  # ===== Guard: empty batch =====
  if (Tn == 0) {
    return(list(
      alpha = matrix(0, M, 0),
      gamma = matrix(0, M, 0),
      xi_sum = matrix(0, M, M),
      a_prob_filter = matrix(0, N, 0),
      a_prob_smooth = matrix(0, N, 0),
      loglik_P = 0,
      loglik_Qtheta = 0
    ))
  }
  
  logA <- log(pmax(A, 1e-300))
  
  logB   <- matrix(NA_real_, M, Tn)
  log_gv <- numeric(Tn)
  for (t in seq_len(Tn)) {
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
  m1 <- max(log_alpha[, 1])
  s1 <- sum(exp(log_alpha[, 1] - m1))
  cvec[1] <- m1 + log(s1)
  log_alpha[, 1] <- log_alpha[, 1] - cvec[1]
  
  # forward
  if (Tn >= 2) {
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
  }
  
  # backward
  if (Tn >= 2) {
    for (t in (Tn-1):1) {
      add <- logB[, t+1] + log_beta[, t+1]
      tmp <- matrix(add, nrow = M, ncol = M, byrow = TRUE) + t(logA)
      log_beta[, t] <- apply(tmp, 1, function(x) {
        m <- max(x); m + log(sum(exp(x - m)))
      }) - cvec[t+1]
    }
  }
  
  # gamma
  log_gamma <- log_alpha + log_beta
  gamma <- matrix(NA_real_, M, Tn)
  for (t in seq_len(Tn)) {
    mt <- max(log_gamma[, t])
    st <- sum(exp(log_gamma[, t] - mt))
    log_gamma[, t] <- log_gamma[, t] - (mt + log(st))
    gamma[, t] <- exp(log_gamma[, t])
  }
  
  # alpha (filtering)
  alpha <- exp(log_alpha)
  
  # xi_sum (smoothing-based)
  xi_sum <- matrix(0, M, M)
  if (Tn >= 2) {
    for (t in 2:Tn) {
      log_xi <- matrix(NA_real_, M, M)
      for (j in seq_len(M)) {
        log_xi[, j] <- log_alpha[j, t-1] + logA[, j] + logB[, t] + log_beta[, t]
      }
      mt <- max(log_xi)
      st <- sum(exp(log_xi - mt))
      log_xi <- log_xi - (mt + log(st))
      xi_sum <- xi_sum + exp(log_xi)
    }
  }
  
  loglik_P      <- sum(cvec)
  loglik_Qtheta <- loglik_P + sum(log_gv)
  
  # marginalize to y_t
  curr_state <- ((seq_len(M) - 1L) %% N) + 1L
  
  a_prob_filter <- matrix(0, N, Tn)
  a_prob_smooth <- matrix(0, N, Tn)
  for (t in seq_len(Tn)) {
    for (j in seq_len(M)) {
      a <- curr_state[j]
      a_prob_filter[a, t] <- a_prob_filter[a, t] + alpha[j, t]
      a_prob_smooth[a, t] <- a_prob_smooth[a, t] + gamma[j, t]
    }
  }
  
  list(
    alpha = alpha,
    gamma = gamma,
    xi_sum = xi_sum,
    a_prob_filter = a_prob_filter,
    a_prob_smooth = a_prob_smooth,
    loglik_P = loglik_P,
    loglik_Qtheta = loglik_Qtheta
  )
}





###########################################################
## 5. Jump posterior r[a,t] and state marginal probability γ_k(a)
###########################################################

jump_posterior <- function(dx, params, N, gamma,
                           eps_lam = 1e-6) {
  Tn <- length(dx)
  M  <- N * N
  
  if (Tn == 0) {
    return(list(
      r = matrix(0, N, 0),
      a_prob = matrix(0, N, 0)
    ))
  }
  
  curr_state <- ((seq_len(M) - 1L) %% N) + 1L
  
  # a_prob from gamma (smoothing)
  a_prob <- matrix(0, N, Tn)
  for (t in seq_len(Tn)) {
    for (j in seq_len(M)) {
      a <- curr_state[j]
      a_prob[a, t] <- a_prob[a, t] + gamma[j, t]
    }
  }
  # Clean a_prob
  a_prob[!is.finite(a_prob)] <- 0
  a_prob <- pmax(a_prob, 0)
  
  theta  <- params$theta
  rho2   <- pmax(params$rho2, 1e-8)
  lambda <- params$lambda
  lambda <- pmin(pmax(lambda, eps_lam), 1 - eps_lam)  # clamp
  mu_b   <- params$mu_beta
  sb2    <- pmax(params$s_beta2, 1e-8)
  
  r <- matrix(0, N, Tn)
  logN <- function(x, m, v) -0.5 * (log(2*pi) + log(v) + (x - m)^2 / v)
  
  for (a in seq_len(N)) {
    m0 <- theta[a];           v0 <- rho2[a]
    m1 <- theta[a] + mu_b[a]; v1 <- rho2[a] + sb2[a]
    
    logphi0 <- logN(dx, m0, v0)
    logphi1 <- logN(dx, m1, v1)
    
    A0 <- log(1 - lambda[a]) + logphi0
    A1 <- log(lambda[a])     + logphi1
    
    log_den <- logsumexp2(A0, A1)
    r[a, ]  <- exp(A1 - log_den)
  }
  
  # Clean r
  r[!is.finite(r)] <- 0
  r <- pmin(pmax(r, 0), 1)
  
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
    
    # λ_a = relative jump frequency
    if (sum(w_a) > 0) {
      lambda_new[a] <- sum(w1) / sum(w_a)
    } else {
      lambda_new[a] <- params$lambda[a]
    }
    
    # ϑ_a (conditional mean using no-jump part only)
    if (sum(w0) > 0) {
      theta_new[a] <- sum(w0 * dx) / sum(w0)
    } else {
      theta_new[a] <- params$theta[a]
    }
    
    # μ_{βa} (jump-sample mean minus ϑ_a)
    if (sum(w1) > 0) {
      mu_b_new[a] <- sum(w1 * dx) / sum(w1) - theta_new[a]
    } else {
      mu_b_new[a] <- params$mu_beta[a]
    }
    
    # ϱ_a^2: second moment from no-jump part
    if (sum(w0) > 0) {
      rho2_new[a] <- sum(w0 * (dx - theta_new[a])^2) / sum(w0)
    } else {
      rho2_new[a] <- params$rho2[a]
    }
    rho2_new[a] <- max(rho2_new[a], 1e-8)
    
    # s_{βa}^2: second moment from jump part, net of diffusion variance
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

###########################################################
## 8. EM main function (with RN derivative)
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
    esscher_theta = 0,
    base_sd = 1,
    verbose = TRUE
) {
  dx <- diff(X)
  Tn <- length(dx)
  M  <- N * N
  
  if (Tn <= 0) stop("diff(X) has length 0: need at least 2 observations in X.")
  
  # ===== Guard: n_batch should not exceed data length (avoid empty batches) =====
  n_batch <- min(n_batch, Tn)
  
  idx_start <- floor((0:(n_batch-1)) * Tn / n_batch) + 1
  idx_end   <- floor((1:n_batch)     * Tn / n_batch)
  
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
  
  # global stats
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
  a_prob_filter_list <- vector("list", n_batch)  # for state estimate (filtering)
  
  # ===== EU defensive hyperparameters =====
  eps_lam <- 1e-6
  var_lb  <- 1e-5
  # =========================
  
  for (b in seq_len(n_batch)) {
    s_idx <- idx_start[b]
    e_idx <- idx_end[b]
    if (s_idx > e_idx) next  # skip empty batch
    
    dx_b <- dx[s_idx:e_idx]
    T_b  <- length(dx_b)
    if (T_b == 0) next
    
    if (verbose) {
      cat("==== Batch", b, "/", n_batch,
          "| t =", s_idx, ":", e_idx, "====\n")
    }
    
    A  <- build_A_from_p(p)
    
    # ===== smoothing FB: for parameter estimation =====
    fb <- fb_augmented_mixture_rn(
      dx_b, A, params, N,
      dt = dt, base_sd = base_sd,
      esscher_theta = esscher_theta,
      init_logpi = init_logpi
    )
    
    gamma_b  <- fb$gamma
    xi_sum_b <- fb$xi_sum
    Xi_global <- Xi_global + xi_sum_b
    
    # ===== filtering marginal: only for state estimate (extra output) =====
    a_prob_filter_b <- fb$a_prob_filter
    a_prob_filter_b[!is.finite(a_prob_filter_b)] <- 0
    a_prob_filter_b <- pmax(a_prob_filter_b, 0)
    a_prob_filter_list[[b]] <- a_prob_filter_b
    
    # ===== jump posterior: use smoothing gamma to estimate parameters =====
    stats_b <- jump_posterior(dx_b, params, N, gamma_b, eps_lam = eps_lam)
    r_b      <- stats_b$r
    a_prob_b <- stats_b$a_prob  # smoothing-based
    
    # (1) clean r / a_prob (double safety)
    r_b[!is.finite(r_b)] <- 0
    r_b <- pmin(pmax(r_b, 0), 1)
    a_prob_b[!is.finite(a_prob_b)] <- 0
    a_prob_b <- pmax(a_prob_b, 0)
    
    # accumulate sufficient stats
    dx2_b <- dx_b^2
    for (a in seq_len(N)) {
      w_a_t <- a_prob_b[a, ]
      r_t   <- r_b[a, ]
      
      w_a_t[!is.finite(w_a_t)] <- 0
      r_t[!is.finite(r_t)]     <- 0
      r_t <- pmin(pmax(r_t, 0), 1)
      
      w0_t <- w_a_t * (1 - r_t)
      w1_t <- w_a_t * r_t
      
      # (2) na.rm=TRUE
      W[a]  <- W[a]  + sum(w_a_t, na.rm = TRUE)
      W0[a] <- W0[a] + sum(w0_t,  na.rm = TRUE)
      W1[a] <- W1[a] + sum(w1_t,  na.rm = TRUE)
      
      S0_dx[a]  <- S0_dx[a]  + sum(w0_t * dx_b,  na.rm = TRUE)
      S0_dx2[a] <- S0_dx2[a] + sum(w0_t * dx2_b, na.rm = TRUE)
      S1_dx[a]  <- S1_dx[a]  + sum(w1_t * dx_b,  na.rm = TRUE)
      S1_dx2[a] <- S1_dx2[a] + sum(w1_t * dx2_b, na.rm = TRUE)
    }
    
    # M-step with defenses
    theta_new  <- params$theta
    rho2_new   <- params$rho2
    lambda_new <- params$lambda
    mu_b_new   <- params$mu_beta
    sb2_new    <- params$s_beta2
    
    for (a in seq_len(N)) {
      # (3) defense: is.finite + fallback + clamp lambda
      if (is.finite(W[a]) && W[a] > 0 && is.finite(W1[a])) {
        lambda_new[a] <- W1[a] / W[a]
        lambda_new[a] <- pmin(pmax(lambda_new[a], eps_lam), 1 - eps_lam)
      } else {
        lambda_new[a] <- pmin(pmax(params$lambda[a], eps_lam), 1 - eps_lam)
      }
      
      if (is.finite(W0[a]) && W0[a] > 0 && is.finite(S0_dx[a])) {
        theta_new[a] <- S0_dx[a] / W0[a]
      } else {
        theta_new[a] <- params$theta[a]
      }
      
      if (is.finite(W1[a]) && W1[a] > 0 && is.finite(S1_dx[a])) {
        mu_b_new[a] <- S1_dx[a] / W1[a] - theta_new[a]
      } else {
        mu_b_new[a] <- params$mu_beta[a]
      }
      
      if (is.finite(W0[a]) && W0[a] > 0 &&
          is.finite(S0_dx2[a]) && is.finite(S0_dx[a])) {
        num_rho <- S0_dx2[a] - 2 * theta_new[a] * S0_dx[a] +
          theta_new[a]^2 * W0[a]
        rho2_new[a] <- num_rho / W0[a]
      } else {
        rho2_new[a] <- params$rho2[a]
      }
      rho2_new[a] <- max(rho2_new[a], var_lb)
      
      if (is.finite(W1[a]) && W1[a] > 0 &&
          is.finite(S1_dx2[a]) && is.finite(S1_dx[a])) {
        num_sb <- S1_dx2[a] - 2 * (theta_new[a] + mu_b_new[a]) * S1_dx[a] +
          (theta_new[a] + mu_b_new[a])^2 * W1[a]
        sb2_new[a] <- num_sb / W1[a] - rho2_new[a]
      } else {
        sb2_new[a] <- params$s_beta2[a]
      }
      sb2_new[a] <- max(sb2_new[a], var_lb)
    }
    
    params <- list(
      theta   = theta_new,
      rho2    = rho2_new,
      lambda  = lambda_new,
      mu_beta = mu_b_new,
      s_beta2 = sb2_new
    )
    
    # transition update (global Xi)
    p <- mstep_transition(Xi_global, N)
    
    # save path
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
    
    # next-batch init: use filtering end distribution (EU setting)
    alpha_end <- fb$alpha[, T_b]
    alpha_end[!is.finite(alpha_end)] <- 0
    alpha_end <- pmax(alpha_end, 1e-300)
    alpha_end <- alpha_end / sum(alpha_end)
    init_logpi <- log(alpha_end)
  }
  
  mu_final    <- params$theta / dt + 0.5 * (params$rho2 / dt)
  sigma_final <- sqrt(params$rho2 / dt)
  
  # concatenate filtering state probabilities
  state_prob_filter <- do.call(cbind, a_prob_filter_list)
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
    
    # state estimate: filtering only
    state_prob_filter = state_prob_filter,
    state_hat_filter  = state_hat_filter,
    
    stats = list(
      W = W, W0 = W0, W1 = W1,
      S0_dx = S0_dx, S0_dx2 = S0_dx2,
      S1_dx = S1_dx, S1_dx2 = S1_dx2,
      Xi_global = Xi_global
    ),
    esscher_theta = esscher_theta
  )
}


#=================================Import data=================================
library(dplyr)
library(readxl)
#===========================EU data==========================================
data1 = read.csv("EUA.csv")
n_batch <- 243
date <- as.Date(data1$date)   # omit if already Date
Tn <- length(date) - 1            # length of dx
idx_start <- floor((0:(n_batch-1)) * Tn / n_batch) + 1
idx_end   <- floor((1:n_batch)   * Tn / n_batch)
# The last dx in each block corresponds to X at e_idx+1, so take date at e_idx+1
end_dates <- date[idx_end + 1]
end_dates

#================================Determine initial values===================================
# ---- log-sum-exp helper ----
logsumexp2 <- function(a, b) {
  m <- pmax(a, b)
  m + log(exp(a - m) + exp(b - m))
}

# ---- Eq.(6.1): full-sample negative log-likelihood (single-state) ----
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
  
  # log densities of two components
  log0 <- dnorm(dx, mean = theta,            sd = sqrt(rho2),       log = TRUE)
  log1 <- dnorm(dx, mean = theta + mu_beta,  sd = sqrt(rho2 + sb2), log = TRUE)
  
  # log mixture density: log((1-lam)*exp(log0) + lam*exp(log1))
  ll <- logsumexp2(log(1 - lambda) + log0, log(lambda) + log1)
  
  # negative log-likelihood
  -sum(ll)
}

# ---- A more stable starting point (tune for your data if needed) ----
make_start <- function(X, dt = 1) {
  dx <- diff(X)
  m  <- mean(dx)
  s  <- sd(dx)
  
  mu0      <- m / dt + 0.5 * (s^2 / dt)   # invert theta = (mu-0.5sigma^2)dt roughly
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

# ---- Main call: fit benchmark parameters via optim ----
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
  
  # back-transform to original parameters
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



mu_init      <- c(0.00031,0.000445,0.000692)
sigma_init   <- c(0.0155 , 0.02562,0.03456)
lambda_init  <- c(0.0132, 0.02356,0.03556)
mu_beta_init <- c(0.002, 0.003642,0.004)
s_beta_init  <- c(0.6435, 0.7345, 0.8342)


N=3
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


apply(p_init, c(2,3), sum)


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
  n_batch = 243,
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
state = df[,2:4]
df_a = data.frame(date,state)
a =  ggplot(data = df_a,aes(x=date,y=df_a[,2]))+
  geom_line(linetype = "solid",size=0.5,aes(color = "State 1"))+
  geom_point(aes(x=date,y=df_a[,2],color = "State 1"),size=1.2)+
  geom_line(data=df_a,aes(x=date, y=df_a[,3],color = "State 2"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_a[,3],color = "State 2"),size=1.2,shape=17)+
  geom_line(data=df_a,aes(x=date, y=df_a[,4],color = "State 3"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_a[,4],color = "State 3"),size=1.2,shape=15)+
  scale_color_manual(values = c("State 1" = "black", "State 2" = "grey","State 3"="red"))+
  labs(x = NULL)+
  ylab(expression(mu))+
  theme_bw()  +
  theme(legend.position ="none",legend.title = element_blank(),legend.text = element_text(size = 8),
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
date = end_dates
state = df[,5:7]
df_b = data.frame(date,state)
b =  ggplot(data = df_b,aes(x=date,y=df_b[,2]))+
  geom_line(linetype = "solid",size=0.5,aes(color = "State 1"))+
  geom_point(aes(x=date,y=df_b[,2],color = "State 1"),size=1.2)+
  geom_line(data=df_b,aes(x=date, y=df_b[,3],color = "State 2"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_b[,3],color = "State 2"),size=1.2,shape=17)+
  geom_line(data=df_b,aes(x=date, y=df_b[,4],color = "State 3"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_b[,4],color = "State 3"),size=1.2,shape=15)+
  scale_color_manual(values = c("State 1" = "black", "State 2" = "grey","State 3"="red"))+
  labs(x = NULL)+
  ylab(expression(sigma))+
  theme_bw()  +
  theme(legend.position = "none",legend.title = element_blank(),legend.text = element_text(size = 8),
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
state = df[,8:10]
df_c = data.frame(date,state)
c =  ggplot(data = df_c,aes(x=date,y=df_c[,2]))+
  geom_line(linetype = "solid",size=0.5,aes(color = "State 1"))+
  geom_point(aes(x=date,y=df_c[,2],color = "State 1"),size=1.2)+
  geom_line(data=df_c,aes(x=date, y=df_c[,3],color = "State 2"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_c[,3],color = "State 2"),size=1.2,shape=17)+
  geom_line(data=df_c,aes(x=date, y=df_c[,4],color = "State 3"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_c[,4],color = "State 3"),size=1.2,shape=15)+
  scale_color_manual(values = c("State 1" = "black", "State 2" = "grey","State 3"="red"))+
  labs(x = NULL)+
  ylab(expression(lambda))+
  theme_bw()  +
  theme(legend.position = "none",legend.title = element_blank(),legend.text = element_text(size = 8),
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
state = df[,11:13]
df_d = data.frame(date,state)
d =  ggplot(data = df_d,aes(x=date,y=df_d[,2]))+
  geom_line(linetype = "solid",size=0.5,aes(color = "State 1"))+
  geom_point(aes(x=date,y=df_d[,2],color = "State 1"),size=1.2)+
  geom_line(data=df_d,aes(x=date, y=df_d[,3],color = "State 2"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_d[,3],color = "State 2"),size=1.2,shape=17)+
  geom_line(data=df_d,aes(x=date, y=df_d[,4],color = "State 3"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_d[,4],color = "State 3"),size=1.2,shape=15)+
  scale_color_manual(values = c("State 1" = "black", "State 2" = "grey","State 3"="red"))+
  labs(x = NULL)+
  labs(y = expression(mu[beta]))+
  theme_bw()  +
  theme(legend.position = "none",legend.title = element_blank(),legend.text = element_text(size = 8),
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
state = df[,14:16]
df_e = data.frame(date,state)
e =  ggplot(data = df_e,aes(x=date,y=df_e[,2]))+
  geom_line(linetype = "solid",size=0.5,aes(color = "State 1"))+
  geom_point(aes(x=date,y=df_e[,2],color = "State 1"),size=1.2)+
  geom_line(data=df_e,aes(x=date, y=df_e[,3],color = "State 2"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_e[,3],color = "State 2"),size=1.2,shape=17)+
  geom_line(data=df_e,aes(x=date, y=df_e[,4],color = "State 3"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_e[,4],color = "State 3"),size=1.2,shape=15)+
  scale_color_manual(values = c("State 1" = "black", "State 2" = "grey","State 3"="red"))+
  labs(x = NULL)+
  labs(y = expression(s[beta]))+
  theme_bw()  +
  theme(legend.position = "none",legend.title = element_blank(),legend.text = element_text(size = 8),
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
text= "European Union Carbon Price"
tgrob <- text_grob(text,size = 15)
# Draw the text
plot_0 <- as_ggplot(tgrob) + theme(plot.margin = margin(0,0,0,0, "cm"))

ggarrange(plot_0,a,b,c,d,e,
          ncol = 1,nrow = 6,heights = c(1, 5, 5,5,5,5,5), 
          labels = c("","A", "B","C","D","E"),label.y = 1.1)

#ggsave("EUA_3_state_par_1.png",width =7,height=15,dpi = 500)

#概率转移图
library(reshape2)
#状态转移图


date = end_dates
data = df[,17:25]
data = as.matrix(data)
colnames(data) = c("p111","p211","p311","p112","p212","p312","p113","p213","p313")
rownames(data) = date
datap <- melt(data)
colnames(datap) = c("x","ci","G")
datap$ci <- factor(datap$ci, levels=c("p111","p211","p311","p112","p212","p312","p113","p213","p313"))
datap$x = as.Date(datap$x)
v1= ggplot(data = datap,aes(x=x,y=G))+
  geom_line(aes(group = ci,color=ci,linetype = ci),size=0.8)+
  geom_point(aes(group = ci,color=ci,shape = ci),size=1.5)+
  scale_shape_manual(values=1:9)+
  #scale_color_manual(values=c("p111","p211","p311","p112","p212","p312","p113","p213","p313"))+
  #xlab("                      Algorithm steps")+#横坐标名称
  ylab("Transition probabilities")+#纵坐标名称
  theme_bw() +#去掉背景灰色
  theme(legend.position = "right",legend.title = element_blank(),legend.text = element_text(size = 12),
        legend.background = element_rect(fill = "transparent"),
        legend.box.background = element_rect(color = "#333333", linewidth = 0.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        axis.text.x = element_text(size = 12,angle = 0,vjust = 0.6),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 10),
        panel.grid.minor = element_blank(),
        axis.title.x = element_text( vjust = -0.2, hjust = -0.2)
  )+#以上theme中代码用于去除网格线且保留坐标轴边框
  scale_x_date(date_breaks="1 year",date_labels="%Y",expand = c(0, 0))+#更改横坐标刻度值
  scale_y_continuous(limits = c(0,1),breaks = seq(0,1,0.5),expand = c(0, 0))
v1

date = end_dates
data = df[,26:34]
data = as.matrix(data)
rownames(data) = date
colnames(data) = c("p121","p221","p321","p122","p222","p322","p123","p223","p323")
datap <- melt(data)
colnames(datap) = c("x","ci","G")
datap$ci <- factor(datap$ci, levels= c("p121","p221","p321","p122","p222","p322","p123","p223","p323"))
datap$x = as.Date(datap$x)
v2=ggplot(data = datap,aes(x=x,y=G))+
  geom_line(aes(group = ci,color=ci,linetype = ci),size=0.8)+
  geom_point(aes(group = ci,color=ci,shape = ci),size=1.5)+
  scale_shape_manual(values=1:9)+
  #xlab("                      Algorithm steps")+#横坐标名称
  ylab("Transition probabilities")+#纵坐标名称
  theme_bw() +#去掉背景灰色
  theme(legend.position = "right",legend.title = element_blank(),legend.text = element_text(size = 12),
        legend.background = element_rect(fill = "transparent"),
        legend.box.background = element_rect(color = "#333333", linewidth = 0.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        axis.text.x = element_text(size = 12,angle = 0,vjust = 0.3,hjust = 0.2),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 10),
        panel.grid.minor = element_blank(),
        axis.title.x = element_text( vjust = -0.2, hjust = -0.2)
  )+#以上theme中代码用于去除网格线且保留坐标轴边框
  scale_x_date(date_breaks="1 year",date_labels="%Y",expand = c(0, 0))+#更改横坐标刻度值
  scale_y_continuous(limits = c(0,1),breaks = seq(0,1,0.5),expand = c(0, 0))
v2

date = end_dates
data = df[,35:43]
data = as.matrix(data)
rownames(data) = date
colnames(data) = c("p131","p231","p331","p132","p232","p332","p133","p233","p333")
datap <- melt(data)
colnames(datap) = c("x","ci","G")
datap$ci <- factor(datap$ci, levels=c("p131","p231","p331","p132","p232","p332","p133","p233","p333"))
datap$x = as.Date(datap$x)
v3=ggplot(data = datap,aes(x=x,y=G))+
  geom_line(aes(group = ci,color=ci,linetype = ci),size=0.8)+
  geom_point(aes(group = ci,color=ci,shape= ci),size=1.5)+
  scale_shape_manual(values=1:9)+
  #xlab("                      Algorithm steps")+#横坐标名称
  ylab("Transition probabilities")+#纵坐标名称
  theme_bw() +#去掉背景灰色
  theme(legend.position = "right",legend.title = element_blank(),legend.text = element_text(size = 12),
        legend.background = element_rect(fill = "transparent"),
        legend.box.background = element_rect(color = "#333333", linewidth = 0.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        axis.text.x = element_text(size = 12,angle = 0,vjust = 0.3,hjust = 0.2),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 10),
        panel.grid.minor = element_blank(),
        axis.title.x = element_text( vjust = -0.2, hjust = -0.2)
  )+#以上theme中代码用于去除网格线且保留坐标轴边框
  scale_x_date(date_breaks="1 year",date_labels="%Y",expand = c(0, 0))+#更改横坐标刻度值
  scale_y_continuous(limits = c(0,1),breaks = seq(0,1,0.5),expand = c(0, 0))
v3


ggarrange(v1,v2,v3,nrow=3,ncol=1)
#ggsave("EUA_3_state_Tran_1.png",width =8,height=12,dpi = 500)



#状态估计图

library(dplyr)
df_g = data.frame(date =as.Date(data1$日期[-1]),fit0$state_prob_filter[1,],fit0$state_prob_filter[2,],fit0$state_prob_filter[3,])
end_dates_data = data.frame(date = end_dates)
df_h = left_join(end_dates_data,df_g,by = "date")

ggplot(data = df_h,aes(x=date,y=df_h[,2]))+
  geom_line(linetype = "solid",size=0.5,aes(color = "State 1"))+
  geom_point(aes(x=date,y=df_h[,2],color = "State 1"),size=1.2)+
  geom_line(data=df_h,aes(x=date, y=df_h[,3],color = "State 2"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_h[,3],color = "State 2"),size=1.2,shape=17)+
  geom_line(data=df_h,aes(x=date, y=df_h[,4],color = "State 3"),linetype = "dashed",size=0.5)+
  geom_point(aes(x=date,y=df_h[,4],color = "State 3"),size=1.2,shape=15)+
  scale_color_manual(values = c("State 1" = "black", "State 2" = "blue","State 3"="red"))+
  labs(x = NULL)+
  labs(y = "state estimate")+
  theme_bw()  +
  theme(legend.position = "right",legend.title = element_blank(),legend.text = element_text(size = 8),
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
  scale_x_date(date_breaks="1 year",date_labels="%Y-%m",expand = c(0, 0))+
  scale_y_continuous(breaks = seq(0,1,0.5))
@ggsave("EUA_state_est_3_1.png",width =7,height=4,dpi = 500)




plot(fit0$state_prob[1,], type="l", xlab="t", ylab="P(y_t=a | data)")
lines(fit0$state_prob[2,], type="l")
legend("topright", legend=c("State 1", "State 2"), lty=1, bty="n")



