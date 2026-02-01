###########################################################
## 0. Utility: augmented state index (second-order HMM -> N^2-dim first-order HMM)
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
## 2. Emission density f_a(Δx) (two-component Gaussian mixture, under Q)
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

stationary_distribution_augmented <- function(A, tol = 1e-12, max_iter = 10000) {
  M <- nrow(A)
  pi <- rep(1 / M, M)
  for (it in 1:max_iter) {
    new_pi <- as.vector(pi %*% A)   # row vector * column-stochastic A
    if (sum(abs(new_pi - pi)) < tol) {
      return(new_pi / sum(new_pi))
    }
    pi <- new_pi
  }
  pi / sum(pi)
}
###########################################################
## 3. Local RN increment φ_k(a) = f_a(Δx_k) / g(Δx_k)
##    Here g is the density of standard normal N(0, 1)
###########################################################

emit_logrn_mixture <- function(dx_t, params, N,
                               dt = 1, base_sd = 1,
                               esscher_theta = 0) {
  M <- N * N
  curr_state <- ((1:M - 1L) %% N) + 1L
  
  # Mixture density under Q: f_a(Δx)
  f_Q <- emit_mix_density_Q(dx_t, params, N, dt = dt)
  log_f_Q <- log(pmax(f_Q, 1e-300))
  
  # Simple reference density under P: g(Δx) (here take N(0, base_sd^2))
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
    
    # Apply Esscher factor: log φ^(θ) = log φ + θΔx - logM
    log_phi <- log_phi + th * dx_t - logM
  }
  # =====================================================
  
  list(
    log_phi = log_phi,          # length M: log φ_k^(θ)(i)
    log_g   = as.numeric(log_g) # scalar: log g(Δx_k)
  )
}

############################################
## 4. Simulate second-order HMM states    ##
############################################

simulate_states_second_order <- function(T, p_tensor, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  N <- dim(p_tensor)[1]
  A  <- build_A_from_p(p_tensor)
  pi <- stationary_distribution_augmented(A)
  
  M <- length(pi)
  init_idx <- sample.int(M, size = 1, prob = pi)
  pair0    <- pair_decode(init_idx, N)
  y <- integer(T + 1L)
  y[1] <- pair0["prev"]
  y[2] <- pair0["curr"]
  
  for (k in 2:T) {
    prev <- y[k - 1L]
    curr <- y[k]
    prob_next <- p_tensor[, curr, prev]
    prob_next <- prob_next / sum(prob_next)
    y[k + 1L] <- sample.int(N, size = 1, prob = prob_next)
  }
  y
}
###########################################################
## 4. Forward-backward algorithm (RN-derivative version)
##
##    Corresponding to Proposition 2: v_{k+1} = A D_{k+1} v_k,
##    implemented in log space for numerical stability.
###########################################################

fb_augmented_mixture_rn <- function(dx, A, params, N,
                                    dt = 1, base_sd = 1,
                                    esscher_theta = 0) {
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
  
  # The forward-backward part below does not need any changes
  log_alpha <- matrix(-Inf, M, Tn)
  log_beta  <- matrix(0,    M, Tn)
  cvec      <- numeric(Tn)
  
  log_alpha[, 1] <- -log(M) + logB[, 1]
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
  
  loglik_P <- sum(cvec)
  loglik_Qtheta <- loglik_P + sum(log_gv)  # this is the log-likelihood under Q^θ
  
  list(
    gamma = gamma,
    xi_sum = xi_sum,
    loglik_P = loglik_P,
    loglik_Qtheta = loglik_Qtheta
  )
}


simulate_jumpdiff_01 <- function(
    T, dt,
    p_tensor,
    mu, sigma, lambda, mu_beta, s_beta,
    X0 = 0,
    seed = NULL,
    return_price = TRUE
) {
  if (!is.null(seed)) set.seed(seed)
  N <- length(mu)
  stopifnot(
    length(sigma)  == N,
    length(lambda) == N,
    length(mu_beta)== N,
    length(s_beta) == N
  )
  if (any(lambda * dt > 1)) stop("lambda * dt must be <= 1 for 0/1 jump model.")
  
  y  <- simulate_states_second_order(T, p_tensor, seed = seed)
  X  <- numeric(T + 1L); X[1] <- X0
  dX <- numeric(T)
  J  <- integer(T)
  jump_size <- numeric(T)
  
  for (k in 1:T) {
    a <- y[k]
    theta_a <- (mu[a] - 0.5 * sigma[a]^2) * dt
    diff_sd <- sigma[a] * sqrt(dt)
    diff_inc<- theta_a + diff_sd * rnorm(1)
    
    prob_jump <- lambda[a] * dt
    J_k <- rbinom(1, size = 1, prob = prob_jump)
    J[k] <- J_k
    if (J_k == 1L) {
      jump_k <- rnorm(1, mean = mu_beta[a], sd = s_beta[a])
    } else {
      jump_k <- 0
    }
    jump_size[k] <- jump_k
    
    inc    <- diff_inc + jump_k
    dX[k]  <- inc
    X[k+1] <- X[k] + inc
  }
  
  df_main <- data.frame(t = 0:T, y = y, X = X)
  df_inc  <- data.frame(t = 1:T, dX = dX, J = J, jump_size = jump_size)
  out <- merge(df_main, df_inc, by = "t", all.x = TRUE)
  if (return_price) out$S <- exp(out$X)
  out
}
###########################################################
## 5. Jump posterior r[a,t] and state marginal probability γ_k(a)
###########################################################

jump_posterior <- function(dx, params, N, gamma) {
  Tn <- length(dx)
  M  <- N * N
  
  # Current regime a corresponding to each augmented state
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
## 6. M-step: update (ϑ_a, ϱ_a^2, λ_a, μ_{βa}, s_{βa}^2)
##    This part corresponds one-to-one with your notes 4.1–4.3.
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
    
    # ϑ_a (use conditional mean from the no-jump part only)
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
    
    # ϱ_a^2: second moment from the no-jump part
    if (sum(w0) > 0) {
      rho2_new[a] <- sum(w0 * (dx - theta_new[a])^2) / sum(w0)
    } else {
      rho2_new[a] <- params$rho2[a]
    }
    rho2_new[a] <- max(rho2_new[a], 1e-8)
    
    # s_{βa}^2: second moment from the jump part, subtracting diffusion variance
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
## 8. EM main function (RN-derivative version)
###########################################################

em_fit_jump_mixture_rn <- function(
    X, dt, N,
    p_init,
    mu_init, sigma_init, lambda_init, mu_beta_init, s_beta_init,
    max_iter = 1000, tol = 1e-5, verbose = TRUE,
    esscher_theta = 0      # <--- new parameter: Esscher tilting strength
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
      esscher_theta = esscher_theta   # <--- pass through
    )
    gamma  <- fb$gamma
    xi_sum <- fb$xi_sum
    
    stats   <- jump_posterior(dx, params, N, gamma)
    par_new <- mstep_params(dx, stats, params, N)
    p_new   <- mstep_transition(xi_sum, N)
    
    params <- par_new
    p      <- p_new
    
    if (verbose) {
      cat(sprintf("Iter %3d | loglik_P=%.6f | theta_E=%.3f | lambda=%s\n",
                  it, fb$loglik_P, esscher_theta,
                  paste(round(params$lambda, 4), collapse=",")))
    }
    
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


## ==============================
## 0. Some ground-truth settings
## ==============================
set.seed(2025)

N  <- 2          # number of states
T  <- 400        # number of steps
dt <- 1

# Second-order transition ground truth p[a,b,c]
p_true <- array(0, dim = c(N, N, N))
p_true[,1,1] <- c(0.92, 0.08)
p_true[,2,1] <- c(0.75, 0.25)
p_true[,1,2] <- c(0.30, 0.70)
p_true[,2,2] <- c(0.10, 0.90)

# Ground truth for continuous part + jump parameters
mu_true      <- c(0.3, 0.2)
sigma_true   <- c(0.15, 0.30)
lambda_true  <- c(0.25, 0.80)
mu_beta_true <- c(0.2, -0.5)
s_beta_true  <- c(0.10, 0.20)

## ==============================
## 1. Simulate a 0/1 jump-diffusion path
## ==============================
# Use the simulation function you already have from before; if not, paste your original version here.
# simulate_jumpdiff_01(T, dt, p_tensor, mu, sigma, lambda, mu_beta, s_beta, ...)

df <- simulate_jumpdiff_01(
  T = T, dt = dt,
  p_tensor = p_true,
  mu = mu_true, sigma = sigma_true,
  lambda = lambda_true,
  mu_beta = mu_beta_true, s_beta = s_beta_true,
  X0 = 0, seed = 2025, return_price = TRUE
)

X <- df$X   # log-price path

## ==============================
## 2. EM initialization (rough guess)
## ==============================
dx <- diff(X)

mu_init      <- c(mean(dx) + 0.02, mean(dx) - 0.02)
sigma_init   <- rep(sd(dx) / 2, N)
lambda_init  <- c(0.2, 0.6)
mu_beta_init <- c(-0.1, -0.3)
s_beta_init  <- c(0.2, 0.3)

# Initial transition probabilities: perturb around the truth a bit
p_init <- array(0, dim = c(N, N, N))
p_init[,1,1] <- c(0.85, 0.15)
p_init[,2,1] <- c(0.60, 0.40)
p_init[,1,2] <- c(0.40, 0.60)
p_init[,2,2] <- c(0.15, 0.85)

## ==============================
## 3. Call the EM function "with RN derivatives"
## ==============================

fit0 <- em_fit_jump_mixture_rn(
  X = X, dt = 1, N = 2,
  p_init = p_init,
  mu_init = mu_init, sigma_init = sigma_init,
  lambda_init = lambda_init,
  mu_beta_init = mu_beta_init,
  s_beta_init  = s_beta_init,
  esscher_theta = 0        # no tilting
)

fit_pos <- em_fit_jump_mixture_rn(
  X = X, dt = 1, N = 2,
  p_init = p_init,
  mu_init = mu_init, sigma_init = sigma_init,
  lambda_init = lambda_init,
  mu_beta_init = mu_beta_init,
  s_beta_init  = s_beta_init,
  esscher_theta = 1      # small tilt
)

fit_neg <- em_fit_jump_mixture_rn(
  X = X, dt = 1, N = 2,
  p_init = p_init,
  mu_init = mu_init, sigma_init = sigma_init,
  lambda_init = lambda_init,
  mu_beta_init = mu_beta_init,
  s_beta_init  = s_beta_init,
  esscher_theta = 1       # large tilt
)

## ==============================
## 4. Ground truth vs estimated values comparison
## ==============================

summary_tbl <- data.frame(
  state        = 1:N,
  mu_true      = mu_true,
  mu_hat       = fit0$mu,
  sigma_true   = sigma_true,
  sigma_hat    = fit0$sigma,
  lambda_true  = lambda_true,
  lambda_hat   = fit0$lambda,
  mu_beta_true = mu_beta_true,
  mu_beta_hat  = fit0$mu_beta,
  s_beta_true  = s_beta_true,
  s_beta_hat   = fit0$s_beta
)
print(summary_tbl)

summary_tbl <- data.frame(
  state        = 1:N,
  mu_true      = mu_true,
  mu_hat       = fit_pos$mu,
  sigma_true   = sigma_true,
  sigma_hat    = fit_pos$sigma,
  lambda_true  = lambda_true,
  lambda_hat   = fit_pos$lambda,
  mu_beta_true = mu_beta_true,
  mu_beta_hat  = fit_pos$mu_beta,
  s_beta_true  = s_beta_true,
  s_beta_hat   = fit_pos$s_beta
)
print(summary_tbl)

summary_tbl <- data.frame(
  state        = 1:N,
  mu_true      = mu_true,
  mu_hat       = fit_neg$mu,
  sigma_true   = sigma_true,
  sigma_hat    = fit_neg$sigma,
  lambda_true  = lambda_true,
  lambda_hat   = fit_neg$lambda,
  mu_beta_true = mu_beta_true,
  mu_beta_hat  = fit_neg$mu_beta,
  s_beta_true  = s_beta_true,
  s_beta_hat   = fit_neg$s_beta
)
print(summary_tbl)

############################################
## 12. Example: 1000 simulations + estimation ##
############################################
###########################################################
## 12.1 Single simulation + estimation: for Monte Carlo outer call run_one_mc ##
###########################################################

run_one_mc <- function(
    T, dt,
    p_tensor,
    mu_true, sigma_true, lam_true, mu_beta_true, s_beta_true,
    seed_sim = NULL,
    init_fixed = NULL,
    max_iter = 1000,esscher_theta = esscher_theta
){
  # ---- 1. Simulate path ----
  df <- simulate_jumpdiff_01(
    T = T, dt = dt,
    p_tensor = p_tensor,
    mu = mu_true, sigma = sigma_true,
    lambda = lam_true,
    mu_beta = mu_beta_true,
    s_beta = s_beta_true,
    X0 = 0, seed = seed_sim, return_price = FALSE
  )
  X <- df$X
  
  # ---- 2. Initialization: fixed or default ----
  if (is.null(init_fixed)) {
    dx <- diff(X)
    mu_init      <- c(mean(dx)+0.02, mean(dx)-0.02)
    sigma_init   <- c(sd(dx)/3, sd(dx)/2)
    lambda_init  <- c(0.2, 0.6)
    mu_beta_init <- c(-0.1, -0.3)
    s_beta_init  <- c(0.2, 0.3)
  } else {
    mu_init      <- init_fixed$mu
    sigma_init   <- init_fixed$sigma
    lambda_init  <- init_fixed$lambda
    mu_beta_init <- init_fixed$mu_beta
    s_beta_init  <- init_fixed$s_beta
  }
  
  # Transition initialization (close to truth but not the same)
  N <- length(mu_true)
  p_init <- array(0, dim = c(N,N,N))
  p_init[,1,1] <- c(0.80, 0.20)
  p_init[,2,1] <- c(0.6, 0.4)
  p_init[,1,2] <- c(0.4, 0.6)
  p_init[,2,2] <- c(0.15, 0.85)
  
  # ---- 3. EM estimation ----
  fit <- em_fit_jump_mixture_rn(
    X = X, dt = dt, N = N,
    p_init = p_init,
    mu_init = mu_init, sigma_init = sigma_init,
    lambda_init = lambda_init,
    mu_beta_init = mu_beta_init,
    s_beta_init  = s_beta_init,
    max_iter = max_iter, tol = 1e-5, verbose = FALSE,esscher_theta = esscher_theta 
  )
  
  # ---- 4. Output ----
  list(
    est = list(
      mu      = fit$mu,
      sigma   = fit$sigma,
      lambda  = fit$lambda,
      mu_beta = fit$mu_beta,
      s_beta  = fit$s_beta
    ),
    p_hat = fit$p
  )
}

###########################################################
## 12.2 Monte Carlo outer loop: R simulations, summarize Bias and Var        ##
###########################################################

mc_summary <- function(R, T, dt,
                       p_tensor,
                       mu_true, sigma_true, lam_true, mu_beta_true, s_beta_true,
                       seed_base = 1234,
                       progress = c("none","counter","bar"),
                       report_every = 10,esscher_theta=esscher_theta)
{
  progress <- match.arg(progress)
  set.seed(seed_base)
  
  res <- vector("list", R)
  
  if (progress == "bar") {
    pb <- utils::txtProgressBar(min=0, max=R, style=3)
    on.exit(close(pb))
  }
  
  for (r in 1:R) {
    res[[r]] <- run_one_mc(
      T = T, dt = dt,
      p_tensor = p_tensor,
      mu_true = mu_true, sigma_true = sigma_true,
      lam_true = lam_true,
      mu_beta_true = mu_beta_true,
      s_beta_true = s_beta_true,
      seed_sim = sample.int(1e9,1),esscher_theta= esscher_theta
    )
    
    if (progress == "counter") {
      if (r %% report_every == 0) {
        cat(sprintf("  finished %d / %d\n", r, R))
      }
    } else if (progress == "bar") {
      utils::setTxtProgressBar(pb, r)
    }
  }
  
  # ---- Summarize parameter matrices ----
  Mhat <- do.call(rbind, lapply(res, function(z) z$est$mu))
  Shat <- do.call(rbind, lapply(res, function(z) z$est$sigma))
  Lhat <- do.call(rbind, lapply(res, function(z) z$est$lambda))
  JmBh <- do.call(rbind, lapply(res, function(z) z$est$mu_beta))
  JsBh <- do.call(rbind, lapply(res, function(z) z$est$s_beta))
  Phat <- do.call(rbind, lapply(res, function(z) as.vector(z$p_hat)))
  p_true_vec <- as.vector(p_tensor)
  
  # ---- Summary helper ----
  summarise_param <- function(mat, true_vec, name) {
    tibble::tibble(
      param = name,
      state = 1:length(true_vec),
      true  = true_vec,
      mean  = colMeans(mat),
      Bias  = colMeans(mat) - true_vec,
      Var   = apply(mat, 2, sd)
    )
  }
  
  tab <- dplyr::bind_rows(
    summarise_param(Mhat, mu_true,  "mu"),
    summarise_param(Shat, sigma_true,"sigma"),
    summarise_param(Lhat, lam_true, "lambda"),
    summarise_param(JmBh, mu_beta_true, "mu_beta"),
    summarise_param(JsBh, s_beta_true,  "s_beta"),
    summarise_param(Phat, p_true_vec,   "p")
  )
  
  list(table = tab)
}

#############################################
## 12.3 Actually run Monte Carlo (e.g., R=100)  ##
#############################################
my_algo <- function() {
  set.seed(2025)
  T  <- 200
  dt <- 1
  R  <- 5   # set to 1000 if you need it
  
  # Ground truth (consistent with your earlier example)
  p_true <- p_true        # from Section 11's p_true
  mu_true      <- mu_true
  sigma_true   <- sigma_true
  lambda_true  <- lambda_true
  mu_beta_true <- mu_beta_true
  s_beta_true  <- s_beta_true
  
  cat("Running Monte Carlo...\n")
  
  mc_res <- mc_summary(
    R = R, T = T, dt = dt,
    p_tensor = p_true,
    mu_true = mu_true, sigma_true = sigma_true,
    lam_true = lambda_true,
    mu_beta_true = mu_beta_true,
    s_beta_true = s_beta_true,
    progress = "counter",
    report_every = 1,esscher_theta=0.0
  )
}

system.time( my_algo() )
mc_res = my_algo()
cat("\n===== Monte Carlo Summary =====\n")
print(mc_res$table)
