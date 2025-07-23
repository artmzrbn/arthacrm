
    data {
      int<lower=1> N; int<lower=1> K; int<lower=1> T; int<lower=1> F;
      array[T + F] int<lower=0> k;
      array[N, T] int<lower=0> x;
      array[N] int<lower=0> t;
    }
    transformed data {
      array[N] int<lower=0> X_total;
      for (i in 1:N) { X_total[i] = sum(x[i]); }
    }
    parameters {
      vector<lower=0>[N] lambda; vector[N] beta_adj; vector[K] s_adj;
      real<lower=0> r; real<lower=0> alpha;
      real<lower=0> tau_s; real<lower=0> tau_beta;
      real<lower=0.1> phi_c; real<lower=0, upper=1> phi_mu;
    }
    transformed parameters {
      vector[N] beta; vector[K] s;
      real<lower=0> sigma_s = 1 / sqrt(tau_s);
      real<lower=0> sigma_beta = 1 / sqrt(tau_beta);
      real<lower=0> a = phi_c * phi_mu;
      real<lower=0> b = phi_c * (1 - phi_mu);
      beta = (beta_adj - mean(beta_adj)) * sigma_beta + 1;
      s = s_adj - mean(s_adj);
    }
    model {
      vector[T] si_xi; vector[T] si_beta; vector[T + 1] l_sum;
      r ~ gamma(0.001, 0.001); alpha ~ gamma(0.001, 0.001);
      phi_c ~ pareto(0.1, 1.5); lambda ~ gamma(r, alpha);
      tau_s ~ gamma(0.001, 0.001); s_adj ~ normal(0, sigma_s);
      tau_beta ~ gamma(0.001, 0.001); beta_adj ~ normal(0, 1);
      for (i in 1:N) {
        for (j in 1:T) {
          si_xi[j] = x[i, j] * s[k[j]];
          si_beta[j] = exp(beta[i] * s[k[j]]);
        }
        for (tau in t[i]:T) {
          l_sum[tau] = lbeta(a + 1, b + tau - 1) - lambda[i] * sum(head(si_beta, tau));
        }
        l_sum[T + 1] = lbeta(a, b + T) - lambda[i] * sum(si_beta);
        target += (X_total[i] * log(lambda[i]) +
                   beta[i] * sum(si_xi) - lbeta(a, b) +
                   log_sum_exp(tail(l_sum, T - t[i] + 2)));
      }
    }
    generated quantities {
      matrix<lower=0>[N, F] f; vector[N] PZF; vector[N] PA;
      vector[T] xp_si_beta; vector[T + 1] xp_tau;
      vector[F + 2] xz_tau; vector[F] xpp_si_beta;
      for (i in 1:N) {
        for (j in 1:T) { xp_si_beta[j] = exp(beta[i] * s[k[j]]); }
        for (j in t[i]:T) {
          xp_tau[j] = lambda[i] * (sum(xp_si_beta) - sum(head(xp_si_beta, j))) +
                      lbeta(a + 1, b + j - 1) - lbeta(a, b + T);
        }
        xp_tau[T + 1] = 0;
        PA[i] = 1 / (exp(log_sum_exp(tail(xp_tau, T - t[i] + 2))));
        for (j in 1:F) {
          xpp_si_beta[j] = exp(beta[i] * s[k[T + j]]);
          f[i, j] = exp((log(lambda[i]) + beta[i] * s[k[j + T]] +
                       lbeta(a, b + j - 1) - lbeta(a, b))) * PA[i];
        }
        for (j in 1:F) {
          xz_tau[j] = -lambda[i] * sum(head(xpp_si_beta, j)) +
                      lbeta(a + 1, b + T + j - 1) - lbeta(a, b + T);
        }
        xz_tau[F + 1] = -lambda[i] * sum(xpp_si_beta) +
                        lbeta(a, b + T + F) - lbeta(a, b + T);
        xz_tau[F + 2] = log_sum_exp(segment(xp_tau, t[i], T - t[i] + 1));
        PZF[i] = exp(log_sum_exp(xz_tau) - log_sum_exp(tail(xp_tau, T - t[i] + 2)));
      }
    }
    