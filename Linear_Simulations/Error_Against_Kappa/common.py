import numpy as np
from scipy import optimize

# ----------------------------------- SIMULATION --------------------------------
# -------------------------------------------------------------------------------
def trace_formula_gamma(d, rho, l, mu, C, Gamma):
    C_hat = C + np.outer(mu, mu)
    x = (1 / d) * np.trace(C_hat) + rho
    alpha = l/d

    A_test = np.hstack([C_hat, (x*mu).reshape(-1,1)])

    B_top = np.hstack([C_hat  + (x/alpha)*np.eye(d), (x*mu).reshape(-1,1)])
    B_bottom = np.hstack([(x*mu).reshape(-1,1).T, np.array([[x**2]])])
    B_test = np.vstack([B_top, B_bottom])

    # Compute e(Gamma)
    term1 = rho
    term2 = (1 / d) * np.trace(C_hat)
    term3 = -(2 / d) * np.trace(Gamma.T @ A_test)
    term4 = (1 / d) * np.trace(Gamma @ B_test @ Gamma.T)

    e_Gamma = term1 + term2 + term3 + term4
    return e_Gamma

def draw_pretraining_data(n, d, l, k, rho, C):
    x = np.random.randn(n, l + 1, d) / np.sqrt(d)
    w_set = np.random.multivariate_normal(mean=np.zeros(d), cov=C, size = k)
    #w_set = np.array([w_set[i]/np.linalg.norm(w_set[i]) for i in range(k)])*np.sqrt(d)
    w_indices = np.random.randint(0, k, size=n)
    w = w_set[w_indices]
    epsilon = np.random.randn(n, l + 1) * np.sqrt(rho)
    y = np.einsum('nij,nj->ni', x, w) + epsilon
    return x, y, w

def construct_H_Z(x, y, l, d):
    y_sum_x = np.einsum('nij,ni->nj', x[:, :l, :], y[:, :l])
    y_sum_y = np.sum(y[:, :l] ** 2, axis=1)
    H_Z = np.zeros((x.shape[0], d, d + 1))
    H_Z[:, :, :d] = x[:, l, :, None] * (d / l) * y_sum_x[:, None, :]
    H_Z[:, :, d] = x[:, l] * (1 / l) * y_sum_y[:, None]
    return H_Z

def compute_Gamma_star(n, d, H_Z, y_l1, lambda_val):
    H_Z_vec = H_Z.reshape(n, -1)
    regularization_term = (n / d) * lambda_val * np.eye(H_Z_vec.shape[1])
    # Compute sum of outer products using matrix multiplication
    sum_term = H_Z_vec.T @ H_Z_vec
    # Compute y_l1 weighted sum using broadcasting
    weighted_sum = H_Z_vec.T @ y_l1
    Gamma_star_vec = np.linalg.inv(regularization_term + sum_term) @ weighted_sum
    return Gamma_star_vec.reshape(d, d + 1)

def simulation_Gamma_error(d, tau, alpha, kappa, rho, Ctr, Ctest, mu, lam=0.000001):
    # Ctr, Ctest, mu MUST have dimensions compatible with d 
    n = int(tau*d*d); l = int(alpha*d); k = int(kappa*d);
    x, y, _ = draw_pretraining_data(n, d, l, k, rho, Ctr)
    H_data = construct_H_Z(x, y, l, d)
    y_l1 = y[:, l]
    Gamma = compute_Gamma_star(n, d, H_data, y_l1, lam)
    return trace_formula_gamma(d, rho, l, mu, Ctest, Gamma)

# -------------------------- STATISTICALLY EQUIVALENT FASTER -------------------
# ------------------------------------------------------------------------------
def Householder(beta, x):
    s = np.sign(beta[0])
    u = np.zeros_like(beta)
    u[0] = np.linalg.norm(beta) * s
    u += beta
    u /= np.linalg.norm(u)

    return -s * (x - 2 * (u.T @ x) * u)

def construct_H_NEW(w, alpha, sigma_noise):
    d = len(w)
    N = np.int64(alpha * d)
    theta_w = np.linalg.norm(w)/np.sqrt(d)

    theta_e = np.linalg.norm(np.random.randn(N)) * (sigma_noise / np.sqrt(d))

    a = np.random.randn(1)
    theta_q = np.linalg.norm(np.random.randn(N-1))/np.sqrt(d)

    # This is h in the notes
    v = np.zeros((d, 1))
    v[0] = theta_e * a / np.sqrt(d) + theta_w * a**2 / d + theta_w * theta_q**2
    v[1:] = np.sqrt(((theta_e + theta_w *a /np.sqrt(d))**2 + theta_w**2 * theta_q**2)/d) * np.random.randn(d-1, 1)
    # This is the (Mh).T part of the first dxd of H
    av = Householder(w, v)
    # g = [s, u]
    g = np.random.randn(d, 1)
    s = g[0]
    # This is the (M[s, u]) part of the first dxd of H
    b = Householder(w, g)
    # final term coming from sum(yy) in H
    yy = np.sqrt(1/d)*(theta_w**2 * theta_q**2 + (theta_w * a/np.sqrt(d) + theta_e)**2)
    new_av = np.append(av, yy)

    # Never used
    # mu1 = theta_w**2 * a**2 / d + theta_w**2 * theta_q**2
    # mu2 = ((theta_w * a**2 / d + theta_w * theta_q**2)**2 - mu1/d)/theta_w**2

    # This is still correct
    y = theta_w * s + sigma_noise * np.random.randn(1)
    return (d/N)*np.outer(b.flatten(), new_av), y

def construct_HHT_fast_NEW(tasks, alpha, sigma_noise):
    n, d = tasks.shape

    H = np.zeros((d*(d+1), n))
    y_ary = np.zeros((n, 1))

    for i in range(n):
        h, y_ary[i] = construct_H_NEW(tasks[i,:].reshape(d, 1), alpha, sigma_noise)
        H[:, i] = h.reshape(-1)

    return H @ H.T, H @ y_ary

def learn_Gamma_fast_NEW(tasks, alpha, sigma_noise, lam, tau_max):
    n, d = tasks.shape

    n_max = np.int64(tau_max * d**2)

    idx = np.append(np.arange(0, n, n_max), n)

    M = np.zeros((d*(d+1), d*(d+1)))
    v = np.zeros((d*(d+1), 1))

    for i in range(len(idx)-1):
        H, y = construct_HHT_fast_NEW(tasks[idx[i]:idx[i+1],:], alpha, sigma_noise)
        M += H
        v += y

    Gamma = np.linalg.solve(M + (n/d)*lam * np.eye(d*(d+1)), v)

    return Gamma.reshape(d, d+1)

def final_gamma(d, tau, alpha, kappa, rho, Ctr, lam=0.000001):
    # Ctr, Ctest, mu MUST have dimensions compatible with d 
    l = int(alpha*d); n = int(tau*d*d); k = int(kappa*d)
    w_set = np.random.multivariate_normal(mean=np.zeros(d), cov=Ctr, size = k)
    w_indices = np.random.randint(0, k, size=n)
    tasks = w_set[w_indices]
    Gamma = learn_Gamma_fast_NEW(tasks, alpha, np.sqrt(rho), lam, 2)
    return Gamma

# ----------------------------------- Spiked Data -------------------------------
# -------------------------------------------------------------------------------
def spikevalue(d, spikefactor, index):
    vals = np.zeros(d)
    for j in range(d):
        if j == index:
            vals[j] = d - spikefactor*(d-1)
        else:
            vals[j] = spikefactor
    return vals

# ----------------------------------- THEORY --------------------------------
# -------------------------------------------------------------------------------

def M_kappa_nu_empirical(kappa, nu, Ctr):
  d = len(Ctr)
  estimate = 0
  numavg = 500
  for _ in range(numavg):
    samples = np.random.multivariate_normal(np.zeros(d), Ctr, int(d*kappa)).T;
    samplecov = (samples@samples.T)/int(d*kappa)
    estimate = estimate + (1/d)*np.trace(np.linalg.inv(samplecov + nu*np.eye(d)))
  estimate = estimate/numavg
  return estimate

def M2_kappa_nu_empirical(kappa, nu, Ctr):
  d = len(Ctr)
  estimate = 0
  numavg = 500
  for _ in range(numavg):
    samples = np.random.multivariate_normal(np.zeros(d), Ctr, int(d*kappa)).T;
    samplecov = (samples@samples.T)/int(d*kappa)
    estimate = estimate + (1/d)*np.trace(np.linalg.inv((samplecov + nu*np.eye(d))@(samplecov + nu*np.eye(d))))
  estimate = estimate/numavg
  return estimate

def objectivefunc(xi, tau, kappa, Ctr, rhotr_alpha):
    return xi*M_kappa_nu_empirical(kappa, xi + rhotr_alpha, Ctr) + tau - 1

def xi_tau_less_1(tau, kappa, Ctr, rhotr_alpha):
    leftbound = 0
    rightbound = 2*rhotr_alpha*tau
    while objectivefunc(rightbound, tau, kappa, Ctr, rhotr_alpha) < 0:
        rightbound = rightbound+1
    root = optimize.brentq(objectivefunc, leftbound, rightbound, args=(tau, kappa, Ctr, rhotr_alpha))
    return root

def ICL_error(Ctr, Ctesthat, tau, alpha, kappa, rho):
    d = len(Ctr)
    rhotr = (1/d)*np.trace(Ctr) + rho
    rhotest = (1/d)*np.trace(Ctesthat) + rho;

    if tau == 1:
        return None
    if tau > 1:
        xi = 0
    if tau < 1:
        xi = xi_tau_less_1(tau, kappa, Ctr, rhotr/alpha)

    nu = rhotr/alpha + xi
    M =  M_kappa_nu_empirical(kappa, nu, Ctr)
    Mprime = -M2_kappa_nu_empirical(kappa, nu, Ctr);
    FR = np.linalg.inv((1 - 1/kappa + (nu/kappa)*M)*Ctr + nu*np.eye(d))
    FR2 = FR @ (((1/kappa)*M + (nu/kappa)*Mprime)*Ctr + np.eye(d)) @ FR

    idg = (rho + nu - (nu**2)*M - xi*(1 - 2*nu*M - (nu**2)*Mprime))/(tau - (1 - 2*xi*M - (xi**2)*Mprime))
    pretraining_term = rho + (rhotest/alpha)*(1 + (idg-2*nu)*M + (xi*idg - nu**2)*Mprime)
    interaction_term = idg*(1/d)*np.trace(Ctesthat@FR) - (idg*xi - nu**2)*(1/d)*np.trace(Ctesthat@FR2)
    return pretraining_term, interaction_term

