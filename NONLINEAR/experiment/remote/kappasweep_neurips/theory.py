import numpy as np
import random
from scipy import optimize

# ------------------------------------------------------------------------------------
# -------------------------STIELTJES TRANSFORMS---------------------------------------

def M_kappa_nu_empirical(kappa, nu, Ctr, numavg=10):
  d = len(Ctr)
  estimate = 0
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

def M_M2_empirical(kappa, nu, Ctr, numavg=10):
    d = len(Ctr)
    estimate1 = 0; estimate2 = 0;
    for _ in range(numavg):
        samples = np.random.multivariate_normal(np.zeros(d), Ctr, int(d*kappa)).T;
        samplecov = (samples@samples.T)/int(d*kappa)
        R = np.linalg.inv(samplecov + nu*np.eye(d))
        estimate1 += (1/d)*np.trace(R)
        estimate2 += (1/d)*np.trace(R@R)
    estimate1 = estimate1/numavg; estimate2 = estimate2/numavg;
    return estimate1, estimate2

def M_kappa(nu, kappa, c):
    # Ctr = c*I
    return 2 / ( (nu + c - c/kappa) + np.sqrt((nu + c - c/kappa)**2 + 4*c*nu/kappa) )

def M_kappa_prime(nu, kappa, c):
    M = M_kappa(nu, kappa, c)
    return (-1/2) * M**2 * (1 + (nu + c + c/kappa)/np.sqrt((nu + c - c/kappa)**2 + 4*c*nu/kappa))

def objectivefunc(xi, tau, kappa, Ctr, rhotr_alpha):
    return xi*M_kappa_nu_empirical(kappa, xi + rhotr_alpha, Ctr) + tau - 1
def xi_tau_less_1(tau, kappa, Ctr, rhotr_alpha):
    leftbound = 0
    rightbound = 2*rhotr_alpha*tau
    while objectivefunc(rightbound, tau, kappa, Ctr, rhotr_alpha) < 0:
        rightbound = rightbound+1
    root = optimize.brentq(objectivefunc, leftbound, rightbound, args=(tau, kappa, Ctr, rhotr_alpha))
    return root

def resolvent_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=10):
    d = len(Ctr)
    rhotr = (1/d)*np.trace(Ctr) + rho

    if tau == 1:
        return None
    if tau > 1:
        xi = 0
    if tau < 1:
        xi = xi_tau_less_1(tau, kappa, Ctr, rhotr/alpha)

    nu = rhotr/alpha + xi
    M, _ = M_M2_empirical(kappa, nu, Ctr, numavg); 
    FR = np.linalg.inv((1 - 1/kappa + (nu/kappa)*M)*Ctr + nu*np.eye(d))
    return (1/d)*np.trace(Ctest@FR)

# ------------------------------------------------------------------------------------
# -----------------------------------ICL ERRORS---------------------------------------

def ICL_error(Ctr, Ctesthat, tau, alpha, kappa, rho, numavg=10):
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
    M, M2 = M_M2_empirical(kappa, nu, Ctr, numavg); Mprime = -M2;
    FR = np.linalg.inv((1 - 1/kappa + (nu/kappa)*M)*Ctr + nu*np.eye(d))
    FR2 = FR @ (((1/kappa)*M + (nu/kappa)*Mprime)*Ctr + np.eye(d)) @ FR
    
    idg = (rho + nu - (nu**2)*M - xi*(1 - 2*nu*M - (nu**2)*Mprime))/(tau - (1 - 2*xi*M - (xi**2)*Mprime))
    pretraining_term = rho + (rhotest/alpha)*(1 + (idg-2*nu)*M + (xi*idg - nu**2)*Mprime)
    interaction_term = idg*(1/d)*np.trace(Ctesthat@FR) - (idg*xi - nu**2)*(1/d)*np.trace(Ctesthat@FR2)
    return pretraining_term+interaction_term

def ICL_pretraining(Ctr, tau, alpha, kappa, rho, numavg=10):
    d = len(Ctr)
    rhotr = (1/d)*np.trace(Ctr) + rho

    if tau == 1:
        return None
    if tau > 1:
        xi = 0
    if tau < 1:
        xi = xi_tau_less_1(tau, kappa, Ctr, rhotr/alpha)

    nu = rhotr/alpha + xi
    M, M2 = M_M2_empirical(kappa, nu, Ctr, numavg); Mprime = -M2;
    
    idg = (rho + nu - (nu**2)*M - xi*(1 - 2*nu*M - (nu**2)*Mprime))/(tau - (1 - 2*xi*M - (xi**2)*Mprime))
    pretraining_term = rho + ((rho+1)/alpha)*(1 + (idg-2*nu)*M + (xi*idg - nu**2)*Mprime)
    return pretraining_term

def ICL_alignment(Ctr, Ctesthat, tau, alpha, kappa, rho, numavg=10):
    d = len(Ctr)
    rhotr = (1/d)*np.trace(Ctr) + rho

    if tau == 1:
        return None
    if tau > 1:
        xi = 0
    if tau < 1:
        xi = xi_tau_less_1(tau, kappa, Ctr, rhotr/alpha)

    nu = rhotr/alpha + xi
    M, M2 = M_M2_empirical(kappa, nu, Ctr, numavg); Mprime = -M2;
    FR = np.linalg.inv((1 - 1/kappa + (nu/kappa)*M)*Ctr + nu*np.eye(d))
    FR2 = FR @ (((1/kappa)*M + (nu/kappa)*Mprime)*Ctr + np.eye(d)) @ FR
    
    idg = (rho + nu - (nu**2)*M - xi*(1 - 2*nu*M - (nu**2)*Mprime))/(tau - (1 - 2*xi*M - (xi**2)*Mprime))
    interaction_term = idg*(1/d)*np.trace(Ctesthat@FR) - (idg*xi - nu**2)*(1/d)*np.trace(Ctesthat@FR2)
    return interaction_term

def icl_scaled_isotropic(tau, alpha_tr, alpha_test, kappa, rho, ctr, ctest):
    if tau == 1:
        return None
    if tau > 1:
        xi = 0
        nu = (rho + ctr)/alpha_tr
    if tau < 1:
        xi = ((1-tau)/tau)/M_kappa((rho + ctr)/alpha_tr, kappa/tau, ctr)
        nu = (rho + ctr)/alpha_tr + xi

    M = M_kappa(nu, kappa, ctr)
    Mprime = M_kappa_prime(nu,kappa,ctr)
    zero = ctest + rho
    linear = ctest*(1 - nu*M)
    quadratic_eq = ((ctest+rho)/alpha_test + ctest)*(1 - 2*nu*M - (nu**2)*Mprime)
    c_e = ((rho + ctr) - (ctr-nu+(nu**2)*M) - xi*(1 - 2*nu*M - (nu**2)*Mprime))/(1 - 2*xi*M - (xi**2)*Mprime - tau)
    quadratic_extra = -c_e * ((ctest+rho)/alpha_test + ctest) * (M + xi*Mprime)
    return zero -2*linear + quadratic_eq + quadratic_extra

def prop_icl(tau, alpha, kappa, rho):
    q = (1 + rho) / alpha
    m = M_kappa(q, kappa, 1)
    M = M_kappa_prime(q, kappa, 1)
    mu = q * M_kappa(q, kappa/tau, 1)

    if tau < 1:
        result = (tau * (1 + q) / (1 - tau)) * (1 - tau * (1 - mu)**2 + mu*((rho/q - 1)) ) - 2 * tau * (1 - mu) + rho + 1
    else:
        result = (1+q)*(1-2*q*m-(q**2)*M + (m/(tau-1))*(rho+q-(q**2)*m))-2*(1-q*m) + 1 + rho
    return result

def piecewise_hypergeometric(d,p,epsilon=0.3):
    p = float(p)
    d = float(d)
    if p < 1-epsilon:
        return (d**(-p))/(1-p)
    if 1-epsilon <= p and p <= 1+epsilon:
        return (np.log(d)/d)*(1- (p-1)*np.log(d)/2 + ((p-1)**2)*(np.log(d)**2)/6)
    if p > 1+epsilon:
        return (d**(-1))/(p-1)

def spike_signal_icl(tau, alpha, kappa, rho, d, p, signalindex, epsilon=0.3):
    ctr = piecewise_hypergeometric(d,p,epsilon)
    nu = (rho + ctr)/alpha
    M = M_kappa(nu, kappa, ctr)
    Mprime = M_kappa_prime(nu, kappa, ctr)
    f = 1/(nu + (1 - 1/kappa + nu*M/kappa)/(signalindex**p))
    f2 = (1 + (M/kappa + nu*Mprime/kappa)/(signalindex**p))*(f**2)

    result = 1+rho - 2*(1 - nu*f) + (1 - 2*nu*f + (nu**2)*f2) + ((1+rho)/alpha)*(1 - 2*nu*M - (nu**2)*Mprime) + (f + ((1+rho)/alpha)*M)*(rho + nu - (nu**2)*M)/(tau - 1)
    return result

# ------------------------------------------------------------------------------------
# -------------------------------SPIKED SIGNALS---------------------------------------

def spikevalue(d, spikefactor, index):
    vals = np.zeros(d)
    for j in range(d):
        if j == index:
            vals[j] = d - spikefactor*(d-1)
        else:
            vals[j] = spikefactor
    return vals

def ICL_for_spiked_test(Ctr, index_eignfuncs, spikefactor, tau, alpha, kappa, rho, numavg=10):
    d = len(Ctr)
    rhotr = (1/d)*np.trace(Ctr) + rho

    if tau == 1:
        return None
    if tau > 1:
        xi = 0
    if tau < 1:
        xi = xi_tau_less_1(tau, kappa, Ctr, rhotr/alpha)

    nu = rhotr/alpha + xi
    M, M2 = M_M2_empirical(kappa, nu, Ctr, numavg); Mprime = -M2;
    FR = np.linalg.inv((1 - 1/kappa + (nu/kappa)*M)*Ctr + nu*np.eye(d))
    FR2 = FR @ (((1/kappa)*M + (nu/kappa)*Mprime)*Ctr + np.eye(d)) @ FR

    spikes = []
    for ind in index_eignfuncs:
        Ctest = np.diag(spikevalue(d,spikefactor,ind))
        rhotest = (1/d)*np.trace(Ctest) + rho;
        gamma_Atest = (1/d)*np.trace(Ctest) - nu*(1/d)*np.trace(Ctest@FR)
        gammaeq_B_gammaeq = (1/d)*np.trace((Ctest + (rhotest/alpha)*np.eye(d)) @ (np.eye(d) - 2*nu*FR + (nu**2)*FR2))
        chi_prime_0_term = (1/d)*np.trace((Ctest + (rhotest/alpha)*np.eye(d))@(FR - xi*FR2))/(1 - 2*xi*M - (xi**2)*Mprime - tau)
        quadratic_extra = -chi_prime_0_term * (rhotr - ((1/d)*np.trace(Ctr) - nu + (nu**2)*M) - xi*(1 - 2*nu*M - (nu**2)*Mprime))
        spikes.append(rhotest - 2*gamma_Atest + (gammaeq_B_gammaeq + quadratic_extra))

    return np.array(spikes)


def cka(d, A, B):
    H = np.eye(d) - (1/d)*np.outer(np.ones(d), np.ones(d)) 
    return np.trace(H@A@H@B)/((d-1)**2)