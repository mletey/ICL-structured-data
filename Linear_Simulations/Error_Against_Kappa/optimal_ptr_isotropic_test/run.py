import numpy as np
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(".."))  # Adds the parent directory to the module search path
from common import *

directory = sys.argv[1]
d = int(sys.argv[2])
alpha = float(sys.argv[3])
tau = float(sys.argv[4])
numavg = int(sys.argv[5])
kappaind = int(sys.argv[6])-1
matched = int(sys.argv[7])
kappas = np.linspace(0.1,1.1,21) #np.linspace(0.1,4.1,21)
kappa = kappas[kappaind]
print('kappa is ', kappa)

if matched == 1:
    Ctr = np.eye(d)
if matched == 0:
    Ctr = np.diag(spikevalue(d,0,0))

rho = 0.01

test_errors = []
for i in tqdm(range(numavg)):
    Gamma = final_gamma(d, tau, alpha, kappa, rho, Ctr, lam=0.000001)
    Ctest = np.eye(d)
    test_errors.append(trace_formula_gamma(d, rho, int(alpha*d), np.zeros(d), Ctest, Gamma))

test_error_m = np.mean(test_errors)
test_error_s = np.std(test_errors)

ind = kappaind
filename = f'{directory}/simulation_m.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {test_error_m}],')
filename = f'{directory}/simulation_s.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {test_error_s}],')
