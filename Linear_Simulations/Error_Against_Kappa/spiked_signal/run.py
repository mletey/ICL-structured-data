import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))  # Adds the parent directory to the module search path
from common import *

directory = sys.argv[1]
d = int(sys.argv[2])
alpha = float(sys.argv[3])
tau = float(sys.argv[4])
numavg = int(sys.argv[5])
kappaind = int(sys.argv[6])-1
kappas = np.linspace(0.2,2.2,21) #np.linspace(0.1,4.1,21)
kappa = kappas[kappaind]
print('kappa is ', kappa)

signals = np.int64(np.linspace(0,d-1,d//2))

# Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d

Ctr = np.diag([i for i in range(1,d+1)]); Ctr = (d/np.trace(Ctr))*Ctr

vals_simulation_FAST = []
rho = 0.01

test_errors = []
for i in range(numavg):
    runs = []
    #vals_simulation.append(simulation_Gamma_error(d, tau, alpha, kappa, rho, Ctr, Ctest, np.zeros(d)))
    Gamma = final_gamma(d, tau, alpha, kappa, rho, Ctr, lam=0.000001)
    for signal_index in signals:
        Ctest = np.diag(spikevalue(d,0,signal_index))
        runs.append(trace_formula_gamma(d, rho, int(alpha*d), np.zeros(d), Ctest, Gamma))
    test_errors.append(runs)

test_errors = np.array(test_errors)
average_test_error = np.mean(test_errors, axis = 0)
std_test_error = np.std(test_errors, axis = 0)

ind = kappaind
average_test_error = average_test_error.tolist() 
std_test_error = std_test_error.tolist() 
filename = f'{directory}/simulation_m.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {average_test_error}],')
filename = f'{directory}/simulation_s.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {std_test_error}],')
