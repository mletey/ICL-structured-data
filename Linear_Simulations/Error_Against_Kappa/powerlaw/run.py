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
kappas = np.array([0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2])#np.linspace(0.1,4.1,21)
kappa = kappas[kappaind]
print('kappa is ', kappa)

train_power = float(sys.argv[7])
test_power = float(sys.argv[8])

Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d
Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d

vals_simulation_FAST = []
rho = 0.01

for i in range(numavg):
    #vals_simulation.append(simulation_Gamma_error(d, tau, alpha, kappa, rho, Ctr, Ctest, np.zeros(d)))
    vals_simulation_FAST.append(simulation_Gamma_error_FAST(d, tau, alpha, kappa, rho, Ctr, Ctest, np.zeros(d)))
vals_simulation_FAST = np.array(vals_simulation_FAST)

ind = kappaind
filename = f'{directory}/simulation_m.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {np.mean(vals_simulation_FAST)}],')
filename = f'{directory}/simulation_s.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {np.std(vals_simulation_FAST)}],')
