import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
figurename = input("figurename: ")

kappas = np.linspace(0.1,1.1,21)
rho = 0.01
# Ctr = np.diag(([i for i in range(1,d+1)])[::-1]); Ctr = (d/np.trace(Ctr))*Ctr
Ctest = np.eye(d)

filepath_m = f'runs/optimal_ptr_d120_match/simulation_m.txt'
with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
data_match = ast.literal_eval(contents)
data_match = sorted(data_match, key=lambda x: x[0])

filepath_m = f'runs/optimal_ptr_d120_match/simulation_s.txt'
with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
stds_match = ast.literal_eval(contents)
stds_match = sorted(stds_match, key=lambda x: x[0])

filepath_s = f'runs/optimal_ptr_d120_spike/simulation_m.txt'
with open(filepath_s, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
data_spike = ast.literal_eval(contents)
data_spike = sorted(data_spike, key=lambda x: x[0])

filepath_s = f'runs/optimal_ptr_d120_spike/simulation_s.txt'
with open(filepath_s, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
stds_spike = ast.literal_eval(contents)
stds_spike = sorted(stds_spike, key=lambda x: x[0])

sns.set(style="white",font_scale=2,palette="mako")
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams["figure.figsize"] = (16,10)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

kappas_theory = np.linspace(0.1,1.1,41) #np.linspace(0.05,1.05,41)
Ctr1 = np.eye(d)
Ctr2 = np.diag(spikevalue(d,0,0))

#plt.plot(kappas_theory, [ICL_error(Ctr1, Ctest, tau, alpha, kappa, rho, 100) for kappa in kappas_theory], color='red', label = f'Pretrain on test')
plt.scatter(kappas, [item[1] for item in data_match],  color='#FAFAFA', edgecolors='red',s=150,linewidth=2.5,zorder=10, label = f'Pretrain on test')
plt.fill_between(kappas, np.array([item[1] for item in data_match]) - np.array([item[1] for item in stds_match]), np.array([item[1] for item in data_match]) + np.array([item[1] for item in stds_match]), color='red', alpha = 0.2)

#plt.plot(kappas_theory, [ICL_error(Ctr2, Ctest, tau, alpha, kappa, rho, 100) for kappa in kappas_theory], color=color_cycle[3], label = f'Pretrain on rank-1')
plt.scatter(kappas, [item[1] for item in data_spike],  color='#FAFAFA', edgecolors=color_cycle[3],s=150,linewidth=2.5,zorder=10, label = f'Pretrain on rank-1')
plt.fill_between(kappas, np.array([item[1] for item in data_spike]) - np.array([item[1] for item in stds_spike]), np.array([item[1] for item in data_spike]) + np.array([item[1] for item in stds_spike]), color=color_cycle[3], alpha = 0.2)

plt.subplots_adjust(right=0.75)  # Makes space for legend
leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
leg.get_frame().set_alpha(0)
plt.gca().spines['top'].set_color('lightgray')
plt.gca().spines['right'].set_color('lightgray')
plt.gca().spines['bottom'].set_color('lightgray')
plt.gca().spines['left'].set_color('lightgray')
plt.xlabel(r'$\kappa$ = k/d')
plt.ylabel('ICL error')
plt.gca().tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig(f'figs/{figurename}.png')



