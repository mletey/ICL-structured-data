import ast
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
experiment = input("experiment: ")
# filepath_s = input("filepath_s: ")
figurename = input("figurename: ")
kappas = np.linspace(0.2,2.2,21)
signals = np.int64(np.linspace(0,d-1,d//2))
rho = 0.01
Ctr = np.diag([i for i in range(1,d+1)]); Ctr = (d/np.trace(Ctr))*Ctr

filepath_m = f'runs/{experiment}/simulation_m.txt'
with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
data = ast.literal_eval(contents)
data = sorted(data, key=lambda x: x[0])

filepath_m = f'runs/{experiment}/simulation_s.txt'
with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
stds = ast.literal_eval(contents)
stds = sorted(data, key=lambda x: x[0])

sns.set(style="white",font_scale=2,palette="mako")
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams["figure.figsize"] = (16,10)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

Ctest1 = Ctr
Ctest2 = np.eye(d)
power = 0.5
Ctest3_antialigned_power = np.diag(np.array([(j + 1) ** -power for j in range(d)])); Ctest3_antialigned_power = (Ctest3_antialigned_power/np.trace(Ctest3_antialigned_power))*d
Ctest4_aligned_power = np.diag(np.array([(d - j) ** -power for j in range(d)])); Ctest4_aligned_power = (Ctest4_aligned_power/np.trace(Ctest4_aligned_power))*d
Ctest5 = np.diag(spikevalue(d,0,d-1))

plt.axhline(rho,linestyle=':',linewidth=2, color='grey')
plt.axvline(1,linestyle=':',linewidth=2,color='grey')

plt.plot(kappas, [ICL_error(Ctr, Ctest5, tau, alpha, kappa, rho, 100) for kappa in kappas], color=color_cycle[2], label = f'Max-aligned spike')
plt.scatter(kappas, [item[1][4] for item in data], color='#FAFAFA', edgecolors=color_cycle[2],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(kappas, np.array([item[1][4] for item in data]) - np.array([item[1][4] for item in stds]), np.array([item[1][4] for item in data]) + np.array([item[1][4] for item in stds]), color=color_cycle[2], alpha = 0.3)

plt.plot(kappas, [ICL_error(Ctr, Ctest1, tau, alpha, kappa, rho, 100) for kappa in kappas],  color = 'red', label = f'Test on pretrain')
plt.scatter(kappas, [item[1][0] for item in data], color='#FAFAFA', edgecolors='red',s=150,linewidth=2.5,zorder=10)
# plt.fill_between(kappas, np.array([item[1][0] for item in data]) - np.array([item[1][0] for item in stds]), np.array([item[1][0] for item in data]) + np.array([item[1][0] for item in stds]), color='red', alpha = 0.3)

plt.plot(kappas, [ICL_error(Ctr, Ctest4_aligned_power, tau, alpha, kappa, rho, 100) for kappa in kappas], color=color_cycle[3], label = f'Test on aligned powerlaw')
plt.scatter(kappas, [item[1][3] for item in data], color='#FAFAFA', edgecolors=color_cycle[3],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(kappas, np.array([item[1][3] for item in data]) - np.array([item[1][3] for item in stds]), np.array([item[1][3] for item in data]) + np.array([item[1][3] for item in stds]), color=color_cycle[3], alpha = 0.3)

plt.plot(kappas, [ICL_error(Ctr, Ctest2, tau, alpha, kappa, rho, 100) for kappa in kappas], color=color_cycle[4],label = f'Test on isotropic')
plt.scatter(kappas, [item[1][1] for item in data], color='#FAFAFA', edgecolors=color_cycle[4],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(kappas, np.array([item[1][1] for item in data]) - np.array([item[1][1] for item in stds]), np.array([item[1][1] for item in data]) + np.array([item[1][1] for item in stds]), color=color_cycle[4], alpha = 0.3)

plt.plot(kappas, [ICL_error(Ctr, Ctest3_antialigned_power, tau, alpha, kappa, rho, 100) for kappa in kappas], color=color_cycle[5], label = f'Test on unaligned powerlaw')
plt.scatter(kappas, [item[1][2] for item in data], color='#FAFAFA', edgecolors=color_cycle[5],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(kappas, np.array([item[1][2] for item in data]) - np.array([item[1][2] for item in stds]), np.array([item[1][2] for item in data]) + np.array([item[1][2] for item in stds]), color=color_cycle[5], alpha = 0.3)

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


