import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
experiment = input("experiment: ")
figurename = input("figurename: ")

kappas = np.linspace(0.1,2.1,21)
signals = np.int64(np.linspace(0,d-1,d//2))
rho = 0.01
Ctr = np.diag(([i for i in range(1,d+1)])[::-1]); Ctr = (d/np.trace(Ctr))*Ctr

filepath_m = f'runs/{experiment}/simulation_m.txt'
with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
data = ast.literal_eval(contents)
data = sorted(data, key=lambda x: x[0])

filepath_s = f'runs/{experiment}/simulation_s.txt'
with open(filepath_s, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
stds = ast.literal_eval(contents)
stds = sorted(stds, key=lambda x: x[0])

sns.set(style="white",font_scale=2,palette="mako")
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams["figure.figsize"] = (16,10)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plot_inds = [0, 39, 49, 54, 59] 

kappas_theory = np.linspace(0.05,2.15,43)
plt.plot(kappas_theory, [ICL_error(Ctr, Ctr, tau, alpha, kappa, rho, 100) for kappa in kappas_theory], color='red', label = f'Test on pretrain')
plt.scatter(kappas, [item[1][-1] for item in data],  color='#FAFAFA', edgecolors='red',s=150,linewidth=2.5,zorder=10)
plt.fill_between(kappas, np.array([item[1][-1] for item in data]) - np.array([item[1][-1] for item in stds]), np.array([item[1][-1] for item in data]) + np.array([item[1][-1] for item in stds]), color='red', alpha = 0.2)

for i, plot_ind in enumerate(plot_inds):
    plt.plot(kappas_theory, [ICL_error(Ctr, np.diag(spikevalue(d,0,signals[plot_ind])), tau, alpha, kappa, rho, 100) for kappa in kappas_theory], color=color_cycle[i+1],label = f'Test direction {signals[plot_ind]+1}/{d}')
    plt.scatter(kappas, [item[1][plot_ind] for item in data], color='#FAFAFA', edgecolors=color_cycle[i+1],s=150,linewidth=2.5,zorder=10)
    plt.fill_between(kappas, np.array([item[1][plot_ind] for item in data]) - np.array([item[1][plot_ind] for item in stds]), np.array([item[1][plot_ind] for item in data]) + np.array([item[1][plot_ind] for item in stds]), color=color_cycle[i+1], alpha = 0.2)

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



