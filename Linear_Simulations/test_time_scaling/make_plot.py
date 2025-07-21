import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
# experiment = input("experiment: ")
figurename = input("figurename: ")

kappa = 1
# alpha_TESTS = np.linspace(alpha-1, alpha+1, 21)
alpha_TESTS = np.logspace(np.log10(0.1), np.log10(1000), 21)

rho = 0.01

Ctr0 = np.eye(d)
Ctr1 = np.diag(([i for i in range(1,d+1)])[::-1]); Ctr1 = (d/np.trace(Ctr1))*Ctr1
Ctr2 = np.diag(np.array([(j + 1) ** (-float(1.5)) for j in range(d)])); Ctr2 = (Ctr2/np.trace(Ctr2))*d

filepath_m = f'runs/d100_lessfuckingnoiseagain/simulation_m.txt'
with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
data_0 = ast.literal_eval(contents)
data_0 = sorted(data_0, key=lambda x: x[0])
print(data_0)
print(data_0[0][1])

filepath_m = f'runs/d100_lessfuckingnoiseagain/simulation_s.txt'
with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
stds_0 = ast.literal_eval(contents)
stds_0 = sorted(stds_0, key=lambda x: x[0])

# filepath_m = f'runs/d100_lessregularisation_morenoise_moreaveraging/simulation_m.txt'
# with open(filepath_m, 'r') as f:
#     contents = f.read().strip()
# if contents.endswith(','):
#     contents = contents[:-1]
# contents = f'[{contents}]'
# data_2 = ast.literal_eval(contents)
# data_2 = sorted(data_2, key=lambda x: x[0])
# filepath_m = f'runs/testime_d120_option2_5average/simulation_s.txt'
# with open(filepath_m, 'r') as f:
#     contents = f.read().strip()
# if contents.endswith(','):
#     contents = contents[:-1]
# contents = f'[{contents}]'
# stds_2 = ast.literal_eval(contents)
# stds_2 = sorted(stds_2, key=lambda x: x[0])

# filepath_m = f'runs/testime_d120_option1_5average/simulation_m.txt'
# with open(filepath_m, 'r') as f:
#     contents = f.read().strip()
# if contents.endswith(','):
#     contents = contents[:-1]
# contents = f'[{contents}]'
# data_1 = ast.literal_eval(contents)
# data_1 = sorted(data_1, key=lambda x: x[0])
# filepath_m = f'runs/testime_d120_option0_5average/simulation_s.txt'
# with open(filepath_m, 'r') as f:
#     contents = f.read().strip()
# if contents.endswith(','):
#     contents = contents[:-1]
# contents = f'[{contents}]'
# stds_1 = ast.literal_eval(contents)
# stds_1 = sorted(stds_1, key=lambda x: x[0])

sns.set(style="white",font_scale=2,palette="mako")
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams["figure.figsize"] = (16,10)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plot_inds = [0, 39, 49, 54, 59] 

plt.axvline(2,linestyle=':',linewidth=2,color='grey')

start = 0

plt.plot(alpha_TESTS[start:], np.array([ICL_error(Ctr0, Ctr0, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS[start:]]) - (np.array(data_0[0][1]))[start:], color=color_cycle[2], label = f'Isotropic task structure')
# plt.plot(alpha_TESTS[start:], np.array([ICL_error(Ctr0, Ctr0, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS[start:]]), color=color_cycle[2], label = f'Isotropic task structure')
# plt.plot(alpha_TESTS[start:], [icl_scaled_isotropic(tau, alpha, alpha_test, kappa, rho, 1, 1) for alpha_test in alpha_TESTS[start:]], color='red')
# plt.scatter(alpha_TESTS[start:], (np.array(data_0[0][1]))[start:],  color='#FAFAFA', edgecolors=color_cycle[2],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(alpha_TESTS[start:], (np.array(data_0[0][1]) - np.array(stds_0[0][1]))[start:], (np.array(data_0[0][1]) + np.array(stds_0[0][1]))[start:], color=color_cycle[2], alpha = 0.2)

# plt.plot(alpha_TESTS, [ICL_error(Ctr1, Ctr1, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS], color=color_cycle[3], label = f'[d, d-1, ..., 1] task structure')
# plt.scatter(alpha_TESTS, data_1[0][1],  color='#FAFAFA', edgecolors=color_cycle[3],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(alpha_TESTS, np.array(data_1[0][1]) - np.array(stds_1[0][1]), np.array(data_1[0][1]) + np.array(stds_1[0][1]), color=color_cycle[3], alpha = 0.2)

# plt.plot(alpha_TESTS, [ICL_error(Ctr2, Ctr2, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS], color=color_cycle[4], label = f'Power-law task structure')
# plt.scatter(alpha_TESTS, data_2[0][1],  color='#FAFAFA', edgecolors=color_cycle[4],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(alpha_TESTS, np.array(data_2[0][1]) - np.array(stds_2[0][1]), np.array(data_2[0][1]) + np.array(stds_2[0][1]), color=color_cycle[4], alpha = 0.2)


# plt.subplots_adjust(right=0.75)  # Makes space for legend
# leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
leg = plt.legend()
leg.get_frame().set_alpha(0)
plt.gca().spines['top'].set_color('lightgray')
plt.gca().spines['right'].set_color('lightgray')
plt.gca().spines['bottom'].set_color('lightgray')
plt.gca().spines['left'].set_color('lightgray')
plt.xscale('log')
plt.xlabel(r'Test-time $\alpha$ = $\ell$/d')
# plt.yscale('log')
plt.ylabel('ICL error')
plt.gca().tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig(f'figs/{figurename}.png')



