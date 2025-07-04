import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
numavg = int(input("numavg: "))
figurename = input("figurename: ")

kappas = [0.2, 0.5, 1, 2, 10]
rho = 0.01
train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d

test_powers = np.linspace(train_power - 0.5, train_power + 0.5, 11)
experiment_d = 10
signals = np.int64(np.linspace(0,experiment_d-1,experiment_d))

sns.set(style="white",font_scale=2,palette="mako")
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams["figure.figsize"] = (16,10)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, kappa in enumerate(kappas):
    alignment_match = ICL_alignment(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
    theory_match = ICL_error(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
    alignment_spikes = [ICL_alignment(Ctr, np.diag(spikevalue(d, 0.5, sig_index)), tau, alpha, kappa, rho, numavg=100) for sig_index in signals]
    theory_spikes = [ICL_error(Ctr, np.diag(spikevalue(d, 0.5, sig_index)), tau, alpha, kappa, rho, numavg=100) for sig_index in signals]
    alignment_powers = []; theory_powers = []
    for test_power in test_powers:
        Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
        alignment_powers.append(ICL_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
        theory_powers.append(ICL_error(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
    if i == 0:
        plt.scatter(alignment_powers, theory_powers, marker='o', s=170, color='grey', label = 'Test on powerlaw')
        plt.scatter(alignment_spikes, theory_spikes, marker='d', s=170, color='grey', label = 'Test on spiked signal')
        plt.scatter(alignment_match, theory_match, marker='*', s=400, color='red', label = 'Test on pretrain')
    plt.scatter(alignment_spikes, theory_spikes, marker='d', s=170, color=color_cycle[i+1])
    # plt.errorbar(alignment_spikes, test_on_spikes_m[i,:], test_on_spikes_s[i,:])
    plt.scatter(alignment_powers, theory_powers, marker='o', s=170, color=color_cycle[i+1],  label =fr"$\kappa = $ {kappa}")
    # plt.errorbar(alignment_powers, test_on_powers_m[i,:], test_on_powers_s[i,:])
    plt.scatter(alignment_match, theory_match, marker='*', s=400, color='red')
    # plt.errorbar(alignment_match, test_equals_train_m[i], test_equals_train_s[i])

# plt.subplots_adjust(right=0.75)  # Makes space for legend
# leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
leg = plt.legend()
leg.get_frame().set_alpha(0)
plt.gca().spines['top'].set_color('lightgray')
plt.gca().spines['right'].set_color('lightgray')
plt.gca().spines['bottom'].set_color('lightgray')
plt.gca().spines['left'].set_color('lightgray')
plt.xlabel(fr"Theoretical alignment measure $e_{{\mathrm{{align}}}}$")
plt.ylabel('ICL error')
plt.gca().tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig(f'figs/{figurename}.png')