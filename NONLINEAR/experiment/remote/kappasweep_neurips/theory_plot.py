import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
figurename = input("figurename: ")

numavg = 100

kappas = [0.2, 0.5, 1, 2, 10]
rho = 0.01
train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d

test_powers = [0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4] #[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4]
signals = np.int64(np.linspace(0,d-1,d))

sns.set(style="white",font_scale=2,palette="rocket")
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams["figure.figsize"] = (50,10)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, axes = plt.subplots(1, 4, constrained_layout=True) 
same_color = "#07C573"

keys = ['mary', 'F', 'trace', 'cka']
for plotting_index, key in enumerate(keys):
    for i, kappa in enumerate(kappas):
        if key == 'mary':
            alignment_match = ICL_alignment(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
            alignment_spikes = [ICL_alignment(Ctr, np.diag(spikevalue(d, 0.5, sig_index)), tau, alpha, kappa, rho, numavg=100) for sig_index in signals]
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append(ICL_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
        elif key == 'trace':
            alignment_match = (1/d)*np.trace(np.linalg.inv(Ctr)@Ctr)
            alignment_spikes = [(1/d)*np.trace(np.linalg.inv(Ctr)@np.diag(spikevalue(d, 0.5, sig_index))) for sig_index in signals]
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append((1/d)*np.trace(np.linalg.inv(Ctr)@Ctest))
        elif key == 'F':
            alignment_match = resolvent_alignment(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
            alignment_spikes = [resolvent_alignment(Ctr, np.diag(spikevalue(d, 0.5, sig_index)), tau, alpha, kappa, rho, numavg=100) for sig_index in signals]
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append(resolvent_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
        elif key == 'cka':
            alignment_match = cka(d, Ctr, Ctr)/np.sqrt(cka(d, Ctr, Ctr)*cka(d, Ctr, Ctr))
            alignment_spikes = [cka(d, Ctr, np.diag(spikevalue(d, 0.5, sig_index)))/np.sqrt(cka(d, np.diag(spikevalue(d, 0.5, sig_index)), np.diag(spikevalue(d, 0.5, sig_index)))*cka(d, Ctr, Ctr)) for sig_index in signals]
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append(cka(d,Ctr,Ctest)/np.sqrt(cka(d,Ctr,Ctr)*cka(d,Ctest,Ctest)))

        theory_match = ICL_error(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
        theory_spikes = [ICL_error(Ctr, np.diag(spikevalue(d, 0.5, sig_index)), tau, alpha, kappa, rho, numavg=100) for sig_index in signals]
        theory_powers = []
        for test_power in test_powers:
            Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
            theory_powers.append(ICL_error(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
        if i == 0:
            axes[plotting_index].scatter(alignment_powers, theory_powers, marker='o', s=170, color='grey', label = 'Test on powerlaw')
            axes[plotting_index].scatter(alignment_spikes, theory_spikes, marker='d', s=170, color='grey', label = 'Test on spiked signal')
            axes[plotting_index].scatter(alignment_match, theory_match, marker='*', s=400, color=same_color, label = 'Test on pretrain')
        axes[plotting_index].scatter(alignment_spikes, theory_spikes, marker='d', s=170, color=color_cycle[i+1],zorder=i+2)
        for x, y, label in zip(alignment_spikes, theory_spikes, signals):
            axes[plotting_index].text(x, y + 0.02, f'{(label+1):.0f}', color=color_cycle[i+1], fontsize=9, ha='center')
        axes[plotting_index].scatter(alignment_powers, theory_powers, marker='o', s=170, color=color_cycle[i+1],zorder=i+2)
        for x, y, label in zip(alignment_powers, theory_powers, test_powers):
            axes[plotting_index].text(x, y + 0.02, f'{(label-train_power):.1f}', color=color_cycle[i+1], fontsize=9, ha='center')
        axes[plotting_index].scatter(alignment_match, theory_match, marker='*', s=400, color=same_color,zorder=i+3)
    
        concatenated_x = alignment_spikes + alignment_powers + [alignment_match]
        concatenated_y = list(theory_spikes) + list(theory_powers) + [theory_match]
        zipped = list(zip(concatenated_x, concatenated_y))
        sorted_pairs = sorted(zipped, key=lambda pair: pair[0])
        sorted_X, sorted_Y = zip(*sorted_pairs)
        axes[plotting_index].plot(sorted_X,sorted_Y,color = color_cycle[i+1], alpha = 0.5, label =fr"$\kappa = $ {kappa}")

    leg = axes[plotting_index].legend()
    leg.get_frame().set_alpha(0)
    axes[plotting_index].spines['top'].set_color('lightgray')
    axes[plotting_index].spines['right'].set_color('lightgray')
    axes[plotting_index].spines['bottom'].set_color('lightgray')
    axes[plotting_index].spines['left'].set_color('lightgray')

    if key == 'mary':
        axes[plotting_index].set_xlabel(fr"Theory-derived measure $e_{{\mathrm{{align}}}}(C_{{\mathrm{{tr}}}}, C_{{\mathrm{{test}}}})$")
    if key == 'trace':
        axes[plotting_index].set_xlabel(fr"Simple matrix measure $\mathrm{{tr}}[C_{{\mathrm{{test}}}}C_{{\mathrm{{tr}}}}^{{-1}}]$")
    if key == 'F':
        axes[plotting_index].set_xlabel(fr"Resolvent measure $\mathrm{{tr}}[C_{{\mathrm{{test}}}}F]$")
    if key == 'cka':
        axes[plotting_index].set_xlabel(fr"Kernel measure $\mathrm{{CKA}}(C_{{\mathrm{{tr}}}}, C_{{\mathrm{{test}}}})$")
    axes[plotting_index].set_ylabel('ICL error')
    axes[plotting_index].tick_params(axis='both', which='major')

plt.savefig(f'figs/{figurename}.png')
plt.clf()