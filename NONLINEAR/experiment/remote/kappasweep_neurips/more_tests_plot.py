import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
numavg = int(input("numavg: "))
experiment = input("experiment: ")
figurename = input("figurename: ")

kappas = [0.2, 0.5, 1, 2, 10]
rho = 0.01
train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d


reads = []
for i in range(numavg):
    filepath_m = f'runs/{experiment}/test_equals_train_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_equals_train_m = np.mean(np.array(reads),axis=0)
print('shape train', test_equals_train_m.shape)
test_equals_train_s = np.std(np.array(reads),axis=0)

reads = []
for i in range(numavg):
    filepath_m = f'runs/{experiment}/test_powers_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_on_powers_m = np.mean(np.array(reads),axis=0)
test_on_powers_s = np.std(np.array(reads),axis=0)
print('shape powers', test_on_powers_m.shape)
# This will be 2d with shape num(kappas) x num(powers)

reads = []
for i in range(numavg):
    filepath_m = f'runs/{experiment}/test_spikes_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_on_spikes_m = np.mean(np.array(reads),axis=0)
test_on_spikes_s = np.std(np.array(reads),axis=0)
print('shape spikes', test_on_spikes_m.shape)
# This will be 2d with shape num(kappas) x num(spikes)

reads = []
for i in range(numavg):
    filepath_m = f'runs/{experiment}/random_cov_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_on_randomcov_m = np.mean(np.array(reads),axis=0)
test_on_randomcov_s = np.std(np.array(reads),axis=0)
print('shape randomcov', test_on_randomcov_m.shape)
# This will be 2d with shape num(kappas) x 2

reads = []
for i in range(numavg):
    filepath_m = f'runs/{experiment}/yuespike_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_on_rankone_m = np.mean(np.array(reads),axis=0)
test_on_rankone_s = np.std(np.array(reads),axis=0)
print('yue spikes', test_on_rankone_m.shape)
# This will be 2d with shape num(kappas) x 2

# test_powers = np.linspace(train_power - 0.5, train_power + 0.5, 11)
# test_powers = [0.05,0.1,0.2,0.4,0.6,0.8,1,1.2]
test_powers = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4]
experiment_d = d
signals = np.int64(np.linspace(0,experiment_d-1,experiment_d))

randomcov1 = [[ 0.83237078, -0.07909702, -0.27739007, -0.19274271,  0.0656331 ,
         0.46741238, -0.19483789,  0.02056367, -0.21153396, -0.47935422],
       [-0.07909702,  0.5363103 , -0.05777238,  0.2062389 , -0.31569435,
        -0.11381103,  0.26002582, -0.14846191,  0.14098197,  0.69347973],
       [-0.27739007, -0.05777238,  0.68543547, -0.19082719,  0.22725912,
        -0.44997642,  0.51621705,  0.17975818,  0.12142992, -0.09894243],
       [-0.19274271,  0.2062389 , -0.19082719,  0.8873404 , -0.53952737,
        -0.01731048,  0.01360022,  0.51452076,  0.10604015,  0.90559764],
       [ 0.0656331 , -0.31569435,  0.22725912, -0.53952737,  0.64833692,
         0.06366982,  0.12486512, -0.1687741 , -0.17977651, -0.89871943],
       [ 0.46741238, -0.11381103, -0.44997642, -0.01731048,  0.06366982,
         0.73127793, -0.09141847,  0.26434272, -0.15774903, -0.15658172],
       [-0.19483789,  0.26002582,  0.51621705,  0.01360022,  0.12486512,
        -0.09141847,  1.68647604,  0.52695443, -0.19347475,  0.40341816],
       [ 0.02056367, -0.14846191,  0.17975818,  0.51452076, -0.1687741 ,
         0.26434272,  0.52695443,  1.34403112, -0.02766983,  0.22429092],
       [-0.21153396,  0.14098197,  0.12142992,  0.10604015, -0.17977651,
        -0.15774903, -0.19347475, -0.02766983,  0.51622984,  0.60996857],
       [-0.47935422,  0.69347973, -0.09894243,  0.90559764, -0.89871943,
        -0.15658172,  0.40341816,  0.22429092,  0.60996857,  2.1321912 ]]
randomcov2 = [[ 0.51025558,  0.33055659, -0.08249114, -0.05412674,  0.17032635,
        -0.08733811,  0.51211055, -0.27687922,  0.3346133 , -0.00972467],
       [ 0.33055659,  1.09246634,  0.08558953, -0.25151201,  0.26700021,
        -0.2466319 ,  0.53765215, -0.48282689,  0.07862001, -0.4464572 ],
       [-0.08249114,  0.08558953,  0.29423188,  0.06586987, -0.33516675,
         0.14550537,  0.19025169,  0.16201828,  0.03059837,  0.01215256],
       [-0.05412674, -0.25151201,  0.06586987,  1.937969  ,  0.4062466 ,
        -0.03540329,  0.3684063 , -0.60573562, -0.27364811,  0.15528306],
       [ 0.17032635,  0.26700021, -0.33516675,  0.4062466 ,  0.93602587,
        -0.62939035, -0.04482296, -0.28048186,  0.00576536, -0.29737925],
       [-0.08733811, -0.2466319 ,  0.14550537, -0.03540329, -0.62939035,
         0.88219179, -0.15558   , -0.31852878, -0.00371844,  0.26805076],
       [ 0.51211055,  0.53765215,  0.19025169,  0.3684063 , -0.04482296,
        -0.15558   ,  1.40934753, -0.03243841,  0.03097946, -0.14072824],
       [-0.27687922, -0.48282689,  0.16201828, -0.60573562, -0.28048186,
        -0.31852878, -0.03243841,  1.67390912,  0.06370284,  0.05888919],
       [ 0.3346133 ,  0.07862001,  0.03059837, -0.27364811,  0.00576536,
        -0.00371844,  0.03097946,  0.06370284,  0.68217878, -0.03994515],
       [-0.00972467, -0.4464572 ,  0.01215256,  0.15528306, -0.29737925,
         0.26805076, -0.14072824,  0.05888919, -0.03994515,  0.5814241 ]]

u1 = [-0.28346192, -0.18501277,  0.08458142, -0.27086306, -0.03163529,
       -0.64428289, -0.54159727, -0.10366208,  0.28735085,  0.04640711]
u2 = [ 0.5421623 , -0.00124371, -0.16994279, -0.37308776,  0.40306148,
       -0.31057368, -0.28910356,  0.33269666,  0.21158544,  0.2000814 ]
C_yue_1 = (np.eye(d) + 4*d*np.outer(u1,u1))*d/(d+4*d)
C_yue_2 = (np.eye(d) + 5*np.outer(u2,u2))*d/(d+5)

sns.set(style="white",font_scale=2,palette="rocket")
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams["figure.figsize"] = (16,10)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
same_color = "#07C573"

keys = ['mary', 'trace', 'F']
for key in keys:
    for i, kappa in enumerate(kappas):
        if key == 'mary':
            alignment_match = ICL_alignment(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
            alignment_spikes = [ICL_alignment(Ctr, np.diag(spikevalue(d, 0.5, sig_index)), tau, alpha, kappa, rho, numavg=100) for sig_index in signals]
            alignment_randomcov = [ICL_alignment(Ctr, randomcov1, tau, alpha, kappa, rho, numavg=100), ICL_alignment(Ctr, randomcov2, tau, alpha, kappa, rho, numavg=100)]
            alignment_yue = [ICL_alignment(Ctr, C_yue_1, tau, alpha, kappa, rho, numavg=100), ICL_alignment(Ctr, C_yue_2, tau, alpha, kappa, rho, numavg=100)]
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append(ICL_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
        elif key == 'trace':
            alignment_match = (1/d)*np.trace(np.linalg.inv(Ctr)@Ctr)
            alignment_spikes = [(1/d)*np.trace(np.linalg.inv(Ctr)@np.diag(spikevalue(d, 0.5, sig_index))) for sig_index in signals]
            alignment_randomcov = [(1/d)*np.trace(np.linalg.inv(Ctr)@randomcov1), (1/d)*np.trace(np.linalg.inv(Ctr)@randomcov2)]
            alignment_yue = [(1/d)*np.trace(np.linalg.inv(Ctr)@C_yue_1), (1/d)*np.trace(np.linalg.inv(Ctr)@C_yue_2)]
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append((1/d)*np.trace(np.linalg.inv(Ctr)@Ctest))
        elif key == 'F':
            alignment_match = resolvent_alignment(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
            alignment_spikes = [resolvent_alignment(Ctr, np.diag(spikevalue(d, 0.5, sig_index)), tau, alpha, kappa, rho, numavg=100) for sig_index in signals]
            alignment_randomcov = [resolvent_alignment(Ctr, randomcov1, tau, alpha, kappa, rho, numavg=100), resolvent_alignment(Ctr, randomcov2, tau, alpha, kappa, rho, numavg=100)]
            alignment_yue = [resolvent_alignment(Ctr, C_yue_1, tau, alpha, kappa, rho, numavg=100), resolvent_alignment(Ctr, C_yue_2, tau, alpha, kappa, rho, numavg=100)]
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append(resolvent_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
        if i == 0:
            plt.scatter(alignment_powers, test_on_powers_m[i,:], marker='o', s=100, color='grey', label = 'Test on powerlaw')
            plt.scatter(alignment_spikes, test_on_spikes_m[i,:], marker='d', s=100, color='grey', label = 'Test on spiked signal')
            plt.scatter(alignment_randomcov, test_on_randomcov_m[i,:], marker="$R$", s=100, color='grey', label='Test on random covariance')
            plt.scatter(alignment_yue, test_on_rankone_m[i,:], marker="$1$", s=100, color='grey', label='Test on isotropic perturbation')
            plt.scatter(alignment_match, test_equals_train_m[i], marker='*', s=300, color=same_color, label = 'Test on pretrain')

        concatenated_x = alignment_spikes + alignment_powers + alignment_randomcov + alignment_yue + [alignment_match]
        concatenated_y = list(test_on_spikes_m[i,:]) + list(test_on_powers_m[i,:]) + list(test_on_randomcov_m[i,:]) + list(test_on_rankone_m[i,:]) + [test_equals_train_m[i]]
        zipped = list(zip(concatenated_x, concatenated_y))
        sorted_pairs = sorted(zipped, key=lambda pair: pair[0])
        sorted_X, sorted_Y = zip(*sorted_pairs)
        plt.plot(sorted_X,sorted_Y,color = color_cycle[i+1], alpha = 0.5, label =fr"$\kappa = $ {kappa}")

        plt.scatter(alignment_spikes, test_on_spikes_m[i,:], marker='d', s=100, color=color_cycle[i+1], zorder = i+1)
        for x, y, label in zip(alignment_spikes, test_on_spikes_m[i,:], signals):
            plt.text(x, y + 0.02, f'{(label+1):.0f}', color=color_cycle[i+1], fontsize=9, ha='center')
        plt.scatter(alignment_powers, test_on_powers_m[i,:], marker='o', s=100, color=color_cycle[i+1], zorder = i+1)
        for x, y, label in zip(alignment_powers, test_on_powers_m[i,:], test_powers):
            plt.text(x, y + 0.02, f'{(label-train_power):.1f}', color=color_cycle[i+1], fontsize=9, ha='center')
        plt.scatter(alignment_randomcov, test_on_randomcov_m[i,:], marker="$R$", s=100, color=color_cycle[i+1], zorder = i+1)
        for x, y, label in zip(alignment_randomcov, test_on_randomcov_m[i,:], ['C1','C2']):
            plt.text(x, y + 0.02, label, color=color_cycle[i+1], fontsize=9, ha='center')
        plt.scatter(alignment_yue, test_on_rankone_m[i,:], marker="$1$", s=100, color=color_cycle[i+1], zorder = i+1)
        for x, y, label in zip(alignment_yue, test_on_rankone_m[i,:], ['u1','u2']):
            plt.text(x, y + 0.02, label, color=color_cycle[i+1], fontsize=9, ha='center')
        plt.scatter(alignment_match, test_equals_train_m[i], marker='*', s=300, color=same_color, zorder = i+1)

    # plt.subplots_adjust(right=0.75)  # Makes space for legend
    # leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    leg = plt.legend()
    leg.get_frame().set_alpha(0)
    plt.gca().spines['top'].set_color('lightgray')
    plt.gca().spines['right'].set_color('lightgray')
    plt.gca().spines['bottom'].set_color('lightgray')
    plt.gca().spines['left'].set_color('lightgray')
    if key == 'mary':
        plt.xlabel(fr"Theoretical alignment measure $e_{{\mathrm{{align}}}}$")
    if key == 'trace':
        plt.xlabel(fr"Trace alignment measure $\mathrm{{tr}}[\mathrm{{Ctest}} \mathrm{{Ctr}}^{{-1}}]$")
    if key == 'F':
        plt.xlabel(fr"Resolvent alignment measure $\mathrm{{tr}}[\mathrm{{Ctest}} F_{{\mathrm{{Ctr}}}}]$")
    plt.ylabel('ICL error')
    plt.gca().tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.savefig(f'figs/{figurename}_{key}.png')
    plt.clf()