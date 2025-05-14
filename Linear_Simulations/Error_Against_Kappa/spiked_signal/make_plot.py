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
filepath_m = input("filepath_m: ")
# filepath_s = input("filepath_s: ")
figurename = input("figurename: ")
xaxis = input("xaxis: ")
kappas = np.linspace(0.2,2.2,21)
signals = np.int64(np.linspace(0,d-1,d//2))
rho = 0.01
Ctr = np.diag([i for i in range(1,d+1)]); Ctr = (d/np.trace(Ctr))*Ctr

with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
data = ast.literal_eval(contents)
data = sorted(data, key=lambda x: x[0])

if xaxis == 'signal':
    kappa_plot = [0,4,8,12,20]
    align = []
    for signal_index in signals:
        Ctest = np.diag(spikevalue(d,0,signal_index))
        align.append((1/d)*np.trace(Ctest@Ctr))
    sns.set(style="white",font_scale=1,palette="mako")
    for kappa_ind in kappa_plot:
        plt.plot(align, ICL_for_spiked_test(Ctr, signals, 0, tau, alpha, kappas[kappa_ind], rho, numavg=100), label = rf'$\kappa = $ {kappas[kappa_ind]:.3f}')
        plt.scatter(align, data[kappa_ind][1])
    plt.legend()
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams["figure.figsize"] = (6,5)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.gca().spines['top'].set_color('lightgray')
    plt.gca().spines['right'].set_color('lightgray')
    plt.gca().spines['bottom'].set_color('lightgray')
    plt.gca().spines['left'].set_color('lightgray')
    plt.xlabel('Pretrain-test alignment')
    plt.ylabel('ICL error')
    plt.savefig(figurename)

else:
    sns.set(style="white",font_scale=1,palette="mako")
    plot_inds = np.int64(np.linspace(0,len(signals)-1,6))
    for plot_ind in plot_inds:
        plt.plot(kappas, [ICL_error(Ctr, np.diag(spikevalue(d,0,signals[plot_ind])), tau, alpha, kappa, rho, 100) for kappa in kappas], label = f'Test signal = {signals[plot_ind]}')
        plt.scatter(kappas, [item[1][plot_ind] for item in data])
    plt.plot(kappas, [ICL_error(Ctr, Ctr, tau, alpha, kappa, rho, 100) for kappa in kappas], color='red', label = f'Test on pretrain')
    plt.legend()
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams["figure.figsize"] = (6,5)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.gca().spines['top'].set_color('lightgray')
    plt.gca().spines['right'].set_color('lightgray')
    plt.gca().spines['bottom'].set_color('lightgray')
    plt.gca().spines['left'].set_color('lightgray')
    plt.xlabel('Task diversity')
    plt.ylabel('ICL error')
    plt.savefig(figurename)



