import jax
from flax.serialization import from_state_dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import optax
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainmini import train
from model.transformer import TransformerConfig
from task.regression_structured import fulltasksampler, finitetasksampler

# finitetasksampler(self, d, l, n, k, rho, Ctask)
# fulltasksampler(self, d, l, n, rho, Ctask)

def spikes(d, index):
    vals = np.zeros(d)
    for j in range(d):
        if j == index:
            vals[j] = d 
        else:
            vals[j] = 0
    return vals

rho = 0.01
d = int(sys.argv[1]);
alpha = 2; l = int(alpha*d);
tau = 4; n = int(tau*(d**2));
kappas = [0.2, 0.5, 1, 2, 10]
h = d;

myname = sys.argv[2] # grab value of $mydir to add results
kappaind = int(sys.argv[3]) # kappa index specified by array
avgind = int(sys.argv[4]) # average index specified by array
kappa = kappas[kappaind]; k = int(kappa*d);

train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d

trainobject = finitetasksampler(d, l, n, k, rho, Ctr)
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=1, n_mlp_layers=1, pure_linear_self_att=False)
state, hist = train(config, data_iter=iter(trainobject), batch_size=n, loss='mse', test_every=1000, train_iters=5000, optim=optax.adamw,lr=1e-4)

print('TRAINING DONE')

loss_func = optax.squared_error
numsamples = 500
testobject = fulltasksampler(d, l, n, rho, Ctr)
tracker = []
for _ in range(numsamples):
    xs, labels = next(testobject); # generates data
    logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
    tracker.append(loss_func(logits, labels).mean())
tracker = np.array(tracker)

print('DONE: TESTING ON PRETRAIN')

file_path = f'./{myname}/test_equals_train_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {np.mean(tracker)}],')
    file_path = f'./{myname}/test_equals_train_s_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {np.std(tracker)}],')

test_powers = np.linspace(train_power - 0.5, train_power + 0.5, 11)
power_test_m = []
power_test_s = []
for test_power in test_powers:
    Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
    testobject = fulltasksampler(d, l, n, rho, Ctest)
    tracker = []
    for _ in range(numsamples):
        xs, labels = next(testobject); # generates data
        logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
        tracker.append(loss_func(logits, labels).mean())
    tracker = np.array(tracker)
    power_test_m.append(np.mean(tracker))
    power_test_s.append(np.std(tracker))

file_path = f'./{myname}/test_powers_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {power_test_m}],')
    file_path = f'./{myname}/test_powers_s_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {power_test_s}],')

print('DONE: TESTING ON POWERS')

signals = np.int64(np.linspace(0,d-1,d))
spike_test_m = []
spike_test_s = []
for signal_index in signals:
    Ctest = np.diag(spikes(d, signal_index)); 
    testobject = fulltasksampler(d, l, n, rho, Ctest)
    tracker = []
    for _ in range(numsamples):
        xs, labels = next(testobject); # generates data
        logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
        tracker.append(loss_func(logits, labels).mean())
    tracker = np.array(tracker)
    spike_test_m.append(np.mean(tracker))
    spike_test_s.append(np.std(tracker))

file_path = f'./{myname}/test_spikes_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {spike_test_m}],')
    file_path = f'./{myname}/test_spikes_s_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {spike_test_s}],')

# file_path = f'./{myname}/pickles/train-{kappaind}-{avgind}.pkl'
# with open(file_path, 'wb') as fp:
#     pickle.dump(hist, fp)