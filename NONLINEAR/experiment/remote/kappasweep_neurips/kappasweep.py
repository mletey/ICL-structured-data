import numpy as np
import optax
from theory import *
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainmini import train
from model.transformer import TransformerConfig
from task.regression_structured import fulltasksampler, finitetasksampler

rho = 0.01
d = int(sys.argv[1]);
alpha = 0.5; l = int(alpha*d);
tau = 4; n = int(tau*(d**2));
kappas = [0.2, 0.5, 1, 2, 10]
h = d+1;

myname = sys.argv[2] # grab value of $mydir to add results
kappaind = int(sys.argv[3]) # kappa index specified by array
avgind = int(sys.argv[4]) # average index specified by array
kappa = kappas[kappaind]; k = int(kappa*d);

train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d
#Ctr = np.eye(d)

trainobject = finitetasksampler(d, l, n, k, rho, Ctr)
testobject_1 = fulltasksampler(d, l, n, rho, Ctr)
testobject_2 = fulltasksampler(d, l, n, rho, np.diag(spikevalue(d, 0.5, 5)))

config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=1, n_mlp_layers=0, pure_linear_self_att=False)
state, hist = train(config, data_iter=iter(trainobject), test_1_iter=iter(testobject_1), test_2_iter=iter(testobject_2), batch_size=16, loss='mse', test_every=100, train_iters=1000, optim=optax.adamw,lr=1e-4)

print('TRAINING DONE',flush=True)
file_path = f'./{myname}/pickles/train-{kappaind}-{avgind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)

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

# test_powers = np.linspace(train_power - 0.5, train_power + 0.5, 11)
# test_powers = [0.05,0.1,0.2,0.4,0.6,0.8,1,1.2]
test_powers = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4]
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
    Ctest = np.diag(spikevalue(d, 0.5, signal_index)); 
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

print('DONE: TESTING ON SPIKES')

# randomcov1 = [[ 0.83237078, -0.07909702, -0.27739007, -0.19274271,  0.0656331 ,
#          0.46741238, -0.19483789,  0.02056367, -0.21153396, -0.47935422],
#        [-0.07909702,  0.5363103 , -0.05777238,  0.2062389 , -0.31569435,
#         -0.11381103,  0.26002582, -0.14846191,  0.14098197,  0.69347973],
#        [-0.27739007, -0.05777238,  0.68543547, -0.19082719,  0.22725912,
#         -0.44997642,  0.51621705,  0.17975818,  0.12142992, -0.09894243],
#        [-0.19274271,  0.2062389 , -0.19082719,  0.8873404 , -0.53952737,
#         -0.01731048,  0.01360022,  0.51452076,  0.10604015,  0.90559764],
#        [ 0.0656331 , -0.31569435,  0.22725912, -0.53952737,  0.64833692,
#          0.06366982,  0.12486512, -0.1687741 , -0.17977651, -0.89871943],
#        [ 0.46741238, -0.11381103, -0.44997642, -0.01731048,  0.06366982,
#          0.73127793, -0.09141847,  0.26434272, -0.15774903, -0.15658172],
#        [-0.19483789,  0.26002582,  0.51621705,  0.01360022,  0.12486512,
#         -0.09141847,  1.68647604,  0.52695443, -0.19347475,  0.40341816],
#        [ 0.02056367, -0.14846191,  0.17975818,  0.51452076, -0.1687741 ,
#          0.26434272,  0.52695443,  1.34403112, -0.02766983,  0.22429092],
#        [-0.21153396,  0.14098197,  0.12142992,  0.10604015, -0.17977651,
#         -0.15774903, -0.19347475, -0.02766983,  0.51622984,  0.60996857],
#        [-0.47935422,  0.69347973, -0.09894243,  0.90559764, -0.89871943,
#         -0.15658172,  0.40341816,  0.22429092,  0.60996857,  2.1321912 ]]
# randomcov2 = [[ 0.51025558,  0.33055659, -0.08249114, -0.05412674,  0.17032635,
#         -0.08733811,  0.51211055, -0.27687922,  0.3346133 , -0.00972467],
#        [ 0.33055659,  1.09246634,  0.08558953, -0.25151201,  0.26700021,
#         -0.2466319 ,  0.53765215, -0.48282689,  0.07862001, -0.4464572 ],
#        [-0.08249114,  0.08558953,  0.29423188,  0.06586987, -0.33516675,
#          0.14550537,  0.19025169,  0.16201828,  0.03059837,  0.01215256],
#        [-0.05412674, -0.25151201,  0.06586987,  1.937969  ,  0.4062466 ,
#         -0.03540329,  0.3684063 , -0.60573562, -0.27364811,  0.15528306],
#        [ 0.17032635,  0.26700021, -0.33516675,  0.4062466 ,  0.93602587,
#         -0.62939035, -0.04482296, -0.28048186,  0.00576536, -0.29737925],
#        [-0.08733811, -0.2466319 ,  0.14550537, -0.03540329, -0.62939035,
#          0.88219179, -0.15558   , -0.31852878, -0.00371844,  0.26805076],
#        [ 0.51211055,  0.53765215,  0.19025169,  0.3684063 , -0.04482296,
#         -0.15558   ,  1.40934753, -0.03243841,  0.03097946, -0.14072824],
#        [-0.27687922, -0.48282689,  0.16201828, -0.60573562, -0.28048186,
#         -0.31852878, -0.03243841,  1.67390912,  0.06370284,  0.05888919],
#        [ 0.3346133 ,  0.07862001,  0.03059837, -0.27364811,  0.00576536,
#         -0.00371844,  0.03097946,  0.06370284,  0.68217878, -0.03994515],
#        [-0.00972467, -0.4464572 ,  0.01215256,  0.15528306, -0.29737925,
#          0.26805076, -0.14072824,  0.05888919, -0.03994515,  0.5814241 ]]
# randomcovs_m = []
# randomcovs_s = []
# testobject = fulltasksampler(d, l, n, rho, randomcov1)
# tracker = []
# for _ in range(numsamples):
#     xs, labels = next(testobject); # generates data
#     logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
#     tracker.append(loss_func(logits, labels).mean())
# tracker = np.array(tracker)
# randomcovs_m.append(np.mean(tracker))
# randomcovs_s.append(np.std(tracker))
# testobject = fulltasksampler(d, l, n, rho, randomcov2)
# tracker = []
# for _ in range(numsamples):
#     xs, labels = next(testobject); # generates data
#     logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
#     tracker.append(loss_func(logits, labels).mean())
# tracker = np.array(tracker)
# randomcovs_m.append(np.mean(tracker))
# randomcovs_s.append(np.std(tracker))

# file_path = f'./{myname}/random_cov_m_{avgind}.txt'
# with open(file_path, 'a') as file:
#     file.write(f'[{kappaind}, {randomcovs_m}],')
#     file_path = f'./{myname}/random_cov_s_{avgind}.txt'
# with open(file_path, 'a') as file:
#     file.write(f'[{kappaind}, {randomcovs_s}],')

# print('DONE: TESTING ON RANDOM COVARIANCE MATRICES')

# u1 = [-0.28346192, -0.18501277,  0.08458142, -0.27086306, -0.03163529,
#        -0.64428289, -0.54159727, -0.10366208,  0.28735085,  0.04640711]
# u2 = [ 0.5421623 , -0.00124371, -0.16994279, -0.37308776,  0.40306148,
#        -0.31057368, -0.28910356,  0.33269666,  0.21158544,  0.2000814 ]
# C_yue_1 = (np.eye(d) + 4*d*np.outer(u1,u1))*d/(d+4*d)
# C_yue_2 = (np.eye(d) + 5*np.outer(u2,u2))*d/(d+5)

# yuespike_m = []
# yuespike_s = []
# testobject = fulltasksampler(d, l, n, rho, C_yue_1)
# tracker = []
# for _ in range(numsamples):
#     xs, labels = next(testobject); # generates data
#     logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
#     tracker.append(loss_func(logits, labels).mean())
# tracker = np.array(tracker)
# yuespike_m.append(np.mean(tracker))
# yuespike_s.append(np.std(tracker))
# testobject = fulltasksampler(d, l, n, rho, C_yue_2)
# tracker = []
# for _ in range(numsamples):
#     xs, labels = next(testobject); # generates data
#     logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
#     tracker.append(loss_func(logits, labels).mean())
# tracker = np.array(tracker)
# yuespike_m.append(np.mean(tracker))
# yuespike_s.append(np.std(tracker))

# file_path = f'./{myname}/yuespike_m_{avgind}.txt'
# with open(file_path, 'a') as file:
#     file.write(f'[{kappaind}, {yuespike_m}],')
#     file_path = f'./{myname}/yuespike_s_{avgind}.txt'
# with open(file_path, 'a') as file:
#     file.write(f'[{kappaind}, {yuespike_s}],')

# print('DONE: TESTING ON RANK1 YUE SPIKE MATRCIES')