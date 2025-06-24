import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

from theory import *
sys.path.append('../../')
sys.path.append('../../../')
from common import *

mydir = sys.argv[1]
myfile = sys.argv[2]

trainvals = []
idgvals = []
iclvals = []
file_path = f'runs/{mydir}/pickles/{myfile}.pkl'
with open(file_path, 'rb') as fp:
    loaded = pickle.load(fp)
trainloss = [Metrics.loss for Metrics in loaded['train']]
idgloss = [Metrics.loss for Metrics in loaded['test1']]
iclloss = [Metrics.loss for Metrics in loaded['test2']]
for loss_array in trainloss:
    trainvals.append(loss_array.item())
for loss_array in idgloss:
    idgvals.append(loss_array.item())
for loss_array in iclloss:
    iclvals.append(loss_array.item())
trainvals=np.array(trainvals)
idgvals=np.array(idgvals)
iclvals=np.array(iclvals)

plt.plot(range(len(trainvals)),trainvals,label='Train')
plt.plot(range(len(idgvals)),idgvals,label='Test on train')
plt.plot(range(len(iclvals)),iclvals,label='Test on medium spike')
plt.title(f'{myfile} training error dynamics')
plt.legend()
plt.savefig(f'runs/{mydir}/pickles/{myfile}plot.png')