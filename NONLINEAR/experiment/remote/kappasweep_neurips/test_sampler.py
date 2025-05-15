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

d = 50
l = 50
n = 1000

train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d

testobject = fulltasksampler(d, l, n, 0.01, Ctr)
xs, labels = next(testobject)
print(xs.shape)
print(labels.shape)