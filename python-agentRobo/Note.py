# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np

import matplotlib.pyplot as plt

from chainer import cuda, FunctionSet, Variable, optimizers, serializers
from chainer import cuda

print 3

actions = range(9)

def action_to_ah(action):
    a = int(action / 3)
    h = action%3
    return a,h

for i in range(9):
    a,h =  action_to_ah(i)
    print a
    print h
    print "--------"


last_action = np.array(range(3,6))
last_last_action = 2
last_last_last_action = 1

np.r_[last_action/5.0,last_last_action,last_last_last_action]

action = 4

last_last_last_action = last_last_action
last_last_action = last_action
last_action = action


in_size = 227
mean_image = np.load('ilsvrc_2012_mean.npy')
mean_image.shape

cropwidth = 256-in_size
start = cropwidth // 2
stop = start + in_size
mean_image2 = mean_image[:, start:stop, start:stop].copy()
mean_image2.shape
start
stop
4.3//2 # 切り捨て除算
