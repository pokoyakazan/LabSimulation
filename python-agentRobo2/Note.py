# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np

import matplotlib.pyplot as plt

from chainer import cuda, FunctionSet, Variable, optimizers, serializers
from chainer import cuda

actions = range(14)
actions
