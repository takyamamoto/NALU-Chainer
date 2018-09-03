# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 15:57:07 2018

@author: yamamoto
"""

import chainer
import chainer.functions as F
from chainer import cuda

from chainer import reporter

xp = cuda.cupy

from nalu import NALU

# Network definition
class NaluLayer(chainer.Chain):
    def __init__(self, n_in=2, n_out=1, return_prediction=False):
        super(NaluLayer, self).__init__()
        with self.init_scope():
            self.nalu = NALU(n_in, n_out)
            self.return_prediction = return_prediction

    def __call__(self, x):
        # Split input
        x_in = x[:,:2]
        x_out = x[:,2:]
        
        prediction = F.relu(self.nalu(x_in))
        
        loss = F.mean_squared_error(prediction, x_out)
        reporter.report({'loss': loss}, self)

        if self.return_prediction == True:
            return loss, prediction
        else:
            return loss