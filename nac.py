# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:08:55 2018

@author: yamamoto
"""

import chainer
from chainer import Parameter
from chainer import initializers
import chainer.functions as F

class NAC(chainer.Chain):
    """Basic NAC unit implementation 
    from https://arxiv.org/pdf/1808.00508.pdf

    >>> x = np.array([[0, 1, 2, 3, 4]], np.float32)
    >>> nac = NAC(5, 10)
    >>> y = nac(x)
    >>> print(y.shape)
    (1, 10)
    >>> print(nac.W_.shape)
    (10, 5)
    """
    def __init__(self, in_shape, out_shape, nobias=False, initial_bias=None):
        """
        in_shape: input sample dimension
        out_shape: output sample dimension
        """
        super(NAC, self).__init__()
        with self.init_scope():
            self.in_shape = in_shape
            self.out_shape = out_shape
            
            W_initializer = initializers.GlorotUniform()
            self.W_ = Parameter(W_initializer, (out_shape, in_shape))
            M_initializer = initializers.GlorotUniform()
            self.M_ = Parameter(M_initializer, (out_shape, in_shape))
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = Parameter(bias_initializer, out_shape)

    def __call__(self, x):         
        W = F.tanh(self.W_) * F.sigmoid(self.M_)
        return F.linear(x, W, self.b)