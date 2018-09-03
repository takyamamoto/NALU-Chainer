# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:52:49 2018

@author: yamamoto
"""

import chainer
from chainer import Parameter
from chainer import initializers
import chainer.functions as F
from nac import NAC 

class NALU(chainer.Chain):
    """Basic NALU unit implementation 
    from https://arxiv.org/pdf/1808.00508.pdf

    >>> x = np.array([[0, 1, 2, 3, 4]], np.float32)
    >>> nalu = NALU(5, 10)
    >>> y = nalu(x)
    >>> print(y.shape)
    (1, 10)
    >>> print(nalu.G.shape)
    (10, 5)
    """
    def __init__(self, in_shape, out_shape, nobias=False, initial_bias=None):
        """
        in_shape: input sample dimension
        out_shape: output sample dimension
        """
        super(NALU, self).__init__()
        with self.init_scope():
            self.in_shape = in_shape
            self.out_shape = out_shape
            
            G_initializer = initializers.GlorotUniform()
            self.G = Parameter(G_initializer, (out_shape, in_shape))
            self.nac = NAC(in_shape, out_shape)
            self.eps = 1e-5
            
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = Parameter(bias_initializer, out_shape)

    def __call__(self, x):   
        a = self.nac(x)
        g = F.sigmoid(F.linear(x, self.G, self.b))
        ag = g * a
        log_in = F.log(abs(x) + self.eps)
        m = F.exp(self.nac(log_in))
        md = (1 - g) * m
        return ag + md