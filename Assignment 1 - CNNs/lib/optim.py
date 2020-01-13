from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        raise ValueError("Not Implemented Error")


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def step(self):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                try:
                    self.velocity[n] = self.momentum * self.velocity[n] - self.lr * layer.grads[n]
                    layer.params[n] += self.velocity[n]
                except:
                    self.velocity[n] = np.zeros(layer.params[n].shape)
                    self.velocity[n] = self.momentum * self.velocity[n] - self.lr * layer.grads[n]
                    layer.params[n] += self.velocity[n]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def step(self):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        for layer in self.net.layers:
            for n, v in layer.grads.items():
                try:
                    self.cache[n] = (self.decay*self.cache[n])+((1.0-self.decay)*np.square(layer.grads[n]))
                    layer.params[n] -= self.lr*layer.grads[n]/np.sqrt(self.cache[n]+self.eps)
                except:
                    self.cache[n] = np.zeros(layer.params[n].shape)
                    self.cache[n] = (self.decay*self.cache[n])+((1.0-self.decay)*np.square(layer.grads[n]))
                    layer.params[n] -= self.lr*layer.grads[n]/np.sqrt(self.cache[n]+self.eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def step(self):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        for layer in self.net.layers:
            for n, v in layer.grads.items():
                if n not in self.vt.keys():
                    self.vt[n] = np.zeros(layer.params[n].shape)
                if n not in self.mt.keys():
                    self.mt[n] = np.zeros(layer.params[n].shape)
                self.t = self.t+1
                self.mt[n]=self.beta1*self.mt[n]+(1.0-self.beta1)*layer.grads[n]
                self.vt[n]=self.beta2*self.vt[n]+(1.0-self.beta2)*np.square(layer.grads[n])
                m_t_hat = self.mt[n] / (1. - self.beta1**(self.t))
                v_t_hat = self.vt[n] / (1. - self.beta2**(self.t))
                layer.params[n] -= self.lr*m_t_hat/(np.sqrt(v_t_hat)+self.eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
