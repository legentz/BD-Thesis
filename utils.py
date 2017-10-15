# -*- coding: utf-8 -*- 

from numpy.random import uniform
from numpy import zeros
from tensorflow import constant_initializer

def random_uniform_custom(shape, min_, max_):
    i = uniform(min_, max_, size=shape)
    
    if isinstance(shape, tuple) and len(shape) >= 2:
    	i[0] = zeros(shape[1])
    
    return constant_initializer(i)