# -*- coding: utf-8 -*- 

from numpy.random import uniform
from numpy import zeros, array
from tensorflow import constant_initializer

def random_uniform_custom(shape, min_, max_):
    i = uniform(min_, max_, size=shape)
    
    if isinstance(shape, tuple) and len(shape) >= 2:
    	i[0] = zeros(shape[1])
    
    return constant_initializer(i)

# TODO: refactor
def create_prior(label2id_file):
    nodes = []
    num_label = 0

    with open(label2id_file) as f:
        for line in f:
            num_label += 1
            (id,label,freq) = line.strip().split()
            nodes += [label]

    prior = zeros((num_label,len(nodes)))
    
    with open(label2id_file) as f:
        for line in f:
            (id,label,freq) = line.strip().split()
            temp_ =  label.split("/")[1:]
            temp = ["/"+"/".join(temp_[:q+1]) for q in range(len(temp_))]
            code = []
            for i,node in enumerate(nodes):
                if node in temp:
                    code.append(1)
                else:
                    code.append(0)
            prior[int(id),:] = array(code)
    return prior


