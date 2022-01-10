import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import tensorflow as tf

def post_build_ops(dataset, prefetch, batch_size, test = False):
    dataset = dataset.shuffle(2110)
    #Repeat for better training
    #dataset = dataset.repeat(15)
    if test == False:
        for i in range(1,6):
            dataset = dataset.repeat(2)
            dataset = dataset.shuffle(125779 * i)

    return dataset