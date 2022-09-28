#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:46:31 2022

@author: va18024
"""

import os 
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt     
import sys
# =============================================================================
# Set path and parameters to load data 
# =============================================================================
datapath = 'path to data'
date_results = 'folder to results'
#set parameters to create filename
nseeds = 10
models = ['cRNN', 'ccRNN']
input_D = '1'
nhid = 50
bptt = 1
synth_nhid =400
metric=0
vecname = '_nseeds-' + str(nseeds) + "_nhid-" + str(nhid) + "_synthnhid-" + str(synth_nhid) + "_Inpsize-" + str(input_D) + "_T-" +str(bptt) + "_.npy"
# =============================================================================
# figure settings
# =============================================================================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0, 0, 1, 1])
colours =[(0.5,0.5,0.5),(249/256, 170/256, 56/256), (91/256, 206/256, 173/256)]
# =============================================================================
# Function to load data 
# =============================================================================
def import_scores(data_path, date_results, vecname, bptt, nhid, models, synth_nhid):
    if models in ['cRNN']:
        model ='LSTM'
        vecname =  model + "_" + vecname 
        mypath = join(datapath, date_results)
        print(mypath)
    else:
        model ='DNI_LSTM'
        vecname =  model + "_" +vecname 
        nfibres = nhid*2
        suff = "_nfibres-{}_.npy".format(nfibres)
        vecname = vecname.replace('_.npy', suff)
        mypath = join(datapath, date_results)
        print(mypath)
    if spars_int is not None:
        vecname = "sparsintnew-" + str(spars_int)  + "_"  +vecname 
    vecname = vecname.replace('args.npy', 'scores' + '.npy')
    scores=np.load(join(mypath, vecname))
    return scores
# =============================================================================
# Function to plot learning curves for linedrawing
# =============================================================================
def plot_learning(ax):
    scores =[]
    metric =0
    for j, model_t in enumerate(models):
        scores = import_scores(datapath, date_results, vecname, bptt, nhid, model_t, synth_nhid)
        epochs=scores.shape[2]
        mean = np.mean(scores[:, metric,:], axis=0)# synthid layers, sparsity, truncations, models, metric, epochs
        ebar = np.std(scores[:, metric,:], axis =0) / np.sqrt(nseeds)
        ax.plot(range(epochs), mean, color=colours[j],label=model_t) # plot variance 
        ax.fill_between(range(epochs), mean - ebar, mean + ebar, alpha=0.2, linewidth=4, linestyle='-', color=colours[j])
        ax.set_xlabel('training session')
        ax.set_ylabel('error (MSE)')
        ax.legend()
    ax.set_facecolor((1,1,1))
# =============================================================================
# Plot learning curves for linedrawing
# =============================================================================
plot_learning(ax)
    
    