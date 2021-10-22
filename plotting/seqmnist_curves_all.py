#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot seqMNIST drawing tasks for different cortical feedback windows (truncations)
Used to generate NeurIPS Figure S3.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec

metric = 0 #training MSE (sparse feedback)
model_names = ['cRNN', 'ccRNN', 'full BPTT']
BPTTS = [3, 1, 2,4, 5, 7, 10]
colours = [(0.5,0.5,0.5),(249/256, 170/256, 56/256), 'r']

full_bptt_fn = "{}_sparsintnew-4_LSTM_nseeds-3_nhid-30_synthnhid-0_Inpsize-28_T-28_.npy"

root = 'results_dir' #directory where results are stored. Should contain separate 'fixedmag' (linedraw) and 'digits' folders.

tasks = ['fixedmag', 'digits']

def get_vecs(task, bptt):
    dni_fn = "{}_sparsintnew-4_DNI_LSTM_nseeds-3_nhid-30_synthnhid-300_Inpsize-28_T-{}_.npy".format(task, bptt)    
    lstm_fn = dni_fn.replace('DNI_LSTM', 'LSTM')
    lstm_fn = lstm_fn.replace('synthnhid-300', 'synthnhid-0')
    
    lstm = np.load(os.path.join(root, task, lstm_fn))
    dni = np.load(os.path.join(root, task, dni_fn))
    
    lstm = lstm[:, 0, metric]
    dni = dni[:, 0, metric]
        
    if task == 'digits':
        lstm *= 100  
        dni *= 100
    
    return lstm, dni

def load_full_bptt(task):
    full_bptt = np.load(os.path.join(root, task, full_bptt_fn.format(task)))
    full_bptt = full_bptt[:, 0, metric]
    
    if task == 'digits':
        full_bptt *= 100  
    
    return full_bptt

nrow = len(BPTTS) // 2 + 1
ncol = 2

fig = plt.figure(figsize=(12,nrow*2.2))
outer = gridspec.GridSpec(1, 2, hspace=0.3, wspace=0.4)

hspace = 0.35
wspace = 0.35



for i, task in enumerate(tasks):
    inner = gridspec.GridSpecFromSubplotSpec(nrow, ncol,
                    subplot_spec=outer[i], wspace=wspace, hspace=hspace)
    for j, bptt in enumerate(BPTTS):
        if j == 0:
            ax = plt.Subplot(fig, inner[:2])
        else:
            ax = plt.Subplot(fig, inner[j+1])
            
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
            
        perc = int(np.round(100 * bptt/28) )
        
        if j == 0:
            title = 'cortical temporal feedback = {}%'.format(perc)
        else:
            title = '{}%'.format(perc)
        
            
        lstm, dni = get_vecs(task, bptt)
        models = [lstm, dni]
        if bptt == 3:
            full_bptt = load_full_bptt(task)       
            models.append(full_bptt)
        
        x = np.arange(1, lstm.shape[1] + 1)
        
        for k, vec in enumerate(models):
            mean = np.mean(vec, axis=0)
            ebar = np.std(vec, axis=0)/np.sqrt(vec.shape[0])
            ax.plot(x, mean, color=colours[k], label=model_names[k], linestyle='-')
            ax.fill_between(x, mean-ebar, mean+ebar, alpha=0.2, edgecolor=colours[k], 
                                 facecolor=colours[k], linewidth=4, linestyle='--')
            

        
        if i == 0 and j == 0:
            ax.legend(frameon=False)
        
        if j == 0 or j % 2 == 1:
            ax.set_ylabel("error (MSE)")
        
        if j == 0 or j >= len(BPTTS) - 2:
            ax.set_xlabel("epoch")
         
        if i == 1:
            ymin = ax.get_ylim()[0]
            ax.set_ylim(top=ymin+5)
            ax.set_yscale('log')
            
        y = 1 if j == 0 else 0.95
        ax.set_title(title, y=y)
            
        fig.add_subplot(ax)
        
fig.text(0.22, 0.95, 'ld-seqMNIST', fontweight='bold', fontsize=15)
fig.text(0.68, 0.95, 'dd-seqMNIST', fontweight='bold', fontsize=15)

