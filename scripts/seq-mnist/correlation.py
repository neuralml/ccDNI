#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

def get_corr_from_eigs(eigs):
    N = len(eigs)
    num = np.max(np.sqrt(eigs))
    denom = np.sum(np.sqrt(eigs))
    factor = N/(N-1)
    const = -1/N
    
    corr = factor * (num/denom) + const
    
    return corr

def get_popcorr(mat):
    mat = preprocess(mat)   
    cov = np.cov(mat)    
    eigs, _ = np.linalg.eig(cov)
    
    return get_corr_from_eigs(eigs)


def preprocess(mat, batchcorr=False):
    if isinstance(mat, torch.Tensor):
        mat = mat.detach().numpy()
    
    if len(mat.shape) == 1:
        print("")  
    elif len(mat.shape) == 2:
        if mat.shape[0] > mat.shape[1]:
            if not batchcorr:
                mat = mat.T
    else: 
        sys.exit("Unexpected shape for input: {}".format(mat.shape))
    
    return mat

def get_pearson_corr(mat):
    mat = preprocess(mat)
    corrs = np.corrcoef(mat)
    corrs = np.abs(corrs)
    mask = ~np.identity(corrs.shape[0], dtype=bool) #ignore correlations with oneself
    
    mean_corr = np.nanmean(corrs[mask])
    
    return mean_corr



def get_pearson_corr_batch(mat):
    mat = preprocess(mat, batchcorr=True)
    corrs = np.corrcoef(mat)
    corrs = np.abs(corrs)
    mask = ~np.identity(corrs.shape[0], dtype=bool) #ignore correlations with oneself
    
    mean_corr = np.nanmean(corrs[mask])
    
    return mean_corr

def get_coding_space(mat):
    mat = preprocess(mat)
    cov = np.cov(mat)
    mask = np.identity(cov.shape[0], dtype=bool)    
    cod_space = np.sum(cov[mask])
    
    return cod_space


def get_sparsity(mat):
    mat = preprocess(mat)
    mat = np.abs(mat)   
    N = mat.shape[0]    
    
    num = np.sum(mat, axis=0)**2
    denom = np.sum(mat**2, axis=0)    
    meanterm = np.mean(num/denom)
    
    sparsity = (N - meanterm)/(N-1)        
    return sparsity


def get_weight_info(weight, ignore_zeros = False, absolute=False, getvar=False):
    weight = preprocess(weight)
    weight = weight.reshape(-1)
    if getvar:
        std = np.std(weight)
        return std

    if absolute:
        weight = np.abs(weight)

    if ignore_zeros:
        mean = np.mean(weight[weight != 0])
    else:
        mean = np.mean(weight)

    return mean
