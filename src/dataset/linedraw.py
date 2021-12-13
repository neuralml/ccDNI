#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:54:16 2021

@author: ellenboven
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def draw_2dlines(nlines, seq_len):
    X = np.zeros((nlines, seq_len, 2))
    
    end_points = get_circle_points(nlines)
    
    for i in range(nlines):
        line = draw_2dline(seq_len, end_points[i])
        X[i] = line
    
    return torch.from_numpy(X).float()
    

def get_circle_points(npoints, mag=10):
    coords =np.zeros((npoints, 2))
    angles = np.linspace(0, 2*np.pi, npoints-1, endpoint=False)
    coords[1:,:] = np.vstack((np.sin(angles), np.cos(angles))).T
    if npoints >1:
        coords[0,:]=0
    coords = mag * coords
    return coords
    
    
def draw_2dline(seq_len, end_point, start_point = None):
    #pick a random start point and end point on the square, 
    #and make sequence of points between them
    if start_point is None: 
        start_point = np.zeros(2) #each line starts at the origin
    
    seqx = np.linspace(start_point[0], end_point[0], num=seq_len)
    seqy = np.linspace(start_point[1], end_point[1], num=seq_len)
    
    line = np.vstack((seqx, seqy)).T
    
    return line

def sample_data(
    size,
    seq_len,
    min_value,
    max_value,
    noise_var,
    mask_type='int',
    rng=None, scaling = 1.0, 
    input_D=1, npoints=1, targetD = 1
    
):
    
    
    if mask_type == 'int':
        rand = rng.random_integers
    elif mask_type == 'float':
        rand = rng.uniform
    else:
        raise ValueError()


    npoints = npoints
    nlines = npoints
    coords = draw_2dlines(nlines, seq_len) #get target lines
    coords = coords.numpy()

    if input_D == 1:
        inputs = rand(low=min_value, high=max_value, size=(size, seq_len, 1))
        inputs[:, 1:, 0] = 0 #zeroth input from first position
        targets = np.zeros((size, seq_len, 2))
               
        
        if npoints ==1:
             c = np.where(inputs[:, 0, :])
             targets[c[0],:] = coords[0]
        else:
            
            c=0
            for p in np.arange(min_value, max_value+1):
                
                if p ==0:
                    c2 = np.where(inputs[:,0,:]==p)
                    targets[c2[0], :] = coords[0, :] # +1 becayse min_value is -1 
                else:
                    coords_n=np.flip(coords[1:],0)
                    c2 = np.where(inputs[:,0,:]==p)
                    targets[c2[0], :] = coords_n[c, :] # +1 becayse min_value is -1 
                    c=c+1
                    
    # scale the inputs
    inputs=inputs.astype(float)
    inputs[:, 0, :] = inputs[:, 0, :].astype(float)*0.1
    if noise_var > 0:
        inputs[:, 1:, :] = inputs[:, 1:, :].astype(float) + rng.randn(size, seq_len-1, input_D)*noise_var

    return inputs, targets


def get_mask_labels(dataset):
    ops = dataset.swapaxes(1, 2).swapaxes(0, 1)[1]
    labels = np.asarray(['add' if i else 'ignore' for i in ops.reshape(-1)])
    return labels.reshape(ops.shape)


class BatchGenerator:
    def __init__(
        self, size=1000, seq_len=10, noise_var=0.0, min_value=0, max_value=1, 
        batch_size=10, offline_data=False, random_state=None, scaling=1, 
        input_D=1, npoints=1, targetD=2
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.size = size
        self.noise_var = noise_var
        self.offline_data = offline_data
        self.scaling = scaling
        self.input_D = input_D
        self.npoints = npoints
        self.targetD = targetD

        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        self.rng = random_state
        self.init_state = random_state.get_state()

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next_batch()

        if self.offline_data:
            self.reset()

    def __len__(self):
        return int(np.ceil(self.size / self.batch_size))

    def n_instances(self):
        return self.size * self.batch_size

    def reset(self):
        self.rng.set_state(self.init_state)

    def next_batch(self):
        inputs, targets = sample_data(
            size=self.batch_size, seq_len=self.seq_len,
            min_value=self.min_value, max_value=self.max_value,
            noise_var=self.noise_var, mask_type='int', rng=self.rng, 
            scaling=self.scaling, input_D = self.input_D, npoints=self.npoints, 
            targetD=self.targetD
        )

        inputs = torch.as_tensor(inputs, dtype=torch.float)
        if self.targetD ==1:
            targets = torch.as_tensor(targets, dtype=torch.long)
        else:
            targets = torch.as_tensor(targets, dtype=torch.float)

        return inputs, targets


    def torch_dataset(self):
        current_state = self.rng.get_state()
        self.rng.set_state(self.init_state)

        inputs, targets = zip(*[batch for batch in self])

        self.rng.set_state(current_state)

        data = TensorDataset(torch.cat(inputs), torch.cat(targets))

        return DataLoader(
            dataset=data,
            batch_size=self.batch_size, shuffle=True
        )


class SoftMaskBatchGenerator(BatchGenerator):
    def next_batch(self):
        inputs, targets = sample_data(
            size=self.batch_size, seq_len=self.seq_len,
            min_value=self.min_value, max_value=self.max_value,
            noise_var=self.noise_var, num_values=self.num_values,
            mask_type='float', rng=self.rng, scaling=self.scaling, input_D = self.input_D, npoints = self.npoints, targetD = self.targetD
        )

        inputs = torch.as_tensor(inputs, dtype=torch.float)
        if self.targetD ==1:
            targets = torch.as_tensor(targets, dtype=torch.long)
        else:
            targets = torch.as_tensor(targets, dtype=torch.float)

        return inputs, targets


def load_line_draw(
    training_size,
    test_size,
    batch_size,
    seq_len,
    num_addends,
    minval,
    maxval,
    train_val_split,
    train_noise_var,
    test_noise_var,
    fixdata=False,
    mask_type='int',
    random_state=None,
    scaling=1,
    input_D=1,
    npoints=1, 
    targetD = 1
):
    N = int(training_size * (1 - train_val_split))
    val_size = training_size - N

    assert input_D == 1, 'Input dimension > 1 requires (new) code implementation'

    if random_state is None:
        train_rng = np.random.randint(2**16-1)
        val_rng = np.random.randint(2**16-1)
        test_rng = np.random.randint(2**16-1)
    else:
        train_rng = random_state.randint(2**16-1)
        val_rng = random_state.randint(2**16-1)
        test_rng = random_state.randint(2**16-1)

    if mask_type == 'int':
        generator = BatchGenerator
    else:
        generator = SoftMaskBatchGenerator

    training_data = generator(
        size=N, seq_len=seq_len, min_value=minval, max_value=maxval, 
        batch_size=batch_size, noise_var=train_noise_var, offline_data=fixdata, 
        random_state=train_rng, scaling=scaling, input_D = input_D, npoints=npoints, targetD=targetD)

    validation_data = generator(
        size=val_size, seq_len=seq_len, min_value=minval, max_value=maxval, 
        batch_size=val_size, noise_var=train_noise_var, offline_data=fixdata, 
        random_state=val_rng, scaling=scaling, input_D=input_D, npoints=npoints, targetD=targetD)

    test_data = generator(
        size=test_size, seq_len=seq_len, min_value=minval, max_value=maxval, 
        batch_size=test_size, noise_var=test_noise_var, offline_data=fixdata, 
        random_state=test_rng, scaling=scaling, input_D=input_D, npoints=npoints, targetD=targetD)

    if fixdata:
        training_data  = training_data.torch_dataset()
        test_data = test_data.torch_dataset()
        val_size = validation_data.torch_dataset()

    return training_data, test_data, validation_data
