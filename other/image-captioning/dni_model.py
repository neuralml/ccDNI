#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:13:50 2019

@author: joe
dni rnn for the sequential mnist task. To be compatible with Milton's code
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import dni
import sys

class RNNdni(nn.Module):
    
    """Container module a recurrent module, and a backward interface."""

    def __init__(self, input_size, hidden_size, num_layers, rnn_type, batch_first=False, dropout=0,
                 tie_weights=False, use_dni=False, context_dim=None, synth_nlayers=1, synth_nhid=None):
        super(RNNdni, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout, batch_first=batch_first)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=nonlinearity, dropout=dropout)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        self.sgfactor = 0.1 # scale synthetic gradient by a factor of 0.1, as in the paper

        if use_dni:
            if rnn_type == 'LSTM':
                output_dim = 2 * hidden_size
            else:
                output_dim = hidden_size
        
            self.backward_interface = dni.BackwardInterface(
                dni.BasicSynthesizer(output_dim, n_hidden=synth_nlayers, hidden_dim=synth_nhid, context_dim=context_dim)
            )
        else:
            self.backward_interface = None
        
        

    def forward(self, input, hidden, contexts=[None, None]):                
        if self.backward_interface is not None:
            # for LSTM, predict gradient for both cell state and output
            # to do that, concatenate them before feeding to DNI
            hidden = self.join_hidden(hidden)
            with dni.synthesizer_context(contexts[0]):
                hidden = self.backward_interface.make_trigger(hidden)

            hidden = self.split_hidden(hidden)
        output, hidden = self.rnn(input, hidden)
        if self.backward_interface is not None:
            hidden = self.join_hidden(hidden)
            with dni.synthesizer_context(contexts[1]):
                self.backward_interface.backward(hidden, factor=self.sgfactor)
            hidden = self.split_hidden(hidden)
        
        return output, hidden
    
    
    def join_hidden(self, hidden):
        if hidden is None:
            return None
        if self.rnn_type == 'LSTM':
            hidden = torch.cat(hidden, dim=2)
        return hidden

    def split_hidden(self, hidden):
        if hidden is None:
            return None
        if self.rnn_type == 'LSTM':
            (h, c) = hidden.chunk(2, dim=2)
            hidden = (h.contiguous(), c.contiguous())
        return hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            h0 = (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()))
        else:
            h0 = Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_())   
        return h0



def _detach_hidden_state(hidden_state):
    """
    Use this method to detach the hidden state from the previous batch's history.
    This way we can carry hidden states values across training which improves
    convergence  while avoiding multiple initializations and autograd computations
    all the way back to the start of start of training.
    """

    if hidden_state is None:
        return None
    elif isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    elif isinstance(hidden_state, list):
        return [_detach_hidden_state(h) for h in hidden_state]
    elif isinstance(hidden_state, tuple):
        return tuple(_detach_hidden_state(h) for h in hidden_state)
    raise ValueError('Unrecognized hidden state type {}'.format(type(hidden_state)))
