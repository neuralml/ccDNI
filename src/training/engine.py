#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Important file: where the iterative training procedure is defined (_training_loop)
"""

import numpy as np
import csv

import torch
import torch.nn as nn
from torch.autograd import Variable #JOP

from ignite.engine import Events, Engine, _prepare_batch
from tqdm import tqdm

import sys
dni_path = 'path_to_dni_file' #directory to dni.py (should be in parent directory)
sys.path.append(dni_path) 
import dni

from torch.nn import functional as F


########################################################################################
# Training
########################################################################################


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


def create_rnn_trainer(model, optimizer, loss_fn, grad_clip=0, reset_hidden=True,
                    device=None, non_blocking=False, prepare_batch=_prepare_batch, 
                    bptt=0, batch_size=50, pred_last=False, record_grads=False, spars_int=None, mask=None, ablation_epoch=None,
                    synth_ablation_epoch=None): #JOP
    if device:
        model.to(device)

    #ablate the synthesiser at a given epoch..
    if hasattr(model, "ablation_epoch") and model.ablation_epoch > 0:
        ablation_epoch = model.ablation_epoch

    #..or the synthesiser learning mechanism (fixed weights)
    if hasattr(model, "synth_ablation_epoch"):
        synth_ablation_epoch = model.synth_ablation_epoch

    record_grads_orig = record_grads

    opt_each_trunc = False #optimize the model each truncation (TRUE) or only at end of batch (FALSE) (latter avoids possibly unfair larger # updates for dni)
	
    def _training_loop(engine, batch):
        # Set model to training and zero the gradients  
        model.train()
        optimizer.zero_grad()

        if record_grads_orig:
	    #recording gradients often memory consuming so take sample
            itn = (engine.state.iteration - 1) % engine.state.epoch_length
            record_grads = itn % 100 == 0
            loss2 = 0 #needed later to solve true BPTT gradients
        else:   
            record_grads = record_grads_orig
        
        # Load the input-targetbatches
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
	
        hidden = engine.state.hidden
        
        seq_len = inputs.shape[1]
        ntrunc = (seq_len - 1) // bptt + 1 #number of feedback horizon (bptt) truncations within sequence
        total_loss = 0 
        sparse = not (spars_int is None) #specify if sparse temporal feedback is given or not 
	
	#loop over input in truncations as specified by the feedback horizon (bptt)
        for i in range(0, ntrunc):
            model.train()
            inputs_trunc = inputs[:,i*bptt:(i+1)*bptt]
            
            if not pred_last:
                targets_trunc = targets[:,i*bptt:(i+1)*bptt]

            # Starting each truncation, we detach the hidden state from how it was previously produced (truncated BPTT).
            # If we didn't, the model would try backpropagating all the way to start of the batch.
            detached_hidden = _detach_hidden_state(hidden)             
            
	    #record unit activities and gradients (e.g. to compare activity correlations/gradient cosimilarity)
            if record_grads:
                if model.model_type in ['DNI_TANH', 'cDNI_TANH']:
                    hidden.requires_grad_()
                    detached_hidden.requires_grad_()
                else:
                    hidden[0].requires_grad_()
                    detached_hidden[0].requires_grad_()
                    hidden[1].requires_grad_()
                    detached_hidden[1].requires_grad_()
            
            #we're doing a backwards operation potentially (if dni and loss defined) twice here. 
            #one when calling forwards() on the (dni) model, one when calling loss.backward()
            #that's what the defer_backward() is there for
            with dni.defer_backward():    
		#provide context to synthesiser (see Jaderberg et al.)? (not modeled in neurips2021 paper)      
                if not model.context:
                    pred, new_hidden = model(inputs_trunc, detached_hidden)
                    contexts = [None, None]
                else:
                    if targets.dim() == 1:
                        contexts = [one_hot(targets, model.output_size), one_hot(targets, model.output_size)]   
                    else:
                        contexts = [targets, targets]
                    pred, new_hidden = model(inputs_trunc, detached_hidden, contexts=contexts)
                
                if record_grads:
                    if not sparse or i < ntrunc - 1: 
                        recordgrads(hidden, detached_hidden, contexts)
			
                ########################################################################################
		# Implement sparse temporal feedback
		########################################################################################
		
                #if pred_last then feedback is only given after processing last truncation where loss will be defined
                if pred_last and i == ntrunc - 1:
                   loss = loss_fn((pred, new_hidden), targets)
                   dni.backward(loss)
                   total_loss = loss.item()          
                elif not pred_last: #if not pred_last then feedback is given sparsely throughout the sequence as defined by spars_int
                    if sparse:
			#some magic to work out which timesteps training targets are provided 
                        tstart = i * bptt 
                        sparse_start = (spars_int - tstart % spars_int) % spars_int
                        tsteps = np.arange(sparse_start, pred.shape[1], spars_int)

                        pred = pred[:,tsteps,:]
                        targets_trunc = targets_trunc[:, tsteps,:]
                    if pred.shape[1] > 0:
                        loss = loss_fn((pred, new_hidden), targets_trunc)
                        dni.backward(loss)   # for tgrads  
                        total_loss += pred.shape[1]*loss.item()      
       
            if opt_each_trunc:
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  
               
                optimizer.step()                
                optimizer.zero_grad()

            #(perhaps elaborate) process to record true BPTT gradient 
            if record_grads: # to record the truegrad is hard, because you make a backward hook every hidden state 
                model.eval() # want the true gradient of hidden state - but we dont use this to optimize model parameters,
                inputs_trunc_detach = _detach_hidden_state(inputs_trunc)
                pred2, new_hidden2 = model(inputs_trunc_detach, hidden)

		#backpropagate on separate, accumulated, undetached loss
                if not pred_last or i == ntrunc - 1:
                    model.requires_grad_(False)
                    if pred_last:
                        loss2 = loss_fn((pred2, new_hidden2), targets)
                        loss2.backward()
                    else:
                        if sparse:
                            pred2 = pred2[:,tsteps,:]
                        loss2 += loss_fn((pred2, new_hidden2), targets_trunc)
                        if i == ntrunc - 1:
                            loss2.backward()
                    model.requires_grad_(True) 
                hidden = new_hidden2
            else:
                hidden = new_hidden      
        
	########################################################################################
	# Update model parameters
	########################################################################################
        if not opt_each_trunc:
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip) 
            
            optimizer.step()
            optimizer.zero_grad()
            
        if not reset_hidden:
            engine.state.hidden = new_hidden
            
        #niche difference between nn.MSELoss() and ignite.metrics.MeanSquaredError
        #the latter will find the mean of the total distance between each (x_t,y_t)
        #pair, the former just finds the mean distance between x_t and y_t individually
        if isinstance(loss_fn._loss_fn, torch.nn.modules.loss.MSELoss):
            total_loss = total_loss * 2
        if not pred_last:
            total_loss = total_loss/seq_len
        
        return total_loss

    # If reusing hidden states, detach them from the computation graph
    # of the previous batch. Using the previous value may speed up training
    # but detaching is needed to avoid backprogating to the start of training.
    def _detach_wrapper(engine):
        if not reset_hidden:
            engine.state.hidden = _detach_hidden_state(engine.state.hidden)
    
    def init_hidden_engine(engine):
        nhid = model.rnn.hidden_size
        nlayers = model.rnn.num_layers
        weight = next(model.parameters()).data
        batch_size = engine.state.batch[0].shape[0]
        
        if model.model_type in ['LSTM', 'DNI_LSTM', 'cDNI_LSTM', 'cDNI_TANH']:
            h0 = (Variable(weight.new(nlayers, batch_size, nhid).zero_()),
                    Variable(weight.new(nlayers, batch_size, nhid).zero_()))
        else:
            h0 = Variable(weight.new(nlayers, batch_size, nhid).zero_())   
        
        engine.state.hidden = h0
       
    #target gradient for cerebellar feedfoward network (*not* true gradient, see neurips2021 paper or jaderberg et al)
    def record_tgrad(grad):
        engine.state.tgrads.append(grad)
    def record_tgrad_output(grad): #LSTM output state
        engine.state.tgrads_output.append(grad)
    def record_tgrad_cell(grad): #LSTM cell state
        engine.state.tgrads_cell.append(grad)

    #true, fully backpropagated gradient
    def record_truegrad(grad):
        engine.state.truegrads.append(grad)
    def record_truegrad_output(grad):
        engine.state.truegrads_output.append(grad)
    def record_truegrad_cell(grad):
        engine.state.truegrads_cell.append(grad)

    def recordgrads(hidden, detached_hidden, contexts=[None, None]):
        if model.model_type in ['DNI_TANH', 'cDNI_TANH']:
            hidden.register_hook(record_truegrad)  
            detached_hidden.register_hook(record_tgrad) 
            engine.state.hidds.append(detached_hidden)
            engine.state.sgrads.append(model.rnn.backward_interface.receive(detached_hidden))
        elif model.model_type in ['DNI_LSTM', 'cDNI_LSTM']:
            hidden[0].register_hook(record_truegrad_output) #output state
            detached_hidden[0].register_hook(record_tgrad_output) 
            
            hidden[1].register_hook(record_truegrad_cell) #cell state  
            detached_hidden[1].register_hook(record_tgrad_cell)    
                 
            engine.state.hidds.append(model.rnn.join_hidden(detached_hidden))
            
            if not model.context:
                engine.state.sgrads.append(model.rnn.backward_interface.receive(model.rnn.join_hidden(detached_hidden)))
                
                synth_hidden = F.relu(model.rnn.backward_interface.synthesizer.input_trigger(model.rnn.join_hidden(detached_hidden)))
            else:
                assert pred_last, "Unimplemented!"
                with dni.synthesizer_context(contexts[0]):
                    engine.state.sgrads.append(model.rnn.backward_interface.receive(model.rnn.join_hidden(detached_hidden)))
                    trigger_term = model.rnn.backward_interface.synthesizer.input_trigger(model.rnn.join_hidden(detached_hidden))
                    context_term = model.rnn.backward_interface.synthesizer.input_context(contexts[0])
                    synth_hidden = F.relu(trigger_term + context_term)
                                
            engine.state.shidds.append(synth_hidden)
        elif model.model_type in ['LSTM']:
            hidden[0].register_hook(record_truegrad)    
            hidden[1].register_hook(record_truegrad)    
            engine.state.hidds.append(torch.cat(detached_hidden, dim=2))        

    
    engine = Engine(_training_loop)

    #store gradients (if desired) in these predefined lists
    if record_grads:
        engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'hidds', []))
        #engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'truegrads', []))
        #engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'tgrads', []))
        
        engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'truegrads_cell', []))
        engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'truegrads_output', []))
        
        #engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'tgrads', []))
        engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'tgrads_cell', []))
        engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'tgrads_output', []))

        engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'sgrads', []))
        engine.add_event_handler(Events.STARTED, lambda e: setattr(e.state, 'shidds', []))
    
    engine.add_event_handler(Events.ITERATION_STARTED, init_hidden_engine)
    engine.add_event_handler(Events.ITERATION_STARTED, _detach_wrapper)
    
    #print epoch + ablate synthesiser parts if necessary
    def print_epoch(engine):
        nepoch = engine.state.epoch
        if nepoch % 5 == 0:
            print("Epoch: {}".format(engine.state.epoch))
        
        if ablation_epoch is not None and engine.state.epoch==ablation_epoch:
            print('ablation of synthesiser at epoch {}'.format(nepoch))           
            model.rnn.ablation = True
           
        if synth_ablation_epoch is not None and engine.state.epoch == synth_ablation_epoch:
            print("ablating synthesiser IO at epoch {}".format(nepoch)) 
            del optimizer.param_groups[1] 
    
    engine.add_event_handler(Events.EPOCH_STARTED, print_epoch)

    #sparsify synthesiser input 'mossy fibre' weights (not modeled in neurips2021)
    def controlfibres(engine):
        model.rnn.backward_interface.synthesizer.input_trigger.weight.register_hook(makezero)
    def makezero(grad):
        grad[mask] = 0
            
    if mask is not None:
        engine.add_event_handler(Events.STARTED, controlfibres) 
    
    
    return engine


def create_rnn_evaluator(model, metrics, device=None, hidden=None, non_blocking=False,
                        prepare_batch=_prepare_batch):
    if device:
        model.to(device)
        
    
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)       
            pred, _ = model(inputs, hidden)
            
            if pred.dim() == 3:
                pred = pred.view(-1, pred.shape[2])
                targets = targets.view(-1, targets.shape[2])
                                                
            return pred, targets

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    if indexes.is_cuda:
        result = result.cuda()
    result.zero_()
    indexes_rank = len(indexes.size())
    result.scatter_(
        dim=indexes_rank,
        index=indexes.data.unsqueeze(dim=indexes_rank),
        value=1
    )
    return Variable(result)

def run_training(
        model, train_data, trainer, epochs,
        metrics, test_data, model_checkpoint, device
    ):
    trainer.run(train_data, max_epochs=epochs)
    
    tester = create_rnn_evaluator(model, metrics, device=device)
    tester.run(test_data)
    
    if hasattr(trainer.state, 'tgrads_output'):
        grads = (trainer.state.hidds, trainer.state.truegrads_output, trainer.state.truegrads_cell, trainer.state.tgrads_output, trainer.state.tgrads_cell, trainer.state.sgrads, trainer.state.shidds)
        return tester.state.metrics, grads
    else:    
        return tester.state.metrics
