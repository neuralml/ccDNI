#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:33:21 2021

@author: ellenboven
"""


import sys
import argparse
import datetime
import yaml
import time


import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Events
from ignite.metrics import Loss, Accuracy
import os 
from os import listdir
from os.path import isfile, join
import math


src_path = 'src_path' #path to src directory
sys.path.insert(0, src_path)


from dataset.linedraw import load_line_draw
from model.init import init_model
from training.setup import set_seed_and_device, setup_training, setup_logging
from training.engine import create_rnn_evaluator, run_training
from correlation import get_popcorr, get_pearson_corr, get_pearson_corr_batch, get_coding_space, get_sparsity, get_weight_info


import logging
logging.getLogger("ignite").setLevel(logging.NOTSET)

import torchvision.transforms as trans


def train_model(
        # Model parameters
        model='LSTM', nlayers=1, nhid=50, dropout=0,
        # Data paramters
        seqlen=50, naddends=2, minval=0, maxval=1, train_noise=0.1, test_noise=0.0,
        training_size=10000, testing_size=1000, val_split=0.2, fixdata=False, input_D=1, npoints=1, targetD=2,
        # Training parameters
        epochs=10, batch_size=50, optim='RMSprop', lr=1e-4, l2_norm=0.0, rate_reg=0.0,
        clip=1.0, early_stopping=False, decay_lr=False, lr_scale=0.1,
        lr_decay_patience=10, keep_hidden=False,
        # Replicability and storage
        save='../../data/sims/linedraw/test', seed=18092, no_cuda=False, 
        verbose=False, log_interval=10, bptt=None, nseeds=1, bias=None, 
        record_grads=False, spars_int = None, synth_nlayers=0, synth_nhid=0, nfibres=0, 
        ablation_epoch=None, synth_ablation_epoch=None, 
):
    # Set training seed and get device
    device = set_seed_and_device(seed, no_cuda)


    # Load training data
    train_data, test_data, validation_data = load_line_draw(
        training_size=training_size,
        test_size=testing_size,
        batch_size=batch_size,
        train_val_split=val_split,
        seq_len=seqlen,
        num_addends=naddends,
        minval=minval,
        maxval=maxval,
        train_noise_var=train_noise,
        test_noise_var=test_noise,
        fixdata=fixdata, 
        input_D=input_D,
        npoints = npoints, 
        targetD = targetD
    )


    #if bptt not set then set as sequence length
    if bptt is None:
        bptt = seqlen

    # Initialise model
    input_size, hidden_size, n_responses, model_type = input_D, nhid, targetD, model
    
    model = init_model(
        model_type=model_type,
        n_layers=nlayers, hidden_size=nhid,
        input_size=input_size, output_size=n_responses,
        device=device,
        dropout=dropout,
        synth_nlayers=synth_nlayers,
        synth_nhid=synth_nhid,
        predict_last=False
    )
    
    #bias for RNN intialisation (not modeled in neurips2021)
    if bias is not None:
        with torch.no_grad():
            if model_type in ['DNI_LSTM', 'DNI_TANH', 'cDNI_LSTM', 'cDNI_TANH']:
                model.rnn.rnn.weight_hh_l0[:] = model.rnn.rnn.weight_hh_l0[:] * bias
                model.rnn.rnn.weight_ih_l0[:] = model.rnn.rnn.weight_ih_l0[:] * bias
                           
                model.rnn.rnn.bias_hh_l0[:] = model.rnn.rnn.bias_hh_l0[:]*bias
                model.rnn.rnn.bias_ih_l0[:] = model.rnn.rnn.bias_ih_l0[:]*bias            
            else:
                model.rnn.weight_hh_l0[:] = model.rnn.weight_hh_l0[:] * bias
                model.rnn.weight_ih_l0[:] = model.rnn.weight_ih_l0[:] * bias

    #sparsify input 'mossy fibre' synthesiser weights if desired (not modeled in neurips2021)
    if model_type in ['LSTM', 'TANH']:
        nfibres = None
    mask = None
    if nfibres is not None:       
        shape = model.rnn.backward_interface.synthesizer.input_trigger.weight.shape  
        mask = torch.ones(shape)
        
        if nfibres != -1:
            for i in range(shape[0]):
                connections = np.random.choice(shape[1], nfibres, replace=False)
                mask[i, connections] = 0
        mask = mask.type(torch.bool)
        with torch.no_grad():
            model.rnn.backward_interface.synthesizer.input_trigger.weight[mask] = 0
            print("Initialised all but {} weights to zero".format(nfibres))
        
        if nfibres == -1:
            mask = None


    # JOP if bptt not set then set as sequence length
    if bptt is None:
        bptt = seqlen

    # Set up the training regime
    setup = setup_training(
        model, validation_data, optim, ['mse'], lr, l2_norm,
        rate_reg, clip, early_stopping, decay_lr, lr_scale, lr_decay_patience,
        keep_hidden, save, device, True, True, bptt, batch_size, False, record_grads, 
        spars_int=spars_int, mask=mask, 
    )

    trainer, validator, checkpoint, metrics = setup[:4]
    training_tracer, validation_tracer, timer = setup[4:]

    if verbose:
        setup_logging(
            trainer, validator, metrics,
            len(train_data), log_interval
        )
        

    # Run training
    test_metrics = run_training(
        model=model,
        train_data=train_data,
        trainer=trainer,
        epochs=epochs,
        test_data=test_data,
        metrics=metrics,
        model_checkpoint=checkpoint,
        device=device
    )

    if record_grads:
        test_metrics, grads = test_metrics
        
    test_mse = test_metrics['mse']

    print('Training ended: test loss {:5.4f}'.format(
        test_mse))

    print('Saving results....')

    # Save traces
#    training_tracer.save(save)
#    validation_tracer.save(save)

    # Save experiment metadata
    model_params = {
        'model_type': model_type,
        'hidden_size': hidden_size,
        'n_layers': nlayers,
        'input_size': input_size,
        'output_size': n_responses,
        'dropout': dropout,
    }

    learning_params = {
        'optimizer': optim,
        'learning-rate': lr,
        'l2-norm': l2_norm,
        'criterion': 'mse',
        'batch_size': batch_size,
    }

    # Save data parameters in a dictionary for testing
    data_params = {
        'seqlen': seqlen, 'naddends': naddends,
        'minval': minval, 'maxval': maxval,
        'train-size': training_size, 'test-size': testing_size,
        'train-noise': train_noise, 'test-noise': test_noise,
        'val-split': val_split, 'keep-hidden': keep_hidden
    }
    
    
    meta = {
        'data-params': data_params,
        'model-params': model_params,
        'learning-params': learning_params,
        'info': {
            #'test-score': test_loss,
            'training-time': timer.value(),
            'timestamp': datetime.datetime.now()
        },
        'seed': seed
    }
    


    with open(save + '\\meta.yaml', mode='w') as f:
        yaml.dump(meta, f)

    print('Done.')

    # JOP
    train_mse = training_tracer.trace
    print('train mse', train_mse)
    val_mse = [vt[0] for vt in validation_tracer.trace]


    if not record_grads:
        return train_mse, val_mse, train_data, test_data, model, trainer 
    else:
        return train_mse, val_mse, train_data, test_data, grads, model, trainer



##############################################################################
# PARSE THE INPUT
##############################################################################
    

start = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train an LSTM variant on the Line Drawing task')

    # Model parameters
    parser.add_argument('--model', type=str, default='LSTM',
                        help='RNN model to use. One of:'
                        '|TANH|DNI_TANH|LSTM|DNI_LSTM')  # JOP
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--nhid', type=int, default=50,
                        help='number of hidden units per layer')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='the drop rate for each layer of the network')

    # Data parameters
    parser.add_argument('--seqlen', type=int, default=10,
                        help='sequence length')
    parser.add_argument('--naddends', type=int, default=2,
                        help='the number of addends to be unmasked in a sequence'
                             'must be less than the sequence length')
    parser.add_argument('--minval', type=float, default=0.0,
                        help='minimum value of the addends')
    parser.add_argument('--maxval', type=float, default=1.0,
                        help='maximum value of the addends')
    parser.add_argument('--train-noise', type=float, default=0.0,
                        help='variance of the noise to add to the data')
    parser.add_argument('--test-noise', type=float, default=0.0,
                        help='variance of the noise to add to the data')
    parser.add_argument('--training-size', type=int, default=1000,
                        help='size of the randomly created training set')
    parser.add_argument('--testing-size', type=int, default=10000,
                        help='size of the randomly created test set')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='proportion of trainig data used for validation')
    parser.add_argument('--fixdata', action='store_true',
                        help='flag to keep the data fixed across each epoch')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='max number of training epochs')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='size of each batch. keep in mind that if the '
                             'training size modulo this quantity is not zero, then'
                             'the it will be increased to create a full batch.')
    parser.add_argument('--optim', type=str, default='adam',
                        help='gradient descent method, supports on of:'
                             'adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--l2-norm', type=float, default=0,
                        help='weight of L2 norm')
    parser.add_argument('--rate-reg', type=float, default=0.0,
                        help='regularization factor for hidden variables')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping')
    parser.add_argument('--early-stopping', action='store_true',
                        help='use early stopping')
    parser.add_argument('--decay_lr', action='store_true',
                        help='if provided will decay the learning after not'
                             'improving for a certain amount of epochs')
    parser.add_argument('--lr-scale', type=float, default=0.5,
                        help='the factor by which learning rate should be scaled'
                             'after the patience has been exhausted')
    parser.add_argument('--lr_decay_patience', type=int, default=10,
                        help='specifies the number of epochs to wait until'
                             'decay is applied before applying the decay factor')
    parser.add_argument('--keep-hidden', action='store_true',
                        help='keep the hidden state values across an epoch'
                             'of training, detaching them from the computation graph'
                             'after each batch for gradient consistency')

    # Replicability and storage
    parser.add_argument('--save', type=str,
                        default='../../data/sims/linedraw', #replace 
                        help='path to save the final model')
    parser.add_argument('--seed', type=int, default=18092,
                        help='random seed')
    parser.add_argument('--nseeds', type=int, default=3,
                        help='number of seeds')

    # CUDA
    parser.add_argument('--no-cuda', action='store_true',
                        help='flag to disable CUDA')

    # Print options
    parser.add_argument('--verbose', action='store_true',
                        help='print the progress of training to std output.')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')

    # JOP
    parser.add_argument('--bptt', type=int, metavar='N',
                        help='truncation size')
    parser.add_argument('--nfibres', type=int, default=-1, metavar='N',
                        help='Number of non-zero weights from hidden to synthesiser (like mossy fibres)')
    parser.add_argument('--record-grads', action='store_true',
                        help='Record hidden activity and corresponding true/synthetic gradients')
    parser.add_argument('--synth-nlayers', type=int, default=1,
                        help='Number of hidden layers for synthesiser')
    parser.add_argument('--synth-nhid', type=int, default=400, metavar='N',
                        help='Number of hidden units in synthesiser')
    parser.add_argument('--input-D', type=int, default=1,
                    help='input dimension') 
    parser.add_argument('--targetD', type=int, default=2,
                    help='target dimension') # for now coded that target dimennsion = input_D 
    parser.add_argument('--npoints', type=int, default=7,
                    help='npoints') 
 
    parser.add_argument('--bias', type=float, default=None,
                    help='Bias')   
    parser.add_argument('--spars_int', type=int, default=None,
                    help='sparse int')
    parser.add_argument('--ablation_epoch', type=int, default=-1, metavar='N',
                        help='when to ablate the synthesiser..')
    parser.add_argument('--synth_ablation_epoch', type=int, default=-1, metavar='N',
                        help='when to ablate the synthesiser learning (IO)..')

    args = parser.parse_args()
    
    args.model = 'DNI_LSTM'
 
    args.record_grads = True
    args.epochs = 1
    
    # as in paper
    args.nhid = 50
    args.synth_nhid = 400   
    args.spars_int = 2
    args.npoints=7
    args.bptt =1
    
    assert args.record_grads, "You're in the record grads file with record_grads as false. Switch it one!"
   
    # for 7 points inputs range from (-3, 3)
    args.minval=-math.floor(args.npoints/2)
    args.maxval = math.floor(args.npoints/2)
    if len(np.arange(args.minval, args.maxval+1)) > args.npoints:
        args.minval = args.minval+1 
            
        
    
    time_x = time.time()
    
    print('Variables', args)
    
                
                    #################TEST MODELS AND PLOT RESULTS###############
    def create_dir(path, folder):
        
        
        path = join(path,folder)
        
        
        if not os.path.exists(path):
            try:
                os.mkdir(path)
                      
            
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)
            
        else:
            print ("Directory %s already exists" % path)
        return path

    

    if args.record_grads:
        
        save_path = '/Users/ellenboven/OneDrive - University of Bristol/Documents/PhD/Cerebellearning/Computational/Code/'
        vecname = str(args.model) + "_nseeds-" + str(args.nseeds) + "_nhid-" + str(args.nhid) + "_synthnhid-" + str(args.synth_nhid) + "_Inpsize-" + str(args.input_D) + "_T-" + str(args.bptt) + "_.npy"
    
        today = datetime.datetime.today()
        experiment = 'LD_lines_grad'
        date =today.strftime('%Y%m%d')
        
        save_path= create_dir(save_path, experiment)
        root= create_dir(save_path, date)

    
        nhid = args.nhid
        nbatch = args.batch_size
        biases = [1]
        seeds = [1, 2, 1243, 34521, 135235, 236236, 7, 12, 1115, 987][:args.nseeds]#        biases = [1]
        nfibres_list = [None]
        corr_methods = [get_popcorr, get_pearson_corr, get_coding_space, get_sparsity, get_weight_info]
        
        score_shape = (len(seeds), len(biases), len(nfibres_list), args.epochs) #add 1 for first batch
        scores = np.zeros(score_shape)
        corr_shape = score_shape[:3] + (args.epochs + 1, len(corr_methods)) # x pop/pearson/etc
        
        corrs_hidd = np.zeros(corr_shape) 
        corrs_truegrads = np.zeros(corr_shape)
        corrs_shidds = np.zeros(corr_shape)
        corrs_sgrad = np.zeros(corr_shape)
        corrs_tgrad = np.zeros(corr_shape)

        ncross_comb = 10 #5 choose 2
        corrs_hidd_vs_shidds = np.zeros(corr_shape)
        corrs_hidd_vs_sgrads = np.zeros(corr_shape)
        corrs_shidds_vs_tgrad = np.zeros(corr_shape[:-1] + ((nhid*2)*args.synth_nhid,))
        corrs_sgrads_vs_tgrad = np.zeros(corr_shape[:-1] + ((nhid*2)**2,))
        corrs_hidd_vs_shidds_all = np.zeros(corr_shape[:-1] + ((nhid*2)*args.synth_nhid,))
        all_scores = np.zeros((len(seeds), 2, args.epochs)) 

        weights_info = np.zeros(score_shape[:2] + (2, 4,4)) #dni/sparsedni, #weights, #measures(mean, absolute mean, absolute mean without zeros, variance)
        
        for i, seed in enumerate(seeds):
            args.seed = seed
            for j, bias in enumerate(biases):
                args.bias = bias
                for k, nfibres in enumerate(nfibres_list):
                    if nfibres == -1:
                        args.model = 'LSTM'       
                    else:
                        args.model = 'DNI_LSTM'    
                    args.nfibres = nfibres
                    
                    #run training
                    train_mse, val_mse, train_data, test_data, grads, model, trainer = train_model(**vars(args))
                    all_scores[i, 0, :] = train_mse 
                    all_scores[i, 1, :] = val_mse
                    
                                      

                    hidds, truegrads_output, truegrads_cell, tgrads_output, tgrads_cell, sgrads, shidds = grads                   
                    hidds = torch.stack(hidds).squeeze(1)

                    if nfibres != -1:
                        truegrads_output = torch.stack(truegrads_output).squeeze(1)
                        truegrads_cell = torch.stack(truegrads_cell).squeeze(1)
                        tgrads_output = torch.stack(tgrads_output).squeeze(1)
                        tgrads_cell = torch.stack(tgrads_cell).squeeze(1)
                        sgrads = torch.stack(sgrads).squeeze(1)
                        shidds = torch.stack(shidds).squeeze(1)
                    
                    savedir = "{}_bias-{}_seed-{}".format(args.model, args.bias, args.seed)
                    if nfibres is not None:
                        savedir += "_nfibres-{}".format(nfibres)
                    savedir = os.path.join(root, savedir) + "/"
                    if not os.path.exists(savedir): 
                        os.makedirs(savedir)
                        
                    """
                    np.save(savedir + 'hidds.npy', hidds.detach().numpy())
                    np.save(savedir + 'sgrads.npy', sgrads.detach().numpy())
                    np.save(savedir + 'truegrads_output.npy', truegrads_output.detach().numpy())
                    np.save(savedir + 'truegrads_cell.npy', truegrads_cell.detach().numpy())
                    np.save(savedir + 'tgrads_output.npy', tgrads_output.detach().numpy())
                    np.save(savedir + 'tgrads_cell.npy', tgrads_cell.detach().numpy())
                    np.save(savedir + 'shidds.npy', shidds.detach().numpy())
                    
                    np.save(savedir + 'scores.npy', all_scores)

                    np.save(savedir + 'biases.npy', np.array(biases))
                    np.save(savedir + 'seeds.npy', np.array(seeds))
                    """
                       
    print("All finished and results saved in", root)