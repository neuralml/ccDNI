#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
sequential mnist classification task. Given an input sequence of MNIST pixels,
classify digit at end.
'''

import sys
import os
import argparse
import datetime
import yaml

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Events
from ignite.metrics import Loss, Accuracy

src_path = 'src_path' #path to src directory
sys.path.insert(0, src_path)

from dataset.mnist import load_mnist
from model.init import init_model
from training.setup import set_seed_and_device, setup_training, setup_logging
from training.engine import run_training

import logging
logging.getLogger("ignite").setLevel(logging.NOTSET)

import glob

def train_model(
    # Model parameters
    model='LSTM', nlayers=1, nhid=50, dropout=0.0,
    # Data paramters
    data_path='../../data/exp_raw/', input_size=28, 
    shuffle=True, val_split=0.2, download=False,
    # Training parameters
    epochs=10, batch_size=50, optim='RMSprop', lr=1e-4,
    l2_norm=0.0, rate_reg=0.0, clip=1.0, early_stopping=False,
    decay_lr=False, lr_scale=0.1, lr_decay_patience=10, keep_hidden=False,
    #truncation/synthesiser params
    bptt=None, synth_nlayers=0, synth_nhid=None, nfibres=None,
    ablation_epoch=-1, record_certainty=False, fixed_synth=False,
    synth_ablation_epoch=-1, record_grads=False,
    # Replicability and storage
    save='../../data/sims/seq-mnist/test', seed=18092, no_cuda=False,
    verbose=False, log_interval=10, nseeds=3, 
):
    # Set training seed and get device
    device = set_seed_and_device(seed, no_cuda)

    ###############################################################################
    # Load training data
    ###############################################################################
    train_data, validation_data, test_data = load_mnist(
        data_path, input_size, batch_size, 
        shuffle=shuffle, download=download
    )

    name_helper = "T{}_seed{}".format(bptt, seed)
    if model == 'LSTM':
        name_helper += 'lstm'
    else:
        name_helper += 'dni'
    save = save.replace('filler', name_helper)
   
    ###############################################################################
    # Build the model
    ###############################################################################
    model_type = model
    model = init_model(
        model_type=model,
        n_layers=nlayers, hidden_size=nhid,
        input_size=input_size, output_size=10,
        device=device,
        dropout=dropout,
        synth_nlayers=synth_nlayers, 
        synth_nhid=synth_nhid,
        predict_last=True
    )

    model.ablation_epoch = ablation_epoch
    model.synth_ablation_epoch = synth_ablation_epoch
    synth_ablation = synth_ablation_epoch > 0   

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

    #if bptt not set then set as sequence length
    if bptt is None:
        npixels = 28**2
        assert float.is_integer(npixels/input_size), "Input size ({:d}) doesn't divide" \
                                                     "number of pixels ({:d})".format(input_size, npixels)
        seq_len = int(npixels/input_size)
        bptt = seq_len
                                                    
    metric_types = ['xent', 'acc']
    if record_certainty:
        metric_types.append('certainty')    

    ###############################################################################
    # Setup training regime 
    ###############################################################################
    setup = setup_training(
        model, validation_data, optim, metric_types, lr, l2_norm,
        rate_reg, clip, early_stopping, decay_lr, lr_scale, lr_decay_patience,
        keep_hidden, save, device, True, True,
        bptt, batch_size, True, record_grads, mask=mask,
        fixed_synth=fixed_synth, synth_ablation=synth_ablation 
    )

    trainer, validator, checkpoint, metrics = setup[:4]
    training_tracer, validation_tracer, timer = setup[4:]

    if verbose:
        setup_logging(
            trainer, validator, metrics,
            len(train_data), log_interval
        )

    ###############################################################################
    # Train the model
    ###############################################################################
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
        
    ###############################################################################
    # Test the model performance
    ###############################################################################
    test_acc = test_metrics['acc']
    test_NLL = test_metrics['xent']


    print('Training ended: test accuracy {:5.4f}'.format(
        test_acc))
    print('Test average probability {:5.4f}'.format(np.exp(-
        test_NLL)))

    print('Saving results....')
    
    ###############################################################################
    # Save model
    ###############################################################################

    # Save traces
    training_tracer.save(save)
    validation_tracer.save(save)

    # Save experiment metadata
    model_params = {
        'model_type': model_type,
        'hidden_size': nhid,
        'n_layers': nlayers,
        'input_size': input_size,
        'dropout': dropout,
        'synth_nlayers': synth_nlayers,
        'synth_nhid': synth_nhid,
        'nfibres': nfibres,
    }

    learning_params = {
        'optimizer': optim,
        'learning-rate': lr,
        'l2-norm': l2_norm,
        'criterion': 'mse',
        'batch_size': batch_size,
        'bptt': bptt
    }

    # Save data parameters in a dictionary for testing
    data_params = {
        'val-split': val_split, 'keep-hidden': keep_hidden
    }

    meta = {
        'data-params': data_params,
        'model-params': model_params,
        'learning-params': learning_params,
        'info': {
            'test-accuracy': test_acc,
            'test-log-likelihood': test_NLL,
            'training-time': timer.value(),
            'timestamp': datetime.datetime.now()
        },
        'seed': seed
    }

    with open(save + '/meta.yaml', mode='w') as f:
        yaml.dump(meta, f)

    print('Done.')
    
    train_NLL = training_tracer.trace
    val_NLL = [vt[0] for vt in validation_tracer.trace]
    val_acc = [vt[1] for vt in validation_tracer.trace]

    if record_certainty:
        val_cert = [vt[2] for vt in validation_tracer.trace]
    
    
    if record_grads:
        return train_NLL, val_NLL, val_acc, grads, model
    elif record_certainty:
        return train_NLL, val_NLL, val_acc, val_cert
    else:
        return train_NLL, val_NLL, val_acc

##############################################################################
# PARSE THE INPUT
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train an LSTM variant on the Delayed Addition task')

    # Model parameters
    parser.add_argument('--model', type=str, default='LSTM',
                        help='RNN model to use. One of:'
                        '|TANH|DNI_TANH|LSTM|DNI_LSTM') 
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--nhid', type=int, default=30,
                        help='number of hidden units per layer')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='the drop rate for each layer of the network')

    # Data parameters
    parser.add_argument('--data-path', type=str, default='../../data/exp_raw/MNIST',
        help='location of the data set')
    parser.add_argument('--input-size', type=int, default=28,
        help='the default dimensionality of each input timestep.'
        'defaults to 1, meaning instances are treated like one large 1D sequence')
    parser.add_argument('--val-split', type=float, default=0.2,
        help='proportion of trainig data used for validation')
    parser.add_argument('--shuffle', action='store_true',
        help='shuffle the data at the start of each epoch.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=40,
                        help='max number of training epochs')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='size of each batch. keep in mind that if the '
                        'training size modulo this quantity is not zero, then'
                        'the it will be increased to create a full batch.')
    parser.add_argument('--optim', type=str, default='adam',
                        help='gradient descent method, supports on of:'
                        'adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
    parser.add_argument('--lr', type=float, default=1e-4,
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
    parser.add_argument('--lr-scale', type=float, default=0.1,
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
                        default='../../data/sims/seq-mnist/test',
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
    
    #DNI params
    parser.add_argument('--bptt', type=int, metavar='N',
                        help='truncation size')
    parser.add_argument('--record-grads', action='store_true',
                        help='Record hidden activity and corresponding true/synthetic gradients')
    parser.add_argument('--synth-nlayers', type=int, default=1,
                        help='Number of hidden layers for synthesiser')
    parser.add_argument('--synth-nhid', type=int, metavar='N',
                        help='Number of hidden units in synthesiser')
    parser.add_argument('--nfibres', type=int, metavar='N',
                        help='Number of non-zero weights from model hidden to synthesiser')
    parser.add_argument('--ablation_epoch', type=int, default=-1, metavar='N',
                        help='when to ablate the synthesiser')
    parser.add_argument('--synth_ablation_epoch', type=int, default=-1, metavar='N',
                        help='when to ablate the synthesiser learning (IO)')
    parser.add_argument('--record_certainty', action='store_true', 
                        help='record the average model confidence at each epoch')
    parser.add_argument('--fixed-synth', action='store_true', 
                        help='fix synthesiser weights')

    args = parser.parse_args()
    
    args.model = 'DNI_LSTM'
    args.bptt = 3
    args.epochs = 1
    args.nseeds = 1
    
    if args.model in ['TANH', 'LSTM']:
        args.synth_nhid = 0
        if args.record_grads:
            print("Can't record synthetic gradients for non DNI model. Setting 'record-grads' to false.")
            args.record_grads = False
            
    #delete existing saved models
    files = glob.glob(args.save + "/*")
    for f in files:
        print(f)
        os.remove(f)    
    
    #################TEST MODELS AND PLOT RESULTS###############
    seeds = [521, 1243, 34521, 135235, 236236, 54354, 3938, 5497, 39291, 105043, 232998, 12095][:args.nseeds]
    models = [args.model]
    
    nmetrics = 4 if args.record_certainty else 3
    all_scores = np.zeros((len(seeds), len(models), nmetrics, args.epochs)) #3 for training_NLL, validation_NLL and validation_acc 

    for i, s in enumerate(seeds):
        args.seed = s  
        for j, model in enumerate(models):
            args.model = model
                        
            if args.record_certainty:
                train_NLL, val_NLL, val_acc, val_cert = train_model(**vars(args))
            else: 
                train_NLL, val_NLL, val_acc = train_model(**vars(args))

            all_scores[i, j, 0,:] = train_NLL
            all_scores[i, j, 1,:] = val_NLL
            all_scores[i, j, 2,:] = val_acc

            if args.record_certainty:
                all_scores[i, j, 3,:] = val_cert


    root = 'savepath' #where to save results 
    vecname = str(args.model) + "_nseeds-" + str(args.nseeds) + "_nhid-" + str(args.nhid) + "_synthnhid-" + str(args.synth_nhid) + "_Inpsize-" + str(args.input_size) + "_T-" + str(args.bptt) + "_.npy"

    if args.ablation_epoch != -1:
        suff = "_ablation_{}_.npy".format(args.ablation_epoch)
        vecname = vecname.replace('_.npy', suff)    

    if args.synth_ablation_epoch != -1:
        suff = "_synthablation_{}_.npy".format(args.synth_ablation_epoch)
        vecname = vecname.replace('_.npy', suff)       
 
    if args.nfibres is not None and args.model in ['DNI_LSTM', 'DNI_TANH', 'cDNI_LSTM', 'cDNI_TANH']:
        vecname = vecname.replace('.npy', 'nfibres-' + str(args.nfibres) + '_.npy')
    
    if args.synth_nlayers > 1:
        vecname = vecname.replace('.npy', 'synthnlayers-' + str(args.synth_nlayers) + '_.npy')

    if 'TANH' in args.model:
        root = root + 'RNN/'

    if args.record_certainty:
        vecname = vecname.replace('.npy', 'withcertainty_.npy') 
 
    if args.fixed_synth:
        vecname = vecname.replace('.npy', 'fixedsynth_.npy')   

    np.save(root+vecname, all_scores)
    print("Saved array in {}".format(root))
