#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Linedrawing task. Given an input stimulus predict (x_t, y_t) coordinates for drawing.
'''


import sys
import argparse
import datetime
import yaml
import code
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


###############################################################################
# Set path to src folder 
###############################################################################

src_path = 'INSERT PARTH HERE'#path to src directory
sys.path.insert(0, src_path)


from dataset.linedraw import load_line_draw
from model.init import init_model
from training.setup import set_seed_and_device, setup_training, setup_logging
from training.engine import create_rnn_evaluator, run_training

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
        verbose=False, log_interval=10, bptt=None, nseeds=1, 
        record_grads=False, spars_int=None, synth_nlayers=0, synth_nhid=0, nfibres = 0, 
        ablation_epoch=None, synth_ablation_epoch=None,  
):
    # Set training seed and get device
    device = set_seed_and_device(seed, no_cuda)

    ###############################################################################
    # Load training data
    ###############################################################################

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

    ###############################################################################
    # Build the model
    ###############################################################################

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


    # JOP if bptt not set then set as sequence length
    if bptt is None:
        bptt = seqlen

    ###############################################################################
    # Setup training regime 
    ###############################################################################

    setup = setup_training(
        model, validation_data, optim, ['mse'], lr, l2_norm,
        rate_reg, clip, early_stopping, decay_lr, lr_scale, lr_decay_patience,
        keep_hidden, save, device, True, True, bptt, batch_size, False, record_grads, 
        spars_int=spars_int, mask=mask, synth_ablation=synth_ablation,
    )

    trainer, validator, checkpoint, metrics = setup[:4]
    training_tracer, validation_tracer, timer = setup[4:]

    if verbose:
        setup_logging(
            trainer, validator, metrics,
            len(train_data), log_interval
        )
        

    ###############################################################################
    # Training the model
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
        
    test_mse = test_metrics['mse']

    print('Training ended: test loss {:5.4f}'.format(
        test_mse))

    print('Saving results....')


    ###############################################################################
    # Save model
    ###############################################################################
        
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

    train_mse = training_tracer.trace
    val_mse = [vt[0] for vt in validation_tracer.trace]

    if not record_grads:
        return train_mse, val_mse, train_data, test_data, model, trainer 
    else:
        return train_mse, val_mse, train_data, test_data, grads, model, trainer


##############################################################################
# PARSE THE INPUT PARAMETERS and usage specifications
##############################################################################
    

start = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train an LSTM variant on the Line Drawing task')

    # Model parameters
    parser.add_argument('--model', type=str, default='LSTM',
                        help='RNN model to use. One of:'
                             '|TANH|DNI_TANH|LSTM|DNI_LSTM') #LSTM is equivalent to cerebral RNN and DNI_LSTM is equivalent to cerebro-cerebellar RNN
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
    parser.add_argument('--input-D', type=int, default=1,
                    help='input dimension') 
    parser.add_argument('--targetD', type=int, default=2,
                    help='target dimension') # for now coded that target dimennsion = input_D 
    parser.add_argument('--npoints', type=int, default=7,
                    help='npoints') 

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
                        default='../../data/sims/linedraw', #replace 
                        help='path to save the final model')
    parser.add_argument('--seed', type=int, default=353,
                        help='random seed')
    parser.add_argument('--nseeds', type=int, default=1,
                        help='number of seeds')

    # CUDA
    parser.add_argument('--no-cuda', action='store_true',
                        help='flag to disable CUDA')

    # Print options
    parser.add_argument('--verbose', action='store_true',
                        help='print the progress of training to std output.')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')

    # ccRNN parameters
    parser.add_argument('--bptt', type=int, metavar='N',
                        help='size of feedback horizon as specified by backpropagation through time (bptt)')
    parser.add_argument('--nfibres', type=int, default=-1, metavar='N',
                        help='Number of non-zero weights (mossy fibre input) from cortical RNN to cerebellar feedforward network')
    parser.add_argument('--record-grads', action='store_true',
                        help='Record hidden activity and corresponding true/synthetic gradients')
    parser.add_argument('--synth-nlayers', type=int, default=2,
                        help='Number of hidden layers for cerebellar feedforward network model')
    parser.add_argument('--synth-nhid', type=int, metavar='N', default=400, 
                        help='Number of hidden units in cerebellar feedforward network model')

  
    parser.add_argument('--spars_int', type=int, default=None,
                    help='defines the temporal sparseness of the external feedback to the RNN')
    parser.add_argument('--ablation_epoch', type=int, default=-1, metavar='N',
                        help='when to ablate the cerebellar feedforward network model')
    parser.add_argument('--synth_ablation_epoch', type=int, default=-1, metavar='N',
                        help='when to ablate the learning in cerebellar feedforward network model (like inferior olive (IO) )')

    args = parser.parse_args()
    
    #args.model = 'DNI_LSTM' 
    args.epochs = 100
    
    #as in paper
    args.nhid = 50
    args.synth_nhid = 400   
    args.spars_int = 2
    args.npoints = 7
    args.bptt = 1
   
    # for 7 points inputs range from (-3, 3)
    args.minval=-math.floor(args.npoints/2)
    args.maxval = math.floor(args.npoints/2)
    if len(np.arange(args.minval, args.maxval+1)) > args.npoints:
        args.minval = args.minval+1 
            
        
    time_x = time.time()
    
    
                
    ###############################################################################
    # Run the model according to train_model
    ###############################################################################

    # Specify the different initialisation of the network (seeds)
    seeds = [1, 2, 1243, 34521, 135235, 236236, 7, 12, 1115, 987][:args.nseeds] 

    # Create empty matrices to save model results
    all_scores = np.zeros((len(seeds), 2, args.epochs)) 
    pred_all = np.zeros((len(seeds), int(args.npoints), args.seqlen, args.targetD))

    # Loop over seeds
    for i, s in enumerate(seeds):
        print("Seed number:", s)
        args.seed = s
        print(i)


        if not args.record_grads:
            #run training with parsed input
            train_mse, val_mse, train_data, test_data, model, trainer = train_model(**vars(args))
        else:
            print('Code not set up for recording gradients')
            sys.exit()

        all_scores[i, 0, :] = train_mse 
        all_scores[i, 1, :] = val_mse
        
        #loop to get model output traces unique to the training examples
        if i==0:
            data, target = next(iter(test_data))
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                model.cuda()
            data, target = next(iter(test_data))
            data[:,1:,:]=0
            unique_data, idx=torch.unique(data, dim=0, sorted=False, return_inverse=True)
            unique_targets = np.zeros((args.npoints, args.seqlen, args.targetD))
            unique_targets[:] = ['nan']
            for count, idx_v in enumerate(np.unique(idx)):
                c = np.where(idx==idx_v)
                it=0
                tmp = target.numpy()[c[0][0]]
                unique_targets[count,:, :]=tmp
        
            noise = np.double(np.random.randn(args.seqlen,1)*args.train_noise)

            un_data_noise = unique_data+torch.Tensor([noise])
            
        #retrieve model output at the end of training fir 
        pred, _ = model(un_data_noise)         
        pred_all[i, :,:,:] = pred.detach().numpy()
                
        elapsed = time.time() - time_x

        print("Time elapsed =", elapsed)
            
    ###############################################################################
    # Save model output
    ###############################################################################

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

    

    save_path = 'savepath' #where to save results
    vecname = str(args.model) + "_nseeds-" + str(args.nseeds) + "_nhid-" + str(args.nhid) + "_synthnhid-" + str(args.synth_nhid) + "_Inpsize-" + str(args.input_D) + "_T-" + str(args.bptt) + "_.npy"

    today = datetime.datetime.today()
    experiment = 'LD_lines'
    date =today.strftime('%Y%m%d')
    
    save_path= create_dir(save_path, experiment)
    save_path= create_dir(save_path, date)
    
 
    if args.ablation_epoch != -1:
        suff = "_ablation_{}_.npy".format(args.ablation_epoch)
        vecname = vecname.replace('_.npy', suff)

    if args.synth_ablation_epoch != -1:
        suff = "_synthablation_{}_.npy".format(args.synth_ablation_epoch)
        vecname = vecname.replace('_.npy', suff)

    if args.spars_int is not None:
        vecname = "sparsint-" + str(args.spars_int) + "_" + vecname    

    if args.nfibres > -1 and args.model in ['DNI_LSTM', 'DNI_TANH']:
        vecname = vecname.replace('.npy', 'nfibres-' + str(args.nfibres) + '_.npy')
 

    vecname = vecname.replace('.npy', 'scores' + '_.npy')
    #np.save(join(save_path,vecname), all_scores)
    
    vecname = vecname.replace('scores', 'model-output')
    #np.save(join(save_path, vecname), pred_all)
    #print("Saved array as {}".format(join(save_path, vecname+'_model-output')))
