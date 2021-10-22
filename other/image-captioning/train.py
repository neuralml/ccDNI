import nltk
import argparse
import torch
import torch.nn as nn
import numpy as np
import os, sys

import pickle
from data_loader import get_loader, get_lengthsmat
from build_vocab import Vocabulary

from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import dni
from dni_model import _detach_hidden_state

from correlation import get_popcorr, get_pearson_corr, get_sparsity, get_coding_space

import re

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    args = checkargs(args)
    
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
   
    #set seed
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
 
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    #only look at captions which fall in desired range
    capt_range = (args.capt_min, args.capt_max) 

    if not args.only_validate:
        data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             num_workers=args.num_workers, capt_range=capt_range, balance_data=args.balance_data, shuffle_data=True) 

    if args.validate:

        if args.val_same_transform:
            val_transform = transform
        else:
            val_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
        val_data_loader = get_loader(args.val_image_dir, args.val_caption_path, vocab, val_transform, args.val_batch_size, num_workers=args.num_workers, capt_range=capt_range, validate=not args.validate_whole, balance_data=args.balance_data, shuffle_data=False) 
        val_losses = []

    if args.decoder_path == '/home/oq19042/dni/image-captioning/models/decoder-5-3000.pkl' and args.only_validate:
        print("Reviewing pretrained decoder!")
        from model_orig import EncoderCNN, DecoderRNN 

        # Build the models
        encoder = EncoderCNN(args.embed_size).to(device)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, args.rnn_type, synth_nhid = args.synth_nhid, synth_nlayers = args.synth_num_layers).to(device)
 
    else:
        from model import EncoderCNN, DecoderRNN

        # Build the models
        encoder = EncoderCNN(args.embed_size).to(device)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, args.rnn_type, synth_nhid = args.synth_nhid, synth_nlayers = args.synth_num_layers, dropout=args.dropout).to(device)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.train_enc:
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters()) 
    else:
        params = list(decoder.parameters())

        #set encoder to pretrained optimum (where it will be fixed)
        if torch.cuda.is_available():
            encoder.load_state_dict(torch.load(args.encoder_path)) 
        else:
            encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))
        encoder.eval()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.l2_norm)


    if args.recommence_training:
        print("Reloading pretrained decoder model!")
        print("Decoder path:", args.decoder_path)
        #print("Keys:", torch.load(args.decoder_path).keys())
        if torch.cuda.is_available():
            decoder.load_state_dict(torch.load(args.decoder_path)) 
        else:
            decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu'))
        if args.train_enc:
            print("Reloading encoder model as I should")
            encoder.load_state_dict(torch.load(args.encoder_path))
    
    if args.only_validate:
        only_validate(decoder, encoder, args, val_data_loader, criterion, vocab, val_transform)
        sys.exit("Finished validating!")    
   
    hidden = None
    # Train the models
    total_step = len(data_loader)
    losses = [] #summed losses between each log step
    interval_loss = 0
    k = 0 #number of truncations

    if args.force_forget:
         forget(decoder, args.rnn_type, args.hidden_size)


    #where model will save (using val error)
    fn, _ = getfn(args)
    model_name = 'decoder-' + fn
    model_name = model_name.replace('.npy', '.ckpt')

    #finally starting training
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            if args.rnn_type == 'DNI_LSTM':
                hidden = decoder.rnn.init_hidden(args.batch_size)
            else:
                hidden = None
            
            # Set mini-batch dataset
            images = images.float().to(device)
            captions = captions.to(device)
            features = encoder(images)
            captions_input = captions[:,:-1] #to be fed to model
            
            ntrunc = (captions.shape[1] - 1) // args.T + 1 
            for j in range(ntrunc):
                first_forward = j==0 
                hidden = _detach_hidden_state(hidden)
                optimizer.zero_grad()
                
                #first one receives image as well
                if j == 0:
                    captions_targ = captions[:, :args.T] #used as target
                    caption_trunc_model = captions_input[:, :args.T-1] #to be fed to model as input
                else:
                    captions_targ = captions[:,j*args.T:(j+1)*args.T]
                    caption_trunc_model = captions_input[:,j*args.T-1:(j+1)*args.T-1]
                
                
                with dni.defer_backward():
                    outputs_trunc, hidden = decoder(features, caption_trunc_model, hidden, first_forward = first_forward)
                    
                    if outputs_trunc.shape[1] > 0:
                        loss = criterion(outputs_trunc.reshape(-1, 9956), captions_targ.reshape(-1))
                        dni.backward(loss)

                        nseq = outputs_trunc.shape[1]
                        interval_loss += loss.item() * nseq
                        k += nseq
                optimizer.step()
                

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

                losses.append(interval_loss/k)
                interval_loss = 0
                k = 0
                       
        if args.validate:
            val_losses = validate_model(decoder, encoder, val_data_loader, criterion, val_losses, device)
            mean_val_losses = np.mean(np.array(val_losses).reshape(epoch + 1, -1), axis=1)
            if args.early_stopping and (np.argmin(mean_val_losses) == len(mean_val_losses) - 1):
                torch.save(decoder.state_dict(), os.path.join(args.model_path, model_name))
            decoder.train()
            if args.train_enc:
                encoder.train()
    losses = np.array(losses)
   
    fn, save_path = getfn(args)

    if args.recommence_training:
        fn_old, path_old = getfn(args, getorig=True)
        if not os.path.exists(path_old + fn_old):
            fn_old = fn_old.replace(".npy", "captrange-0.25_.npy") 
        losses_old = np.load(path_old + fn_old)
        losses = np.concatenate((losses_old, losses))
        
    np.save(save_path + fn, losses)

    if args.validate:
        val_losses = np.array(val_losses)
        if args.recommence_training:
            fn_val_old = "val_" + fn_old
            if os.path.exists(path_old + fn_val_old):    
                val_losses_old = np.load(path_old + fn_val_old)
                val_losses = np.concatenate((val_losses_old, val_losses))
            else:
                print("Can't find file {} in folder {}".format(fn_val_old, path_old))   
                    

        if args.validate_whole:
            val_fn = "valwhole_" + fn
        else:
            val_fn = "val_" + fn

        if args.val_same_transform:
            val_fn = "sametrans_" + val_fn
        np.save(save_path + val_fn, val_losses)

    if not args.early_stopping:
        torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, model_name))

        if args.train_enc:
            enc_model_name = 'encoder-' + fn
            enc_model_name = enc_model_name.replace('.npy', '.ckpt')
            torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, enc_model_name))    

        print("Saved model in ", args.model_path)  

    best_model_path = os.path.join(args.model_path, model_name)
    decoder.load_state_dict(torch.load(best_model_path))
    args.validate_position = False
    only_validate(decoder, encoder, args, val_data_loader, criterion, vocab, val_transform)
    args.validate_position = True
    only_validate(decoder, encoder, args, val_data_loader, criterion, vocab, val_transform)
 

def validate_model(decoder, encoder, val_data_loader, criterion, losses, device, lens=None, validate_position=False):
    decoder.eval()
    encoder.eval()

    if validate_position:
        losses = np.zeros((len(val_data_loader), 25)) #word position
        criterion.reduction = 'none'

    for i, (val_images, val_captions, lengths) in enumerate(val_data_loader):
         val_images = val_images.float().to(device)
         val_captions = val_captions.to(device)

         features = encoder(val_images)       
         captions_input = val_captions[:,:-1] 
         val_outputs, _ = decoder(features, captions_input, first_forward = True)
         loss = criterion(val_outputs.reshape(-1, 9956), val_captions.reshape(-1))

         if validate_position:
            orig_shape = val_captions.shape
            loss = loss.reshape(orig_shape).cpu().detach().numpy()
            loss_by_pos = np.mean(loss, axis=0)
            padwidth = losses.shape[1] - len(loss_by_pos)
            padded_loss = np.pad(loss_by_pos, mode='constant', pad_width=(0,padwidth), constant_values=-1)
            losses[i] = padded_loss
        
         else:
            losses.append(loss.item())
            if lens is not None:
                lens.append(lengths[0])
        
    return losses


def forget(model, model_type, nhid, bias=-20):
     #manually set forget gate if necessary 
    gate_min = -10
    gate_max = 0
    if model_type in ['DNI_LSTM', 'DNI_TANH']:
    #Set forget gate bias
        with torch.no_grad():
            model.rnn.rnn.bias_ih_l0[nhid:2*nhid].data.clamp_(gate_min,gate_max)
            model.rnn.rnn.bias_hh_l0[nhid:2*nhid].data.clamp_(gate_min, gate_max)          
            model.rnn.rnn.weight_ih_l0[nhid:2*nhid].data.clamp_(0, 0.1)
            model.rnn.rnn.weight_hh_l0[nhid:2*nhid].data.clamp_(0, 0.1)
 
    else:
        with torch.no_grad():
            model.rnn.bias_ih_l0[nhid:2*nhid].data.clamp_(gate_min, gate_max)
            model.rnn.bias_hh_l0[nhid:2*nhid].data.clamp_(gate_min, gate_max)          
            model.rnn.weight_ih_l0[nhid:2*nhid].data.clamp_(0, 0.1)
            model.rnn.weight_hh_l0[nhid:2*nhid].data.clamp_(0, 0.1)

#validate a pretrained model
def only_validate(decoder, encoder, args, val_data_loader, criterion, vocab, val_transform):
    capt_range = (args.capt_min, args.capt_max) 
    encoder.eval()
    decoder.eval()
    if args.save_lengths:
       val_capt_lengths = []
       train_capt_lengths = [] 
    else:
       val_capt_lengths = None
       train_capt_lengths = None    
 
    val_losses = []
    val_losses = validate_model(decoder, encoder, val_data_loader, criterion, val_losses, device, lens=val_capt_lengths, validate_position = args.validate_position)
 
    val_losses = np.array(val_losses)

    if args.recommence_training and args.decoder_path == '/home/oq19042/dni/image-captioning/models/decoder-5-3000.pkl':
        sys.exit("Shouldn't come into here")
    else:
        fn_old, val_path = getfn(args, getorig=True)

    if args.validate_whole:
        val_fn = "valwhole_" + fn_old
    else:
        val_fn = "val_" + fn_old

    val_path += 'val_only/' 

    if args.validate_position:
        val_fn = "bypos_" + val_fn
        val_path += "bypos/"

    if args.val_same_transform:
       val_fn = "sametrans_" + val_fn 

    np.save(val_path + val_fn, val_losses)   
        
    train_losses = []
    train_data_loader = get_loader(args.image_dir, args.caption_path, vocab, val_transform, args.val_batch_size, num_workers=args.num_workers, capt_range=capt_range, validate=not args.validate_whole, balance_data=args.balance_data, shuffle_data=False)

    train_losses = validate_model(decoder, encoder, train_data_loader, criterion, train_losses, device, lens=train_capt_lengths, validate_position = args.validate_position)
        
    train_losses = np.array(train_losses)
    if args.validate_whole:
        train_fn = "trainwhole_" + fn_old
    else:
        train_fn = "train_" + fn_old

    if args.validate_position:
        train_fn = "bypos_" + train_fn

    if args.val_same_transform:
        train_fn = "sametrans_" + train_fn 

    np.save(val_path + train_fn, train_losses) 
    print("Mean train loss:", np.mean(train_losses)) 
    if args.save_lengths:
        tname = "train_lengths.npy"
        vname = "val_lengths.npy"
        if args.validate_whole:
           tname = tname.replace('train', 'trainwhole')
           vname = vname.replace('val', 'valwhole')
        np.save(val_path + tname, np.array(train_capt_lengths)) 
        np.save(val_path + vname, np.array(val_capt_lengths))
    print("Saved training results as well (all in {})".format(val_path))   


#helper methods
def get_paramval(fn, param):
   
    if param == 'captrange':
        if param in fn:
            helper = re.findall('captrange' + '-' + r'\d+.\d+', fn)
            string = re.findall(r'\d+.\d+', helper[0])[0]
            gap = string.index('.')
            return (int(string[:gap]), int(string[gap+1:]))
        else:
            return (0, 25) #default vals     
    elif param in ['lr', 'dropout']:
        helper = re.findall(param + '-' + r'\d+.\d+', fn)
        string = re.findall(r'\d+.\d+', helper[0])[0]
        return float(string)
        
 
    helper = re.findall(param + '-' + r'\d+', fn)
    
    if len(helper) > 0:
        return int(re.findall(r'\d+', helper[0])[0])
    else:
        sys.exit("Unexpected param {}".format(param))
        return -2    


def checkargs(args):
    if args.only_validate:
        assert args.validate, "You want to only validate but validate=False"
        assert args.recommence_training, "You want to only validate but not select pretrained decoder?"
    else:
        assert not args.save_lengths, "You want to save length but are not only validating? Set save_lengths=False"

    if args.recommence_training:
        if args.decoder_path == '/home/oq19042/dni/image-captioning/models/decoder-5-3000.pkl':
            print("Reviewing pretrained decoder!")
            return args
        if 'trainenc' in args.decoder_path:
            args.train_enc = True
            enc_path = args.decoder_path.replace('decoder', 'encoder')
            args.encoder_path = enc_path
        else:
            args.train_enc = False

        if 'unbalanced' in args.decoder_path:
            print("Should be in here, setting args.balance_data to false")
            args.balance_data = False
            
        else:
            args.balance_data = True
        
        if 'DNI_LSTM' in args.decoder_path:
            args.rnn_type = 'DNI_LSTM'
        else:
            args.rnn_type = 'LSTM'

        args.T = get_paramval(args.decoder_path, 'T')
        args.hidden_size = get_paramval(args.decoder_path, 'nhid')
        args.synth_nhid = get_paramval(args.decoder_path, 'synthnhid') 
        args.seed = get_paramval(args.decoder_path, 'seed')
        args.capt_min, args.capt_max = get_paramval(args.decoder_path, 'captrange')
        args.dropout = get_paramval(args.decoder_path, 'dropout')
        if 'lr' in args.decoder_path:
            args.learning_rate = get_paramval(args.decoder_path, 'lr')

        if 'forceforget' in args.decoder_path:
            args.force_forget = True

        if args.rnn_type == 'DNI_LSTM': 
            if 'synthnlayers' in args.decoder_path: 
                args.synth_num_layers = get_paramval(args.decoder_path, 'synthnlayers')
            else:
                args.synth_num_layers = 2
        orig_fn, orig_path = getfn(args, getorig=True)

        if not args.only_validate and not os.path.exists(orig_path + orig_fn):
            print("First try:", orig_fn)
            orig_fn = orig_fn.replace(".npy", "captrange-0.25_.npy") 
            assert os.path.exists(orig_path + orig_fn), "Can't find original loss folder! Looking for {}".format(orig_path + orig_fn)

    if args.balance_data:
         args.model_path = args.model_path + 'balanced/'
    else:
         args.model_path = args.model_path + 'unbalanced/'
       
 
    if args.dropout > 0:
        args.model_path = args.model_path + 'dropout/'

    if args.l2_norm > 0:
        args.model_path = args.model_path + 'regularisation/'

    if args.early_stopping:
        args.model_path = args.model_path + 'earlystopping/'

    return args
            

def getfn(args, getorig=False):  
    if args.balance_data: 
        save_path = "/home/oq19042/dni/image-captioning/results/balanced/"
    else:
        save_path = "/home/oq19042/dni/image-captioning/results/unbalanced/"

    fn = str(args.rnn_type) + "_T-" + str(args.T) + "_nhid-" + str(args.hidden_size) + "_synthnhid-" + str(args.synth_nhid) + "_.npy"
    if args.recommence_training:
         nepochs_old = get_paramval(args.decoder_path, 'nepochs')
        
         if getorig:
            fn = fn.replace('.npy', 'nepochs-' + str(nepochs_old) + '_.npy')    
         else:
            fn = fn.replace('.npy', 'nepochs-' + str(args.num_epochs + nepochs_old) + '_.npy') 
    else: 
        fn = fn.replace('.npy', 'nepochs-' + str(args.num_epochs) + '_.npy') 
   
    fn = fn.replace('.npy', 'seed-' + str(args.seed) + '_.npy')
    if args.rnn_type == 'DNI_LSTM':
        fn = fn.replace('.npy', 'synthnlayers-' + str(args.synth_num_layers) + '_.npy')
    
    if args.capt_min !=0 or args.capt_max != 25:
        fn = fn.replace('.npy', 'captrange-' + str(args.capt_min) + '.' + str(args.capt_max) + '_.npy')

    if args.train_enc:
        save_path = save_path + "train_enc/"
        fn = fn.replace('.npy', 'trainenc_.npy')

    if args.dropout > 0:
        save_path = save_path + "dropout/"
        fn = fn.replace('.npy', 'dropout-' + str(args.dropout) + '_.npy')

    if args.l2_norm > 0:
        save_path = save_path + "regularisation/"
        fn = fn.replace('.npy', 'lreg-'+str(args.l2_norm)+'_.npy')

    if args.force_forget:
        fn = fn.replace('.npy', 'forceforget_.npy')

    if args.learning_rate != 0.001:
        fn = fn.replace('.npy', 'lr-'+str(args.learning_rate) + '_.npy')

    return fn, save_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #save/meta params
    parser.add_argument('--model_path', type=str, default='/home/oq19042/dni/image-captioning/models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='/scratch/oq19042/dni/image-captioning/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/scratch/oq19042/dni/image-captioning/data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='/scratch/oq19042/dni/image-captioning/data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=50, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--seed', type=int, default=11)
    
    # Model params
    parser.add_argument('--rnn_type', type=str , default='LSTM', help='type of RNN for the decoder. One of {LSTM, DNI_LSTM}')
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--force_forget', action='store_true', help='force bias on LSTM memory to encourage forgetting')
    parser.add_argument('--encoder_path', type=str, default='/home/oq19042/dni/image-captioning/models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/home/oq19042/dni/image-captioning/models/decoder-5-3000.pkl', help='path for trained decoder')
    
    #synthesiser params
    parser.add_argument('--synth_nhid', type=int , help='number of hidden units in synthesiser')
    parser.add_argument('--synth_num_layers', type=int , default=2, help='number of layers in synthesiser')
    parser.add_argument('--T', type=int , default=5, help='truncation size')
    

    #dataset params
    parser.add_argument('--capt_min', type=int, default=11, help='minimum length of captions considered')
    parser.add_argument('--capt_max', type=int, default=15, help='maximum length of captions considered')

    parser.add_argument('--train_enc', action='store_true',  help='train output parameters of encoder as well')

    parser.add_argument('--validate', action='store_true',  help='record validation loss as well')
    parser.add_argument('--val_image_dir', type=str, default='/scratch/oq19042/dni/image-captioning/data/val_resized2014', help='directory for validation resized images')
    parser.add_argument('--val_caption_path', type=str, default='/scratch/oq19042/dni/image-captioning/data/annotations/captions_val2014.json', help='path for validation annotation json file')
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--balance_data', action='store_true', help='ensure that each caption length has the same number of training examples')    
    

    #training params
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--only_validate', action='store_true', help='only validate, do no train/save model')
    
    parser.add_argument('--validate_whole', action='store_true', help='validation on entire dataset')
    parser.add_argument('--validate_position', action='store_true', help='find word by word loss')


    parser.add_argument('--dropout', type=float, default=0.0,
                        help='the drop rate for the *input* layer of the RNN')

    parser.add_argument('--l2_norm', type=float, default=0,
                        help='weight of L2 norm')

    parser.add_argument('--val_same_transform', action='store_true', help='validate model with same image transform as for training')
    parser.add_argument('--early_stopping', type=bool, default=True, help='Save model according to early stopping (as opposed to at the end)')
    parser.add_argument('--no_early_stopping', action='store_false', dest='early_stopping', help='Save model at end of training')
    parser.add_argument('--save_lengths', action='store_true', help='Save validation/training lengths when validating only')
    parser.add_argument('--recommence_training', action='store_true', help='recommence training from saved models') 

 
    args = parser.parse_args()
    
    if args.rnn_type in ['LSTM']:
        args.synth_nhid = 0
    print(args)
    main(args)
