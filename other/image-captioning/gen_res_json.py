"""
generate model predicted captions
"""

from build_vocab import Vocabulary
import torch
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 

from PIL import Image
import re
import sys

from model import EncoderCNN, DecoderRNN
from data_loader import get_loader

import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    data = [] #before anything else (save memory quickly?)

    #check you haven't already generated captions for this model
    save_path = '/home/oq19042/dni/image-captioning/gencapts/unbalanced/dropout/'
    dataType='val2014'
    if 'synthnlayers' in args.decoder_path:
        if 'nfibres' in args.decoder_path:
            model = 2
        else:
            model = 1
    else:
        model = 0  
    T = get_paramval(args.decoder_path, 'T')
    seed = get_paramval(args.decoder_path, 'seed')
    algName = get_algname(model, T, seed)
    ann_file = 'captions_{}_{}_results.json'.format(dataType,algName)

    if T == 6 and seed == 1308 and model == 1:
        print("DNI is a bad model for this case! T=6, seed=1308, returning early")
        #return

    if os.path.exists(save_path+ann_file):
        print("Ann file {} already exists. Returning early".format(ann_file))
        return
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    #return vocab
    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, rnn_type=args.rnn_type, 
                         synth_nhid=args.synth_nhid, synth_nlayers=args.synth_num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    
    
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device('cpu')))    
    decoder.eval()
    
    
    #load data
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.test_size,
                             num_workers=2, gen_capts=True) 
    

    image_ids = data_loader.dataset.coco.getImgIds()

    i = 0
    for batch, (images, _, _) in enumerate(data_loader):
        
        if batch % 20 == 0:
            print("Batch number {}".format(batch+1))
       
        images = images.float().to(device) 
        features = encoder(images)
        sampled_ids_all = decoder.sample(features)
        sampled_ids_all = sampled_ids_all.cpu().numpy()        
        
        for sampled_ids in sampled_ids_all:
            
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption[1:-1])
            
            im_id = image_ids[i]            
            i += 1
            
            entry = {"image_id": im_id, "caption": sentence}
            data.append(entry)

            
    print("Finished generating captions. Now saving them")


    
    with open(save_path+ann_file, 'w') as outfile:
        json.dump(data, outfile)                    
    print("All saved in {} (file {})".format(save_path, ann_file))     
    
    
    

def get_algname(model, T, seed):
    mname = ['LSTM', 'DNI_LSTM', 'DNI_LSTM'][model]  
    algname = '{}_T-{}_seed-{}'.format(mname, T, seed)
    if model == 2:
        algname += '_nfibres-4'
    
    return algname


def get_mname(model, T, seed):

    
    mgeneric = 'decoder-mtype_T-Tval_nhid-256_synthnhid-snhidval_nepochs-10_seed-seedval'
    mgeneric = mgeneric.replace('mtype', ['LSTM', 'DNI_LSTM', 'DNI_LSTM'][model])
    mgeneric = mgeneric.replace('Tval', str(T))
    mgeneric = mgeneric.replace('seedval', str(seed))
    
    if model == 0:
        mgeneric = mgeneric.replace('snhidval', str(0))
    else:
        mgeneric = mgeneric.replace('snhidval', str(1024))
    
    if model > 0:
        mgeneric += '_synthnlayers-2'
    
    mgeneric += '_dropout-0.5'
    
    if model == 2:
        mgeneric += '_nfibres-4'
        
    mgeneric += '_.ckpt'

    return mgeneric

def get_paramval(fn, param):
    
    helper = re.findall(param + '--' + r'\d+', fn)
    if len(helper) > 0:
        assert int(re.findall(r'\d+', helper[0])[0]) == 1
        return -1
    
    helper = re.findall(param + '-' + r'\d+', fn)
    
    if len(helper) > 0:
        return int(re.findall(r'\d+', helper[0])[0])
    else:
        return -2

def sort_args(args):       
    args.hidden_size = get_paramval(args.decoder_path, 'nhid')
    args.synth_nhid = get_paramval(args.decoder_path, 'synthnhid')
    args.synth_num_layers = get_paramval(args.decoder_path, 'synthnlayers')
    args.rnn_type = ['LSTM', 'DNI_LSTM'][args.synth_nhid > 0]   
    return args
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='png/example.png', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='/home/oq19042/dni/image-captioning/models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='home/oq19042/dni/image-captioning/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='/scratch/oq19042/dni/image-captioning/data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--rnn_type', type=str , default='LSTM', help='type of RNN for the decoder. One of {LSTM, DNI_LSTM}')    
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--synth_nhid', type=int , default=1, help='number of hidden units in synthesiser')
    parser.add_argument('--synth_num_layers', type=int , default=1, help='number of hidden layers in synthesiser')
    
    parser.add_argument('--image_dir', type=str, default='/scratch/oq19042/dni/image-captioning/data/val_resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='/scratch/oq19042/dni/image-captioning/data/annotations/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--test_size', type=int, default=300)
    
    
    parser.add_argument('--do_all', action='store_true')
    parser.add_argument('--Tmin', type=int, default=3, help='min T value for checked models')
    parser.add_argument('--Tmax', type=int, default=7, help='max T value for checked models')
    
    args = parser.parse_args()
    
    root = 'path_to_save/'

    if args.do_all:
        seeds = [11, 1308, 42, 240, 7532]
        seeds = [9826, 4098, 39184, 1745, 63033]
        Trange = np.arange(args.Tmin, args.Tmax+1)
        for seed in seeds:
            for T in Trange:
                for model in [0, 1, 2]:
                    mname = get_mname(model, T, seed)
                    args.decoder_path = root + mname
                    if not os.path.exists(args.decoder_path):
                        model_type = ['LSTM', 'DNI', 'SPARSE_DNI'][model]
                        print("Model {} for T={}, seed={} doesn't exist. Continuing".format(model_type,
                              T, seed))
                        continue
                    args = sort_args(args)
                    main(args)
    else:
        if not os.path.exists(args.decoder_path):
            sys.exit("Can't find model (filename {}). Exiting".format(args.decoder_path))
        args = sort_args(args)
        main(args)

    print("Done.")
                    
