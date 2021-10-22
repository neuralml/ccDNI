import torch
import nltk
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
#import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO

import sys

from torch.utils.data.sampler import Sampler

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


#sampler which goes over data according to caption length
class my_sampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, capt_lengths, batch_size, capt_range=(11, 15), validate=False, balance_data=True, shuffle_data=True):
        if isinstance(capt_lengths, list):
            self.capt_lengths = np.array(capt_lengths)
        else:
            self.capt_lengths = capt_lengths
        self.batch_size = batch_size
        
        if isinstance(capt_range, tuple):
            self.capt_range = np.arange(capt_range[0], capt_range[1]+1)
        else:
            self.capt_range = capt_range

        self.validate = validate
        self.nvalbatch = 50 #hardcoded for now
        self.balance_data = balance_data
        self.shuffle_data = shuffle_data

        self.epoch = 0

        self.make_order()

    def __iter__(self):
        self.epoch += 1 #give each epoch a different order
        if self.epoch > 1:
            self.make_order()       
        return iter(self.order)

    def __len__(self):
        return len(self.order)
    
    def make_order(self):
        if self.shuffle_data:
            seed = (self.epoch + 34) * 14
        else:
            seed = 123
        np.random.seed(seed)

 
        print("Making the order list!")
        capt_inds = [list(np.argwhere(self.capt_lengths == capt_length).squeeze(1)) for capt_length in self.capt_range]

        if self.shuffle_data or self.balance_data:
            for x in capt_inds:
                np.random.shuffle(x)

        if self.balance_data:
            ninds = [int(len(x)/self.batch_size) for x in capt_inds]
            nbatch = min(ninds)
            print("Balancing data so that each caption will have {} batches".format(nbatch))
        
        order = []
        
        #checking state for caption length
        notfinished = [True] * len(self.capt_range)
        
        bnumber = 0
        while sum(notfinished) != 0:
            for i, capt_length in enumerate(self.capt_range):
                if notfinished[i]:
                    inds = capt_inds[i][bnumber*self.batch_size:(bnumber+1)*self.batch_size]
                    #print(capt_inds[i])
                    #print("Capt size {} has {} inds".format(capt_length, len(inds)))
                    if len(inds) == self.batch_size:                        
                        order.append(inds)
                    else:
                        notfinished[i] = False
                        print("There will be {} batches for captions of length {}".format(bnumber+1, capt_length))
                    
                    if self.balance_data and bnumber == nbatch - 1:
                        notfinished[i] = False
                        print("There will be {} batches for captions of length {} (balancing data)".format(bnumber+1, capt_length))

                    
            bnumber += 1

        if self.shuffle_data:
            print("Shuffling data with seed {}!".format(seed))
            np.random.shuffle(order)

        if self.validate:
            np.random.shuffle(order)
            order = order[:self.nvalbatch]
 
        self.order = order


class generic_sampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, order):
        self.order = order
    def __iter__(self):
        return iter(self.order)
    def __len__(self):
        return len(self.order)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, num_workers, capt_range=(11,15), validate=False, balance_data=True, shuffle_data=True, gen_capts=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    #data_loader = torch.utils.data.DataLoader(dataset=coco, 
    #                                          batch_size=batch_size,
    #                                           shuffle=shuffle,
    #                                           num_workers=num_workers,
    #                                           collate_fn=collate_fn)

    if gen_capts:
        unique_ims = np.load('/home/oq19042/dni/image-captioning/code/unique_ims_indices.npy')
        sampler = generic_sampler(unique_ims)
        data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    else:
        sampler = my_sampler(capt_lengths=coco.coco.capt_lengths, batch_size=batch_size, capt_range=capt_range, validate=validate, balance_data=balance_data, shuffle_data=shuffle_data)    
        data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader




#JOP
def get_lengthsmat(lengths, T):
    
    #lengthsmat = np.array(lengths) - 1 #super subtle the -1, because we don't want to include the last caption word (<end>) into the model
    
    maxseqlen = np.max(lengths)

    nrows = maxseqlen // T + 1 #for each truncation
    ncols = len(lengths) #for each length
    
    lengthsmat = np.zeros((nrows, ncols))
    for i, length in enumerate(lengths):
        ndiv = length // T
        lengthsmat[:ndiv,i] = T
        lengthsmat[ndiv,i] = length % T #remainder 
    
    return lengthsmat
