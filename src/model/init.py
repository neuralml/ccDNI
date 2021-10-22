import os
import yaml
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.jit as jit

from .nn_dni import RNNdni 

###############################################################################
# RNN WRAPPERS
###############################################################################

class RNNPredictor(nn.Module):
    def __init__(self, rnn, output_size, predict_last=True):
        super(RNNPredictor, self).__init__()
        self.rnn = rnn
        self.linear = nn.Linear(rnn.hidden_size, output_size)
        self.predict_last = predict_last

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.linear.reset_parameters()

    @property
    def input_size(self):
        return self.rnn.input_size

    @property
    def output_size(self):
        return self.linear.weight.shape[0]

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    @property
    def n_layers(self):
        return self.rnn.num_layers

    def forward(self, input, hidden=None, contexts=[None,None]):          
        if self.context:
            output, hidden = self.rnn(input, hidden, contexts=contexts)
        else:
            output, hidden = self.rnn(input, hidden)
        
        if self.predict_last: #only take final output
            pred = self.linear(output[:, -1, :])
        else:            
            pred = self.linear(output.reshape(output.size(0)*output.size(1), output.size(2)))
            pred = pred.reshape(output.size(0), output.size(1), pred.size(1))

        return pred, hidden


###############################################################################
# Model Initialization
###############################################################################


def _weight_init_(module, init_fn_):
    if isinstance(module, nn.Linear):
        init_fn_(module.weight.data)
    else:
        try:
            for layer in module.all_weights:
                w, r = layer[:2]
                init_fn_(w)
                init_fn_(r)
        except AttributeError:
            pass


def weight_init_(rnn, mode=None, **kwargs):
    if mode == 'xavier':
        _weight_init_(rnn, lambda w: init.xavier_uniform_(w, **kwargs))
    elif mode == 'orthogonal':
        _weight_init_(rnn, lambda w: init.orthogonal_(w, **kwargs))
    elif mode == 'kaiming':
        _weight_init_(rnn, lambda w: init.kaiming_uniform_(w, **kwargs))
    elif mode != None:
        raise ValueError(
                'Unrecognised weight initialisation method {}'.format(mode))


def init_model(model_type, hidden_size, input_size, n_layers,
        output_size, dropout=0.0, weight_init=None, device='cpu',
        predict_last=True, script=False, synth_nlayers=0, synth_nhid=None
    ):
    if model_type == 'LSTM':
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    elif model_type == 'GRU':
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
    
    #JOP - dni rnn
    elif model_type == 'TANH':
        rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=dropout
        )
    elif model_type in ['DNI_TANH', 'cDNI_TANH']:   
        
        if model_type == 'DNI_TANH':
            context_dim = None
        else:
            context_dim = output_size
            
        rnn = RNNdni(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            rnn_type='RNN_TANH',
            batch_first=True,
            use_dni=True,
            dropout=dropout, 
            context_dim=context_dim,
            synth_nlayers = synth_nlayers,
            synth_nhid=synth_nhid
        ) 
    elif model_type in ['DNI_LSTM', 'cDNI_LSTM']:

        if model_type == 'DNI_LSTM':
            context_dim = None
        else:
            context_dim = output_size
            
        rnn = RNNdni(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            rnn_type='LSTM',
            batch_first=True,
            use_dni=True,
            dropout=dropout, 
            context_dim=context_dim,
            synth_nlayers = synth_nlayers,
            synth_nhid=synth_nhid
        )     
    else:
        raise ValueError('Unrecognized RNN type')

    weight_init_(rnn, weight_init)

    model = RNNPredictor(
        rnn=rnn,
        output_size = output_size,
        predict_last=predict_last
    ).to(device=device)

    model.model_type = model_type
    model.context = model_type in ['cDNI_TANH', 'cDNI_LSTM']

    model.ablation_epoch = -1 

    return model

#  Load models
###############################################################################
def load_meta(path):
    with open(path, mode='r') as f:
        meta = yaml.safe_load(f)
    return meta


def _load_model(meta, model_file):
    meta = load_meta(meta)
    with open(model_file, mode='rb') as f:
        state_dict = torch.load(f)
        if 'model-state' in state_dict:
            state_dict = state_dict['model-state']
    m = init_model(device='cpu', **meta['model-params'])
    m.load_state_dict(state_dict)

    return m


def load_model(model_folder):
    meta = os.path.join(model_folder, 'meta.yaml')
    for file in os.listdir(model_folder):
        if file.endswith((".pt", ".pth")):
            file = os.path.join(model_folder, file)
            return _load_model(meta, file)
