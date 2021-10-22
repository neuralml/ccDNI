import torch
import torch.nn as nn
import torch.nn.modules.loss as L
import torch.nn.functional as F
import ignite.metrics as M

class RNNLossWrapper(L._Loss):
    def __init__(self, loss_fn, reduction='mean'):
        super(RNNLossWrapper, self).__init__(reduction=reduction)
        self._loss_fn = loss_fn

    def forward(self, input, target):
        pred, _ = input
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.shape[2])     

            if target.dim() == 3: 
                target = target.reshape(-1, target.shape[2])
            else:
                target = target.view(-1)
        return self._loss_fn(pred, target)


class RateDecay(L._Loss):
    def __init__(self, target=0, reduction='mean', device='cpu'):
        super(RNNHiddenL1Loss, self).__init__(reduction)

        self.target = torch.as_tensor(target).to(device)

    def forward(self, input, target):
        _, hidden = input

        return F.l1_loss(hidden, self.target, reduction=self.reduction)


class L1WeightDecay(L._Loss):
    def __init__(self, model_parameters, target=0, device='cpu'):
        super(L1WeightDecay, self).__init__('none')
        self.parameters = model_parameters
        self.target = torch.as_tensor(target).to(device)

    def forward(self, input, target):
        return F.l1_loss(self.parameters, self.target, reduction='none')


class ComposedLoss(L._Loss):
    def __init__(self, terms, weights):
        super(ComposedLoss, self).__init__(reduction='none')

        self.terms = terms
        self.weights = weights

    def forward(self, input, target):
        value = 0
        for term, w in zip(self.terms, self.weights):
            value += w * term(input, target)

        return value


def get_loss_fn(loss_fn_name):
    if loss_fn_name == 'mse':
        return nn.MSELoss()
    elif loss_fn_name == 'xent':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function {}'.format(loss_fn_name))


class certainty_loss_fn(L._Loss):
    def __init__(self):      
        super(certainty_loss_fn, self).__init__()
              
    def forward(self, y_pred, y):
        softmax = torch.softmax(y_pred, dim=1)
        max_soft = torch.max(softmax, dim=1)[0]
        cert = torch.mean(max_soft)
        
        return cert

def get_metric(metric):
    if metric == 'mse':
        return M.MeanSquaredError()
    elif metric == 'xent':
        return M.Loss(nn.CrossEntropyLoss())
    elif metric == 'acc':
        return M.Accuracy()
    elif metric == 'certainty':
        return M.Loss(certainty_loss_fn())
    raise ValueError('Unrecognized metric {}.'.format(metric))


def init_metrics(metrics, rate_reg=0.0, rnn_eval=True):
    criterion = get_loss_fn(metrics[0])

    if rnn_eval:
        criterion = RNNLossWrapper(criterion)

    if rate_reg > 0:
        criterion = ComposedLoss(
            terms=[criterion, RateDacay(device=device)],
            decays=[1.0, rate_reg]
        )

    metrics = {m: get_metric(m) for m in metrics}

    return criterion, metrics
