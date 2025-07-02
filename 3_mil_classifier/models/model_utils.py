import torch.nn as nn
import torch.distributed as dist
import torch

import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)  
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):  
            nn.init.constant_(m.weight, 1)  
            nn.init.constant_(m.bias, 0)


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()



class CoxPHLoss(torch.nn.Module):  
    def __init__(self, reduction='mean'):
        super(CoxPHLoss, self).__init__()

    def forward(self, preds, durations, events):
        """
        Calculate the Cox partial log likelihood.
        :param preds: The model predictions (risk scores).
        :param durations: The observed survival times.
        :param events: The event indicators (1 if event occurred, 0 for censored).
        """
        risk_scores = preds.reshape(-1)
        events = events.reshape(-1)
        durations = durations.reshape(-1)
        
        # Calculate risk scores for all
        exp_risk_scores = torch.exp(risk_scores)
        
        # Sort by survival time
        durations, sorted_idx = torch.sort(durations, descending=True)
        exp_risk_scores = exp_risk_scores[sorted_idx]
        events = events[sorted_idx]
        
        # Calculate the log sum of risks for the risk set
        accumulated_risk_scores = torch.cumsum(exp_risk_scores, dim=0)
        log_risk = torch.log(accumulated_risk_scores)
        
        # Only consider the instances where the event occurred
        observed_risk_scores = risk_scores[events == 1]
        observed_log_risk = log_risk[events == 1]
        
        # Calculate negative partial log likelihood
        neg_partial_log_likelihood = -torch.sum(observed_risk_scores - observed_log_risk)
        
        return neg_partial_log_likelihood

# >>>>>>>>>>>>>>>>>>> Codebase from YuanWei (March 26, 2024) >>>>>>>>>>>>>>>>>>> #

# evaluation for C-Index
# from sksurv.metrics import concordance_index_censored
# cindex = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CoxSurvLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(CoxSurvLoss, self).__init__()

    def forward(self, hazards, time, c):
        '''
        # hazards: Risk value of model output (log risk)
        # time:Time of event occurrence or observation timeence time or observation time
        # c: Event occurrence status (1 indicates event occurred, 0 indicates censored)
        '''

        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        
        hazards = hazards.squeeze()

        current_batch_len = len(time)

        R_mat = torch.zeros(
            [current_batch_len, current_batch_len], 
            dtype=int, 
            device=hazards.device
            )

        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = time[j] >= time[i]
                
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * c)

        return loss_cox

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    
    #!! here uncensored_loss means event happens(death/progression)
    # uncensored_loss = -c * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    # censored_loss = - (1 - c) * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    # neg_l = censored_loss + uncensored_loss
    # loss = (1-alpha) * neg_l + alpha * uncensored_loss
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -c * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - (1 - c) * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss



def weighted_multi_class_log_loss(y_hat, y, w, classes=2):
    a = torch.log(torch.clamp(y_hat, 1e-15, 1.0 - 1e-15)).cuda()
    if classes == 2:
        b = torch.tensor([torch.clamp(torch.sum(y[:,0]), min=1e-15), torch.clamp(torch.sum(y[:,1]), min=1e-15)]).cuda()
    elif classes==5:
        b = torch.tensor([torch.clamp(torch.sum(y[:,0]), min=1e-15), torch.clamp(torch.sum(y[:,1]), min=1e-15),
                          torch.clamp(torch.sum(y[:,2]), min=1e-15),torch.clamp(torch.sum(y[:,3]), min=1e-15),torch.clamp(torch.sum(y[:,4]), min=1e-15),
                          ]).cuda()
    return torch.sum(-torch.sum(w * y * a * 1/b))

class WeightedMultiClassLogLoss(torch.nn.Module):
    def __init__(self, weights=torch.tensor([1.,1.]), classes=2):
        super(WeightedMultiClassLogLoss, self).__init__()
        self.weights = weights.cuda()
        self.classes = classes

    def forward(self, inputs, targets):
        return weighted_multi_class_log_loss(inputs, targets, self.weights, classes=self.classes)



if __name__ == "__main__":
    pass
