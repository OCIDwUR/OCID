from __future__ import absolute_import

import os
import torch
import shutil
import numpy as np
from torch import nn
from torch.autograd import Variable

from os.path import join as ospj
from os.path import exists as ospe

class CheckPointer(object):
    def __init__(self, args, model=None, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_path = ospj(ospj(os.getcwd(), 'backbones'), args['model.name'])
        self.last_ckpt = ospj(self.model_path, 'checkpoint.pth.tar')
        self.best_ckpt = ospj(self.model_path, 'model_best.pth.tar')

    def restore_model(self, ckpt='last', model=True,
                      optimizer=True, strict=True):
        if not os.path.exists(self.model_path):
            assert False, "Model is not found at {}".format(self.model_path)
        self.last_ckpt = ospj(self.model_path, 'checkpoint.pth.tar')
        self.best_ckpt = ospj(self.model_path, 'model_best.pth.tar')
        ckpt_path = self.last_ckpt if ckpt == 'last' else self.best_ckpt

        if os.path.isfile(ckpt_path):
            print("=> load {} ckpt '{}'".format(ckpt, ckpt_path), end='\r')
            ch = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            if self.model is not None and model:
                self.model.load_state_dict(ch['state_dict'], strict=strict)
            if self.optimizer is not None and optimizer:
                self.optimizer.load_state_dict(ch['optimizer'])
        else:
            assert False, "No checkpoint! %s" % ckpt_path

        return ch.get('epoch', None), ch.get('best_val_loss', None), ch.get('best_val_acc', None)

    def save_checkpoint(self, epoch, best_val_acc, best_val_loss,
                        is_best, filename='checkpoint.pth.tar',
                        optimizer=None, state_dict=None, extra=None):
        state_dict = self.model.state_dict() if state_dict is None else state_dict
        state = {'epoch': epoch + 1,
                 'args': self.args,
                 'state_dict': state_dict,
                 'best_val_acc': best_val_acc,
                 'best_val_loss': best_val_loss}

        if extra is not None:
            state.update(extra)
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()

        model_path = check_dir(self.model_path, True, False)
        torch.save(state, ospj(model_path, filename))
        if is_best:
            shutil.copyfile(ospj(model_path, filename),
                            ospj(model_path, 'model_best.pth.tar'))
    

class CosineClassifier(nn.Module):
    def __init__(self, n_feat, num_classes):
        super(CosineClassifier, self).__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(n_feat, num_classes).normal_(
                    0.0, np.sqrt(2.0 / num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
        weight = torch.nn.functional.normalize(self.weight, p=2, dim=0, eps=1e-12)
        cos_dist = x_norm @ weight
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineClassifier: input_channels={}, num_classes={}; learned_scale: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s


def save_checkpoint(state, result_model_path, model_name, is_best=False):
    if not ospe(result_model_path):
        os.makedirs(result_model_path)

    ep = 'last_ep_'
    torch.save(state, ospj(result_model_path, model_name + ep + 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(ospj(result_model_path, model_name + ep + 'checkpoint.pth.tar'),
                        ospj(result_model_path, model_name + 'model_best.pth.tar'))


def load_checkpoint(model, result_model_path, model_name):
    checkpoint = torch.load(ospj(result_model_path, model_name+'model_best.pth.tar'),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def load_last_checkpoint(model, result_model_path, model_name):
    checkpoint = torch.load(ospj(result_model_path, model_name+'last_ep_checkpoint.pth.tar'),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    return model


class UIIDCL_BatchHard(nn.Module):
    def __init__(self, margin=0, hard_type='hard'):
        super(UIIDCL_BatchHard, self).__init__()
        self.margin = margin
        if margin == 0:
            print('soft margin loss')
            self.ranking_loss = nn.SoftMarginLoss()
        else:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin) 
        self.hard_type = hard_type
        assert self.hard_type == 'hard'

    def forward(self, inputs, targets):
        # inputs: features, targets: labels
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            pos = dist[i][mask[i]]
            neg = dist[i][mask[i] == 0]
            dist_ap.append(pos.max().reshape(-1))  
            dist_an.append(neg.min().reshape(-1))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1) 
        y = Variable(y)

        if self.margin == 0:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec, y.size(0)
    

def model_name_generation(args):
    modelname_suffix = '-imrs_'+str(args['img_resize']) + '-' + args['model.backbone']

    if args['model.UIIDCL_bh_margin'] != 0.8:
        modelname_suffix += '-bh_m({:.1f})'.format(args['model.UIIDCL_bh_margin'])
    
    model_name = 'OCID_case_{}{}'.format(args['case'], modelname_suffix)
    
    return model_name, modelname_suffix


def check_dir(dirname, verbose=True):
    """This function creates a directory
    in case it doesn't exist"""
    try:
        # Create target Directory
        os.makedirs(dirname)
        # if verbose:
        #     print("Directory ", dirname, " is now created")
    except FileExistsError:
        # if verbose:
        #     print("Directory ", dirname, " already exists")
        None
    return dirname
