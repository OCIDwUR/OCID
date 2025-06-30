"""
model architecture
"""

import torch
import numpy as np
import torch.nn as nn
from resnet18 import resnet18
from functools import partial
from torch.nn import functional as F

from config import DATASET_MODELS_RESNET18
from model_utils import CheckPointer


eps = 1e-5

class ASNet(nn.Module):
    def __init__(self, args, in_dim=None, out_dim=None, mvit_vle=False):
        # ele_num：attention的子单元的个数
        super(ASNet, self).__init__()

        if mvit_vle is False:
            n_hidden = args['model.ASNet_hidden_dim']
            self.as_net = nn.Sequential(
                    nn.Linear(in_dim, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, out_dim))
        else:
            self.as_net = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        x = self.as_net(x)
        return x


class OCID_UR_Net(nn.Module):
    def __init__(self, args):
        super(OCID_UR_Net, self).__init__()
                    
        bb_num = args['model.backbone_num']
        mvit_in_dim = 9 * args['model.fts_dim'] # feature map size 3*3
        self.mvit_qnet = ASNet(args, in_dim=mvit_in_dim*bb_num, out_dim=mvit_in_dim)
        self.mvit_knet = ASNet(args, in_dim=mvit_in_dim, out_dim=mvit_in_dim)

        self.dropout = None        
        if args['model.dropout_en'] is True:
            self.dropout = torch.nn.Dropout(p=0.1)
        
        if args['model.init']:
            print('weight_init')
            self._weight_init_()

    def _weight_init_(self):
        for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def model_trans(self, input_a, input_b):
        # attention on backbone features

        key = self.mvit_knet(input_a)
        qry = self.mvit_qnet(input_b.reshape(input_b.shape[0], -1))
        beta = torch.bmm(key, qry.unsqueeze(2)) / np.sqrt(key.shape[-1])
        a_score = F.softmax(beta, dim=1)
        out_fts = torch.mean(input_a * a_score.repeat(1, 1, input_a.shape[-1]), dim=1)  

        return out_fts

    
    def forward(self, exmp_fts=None, sear_fts=None, test=False):            

        d0, d1 = exmp_fts.shape[:2]
        exmp_fts = exmp_fts.reshape([d0, d1, -1])
        d0, d1 = sear_fts.shape[:2]
        sear_fts = sear_fts.reshape([d0, d1, -1])

        exmp_fts = self.model_trans(exmp_fts, exmp_fts)
        sear_fts = self.model_trans(sear_fts, sear_fts)

        if test is True:
            exmp_fts, sear_fts = exmp_fts.cpu(), sear_fts.cpu()
            return exmp_fts, sear_fts

        return exmp_fts, sear_fts


def get_multinet_extractor(args):
    extractors = dict()
    domain_names = DATASET_MODELS_RESNET18
    for domain, domain_net in domain_names.items():
        args['model.name'] = domain_net
        model_fn = partial(resnet18, dropout=args.get('model.dropout', 0), ext_layer=args['fts_layer'])
        extractor = model_fn()
        # extractor.cuda()
        checkpointer = CheckPointer(args, extractor, optimizer=None)
        extractor.eval()
        checkpointer.restore_model(ckpt='best', strict=False)
        extractors[domain] = extractor

    def embed_many(images):
        with torch.no_grad():
            all_features = []
            for _, extractor in extractors.items():
                all_features.append(extractor.embed(images).unsqueeze(1).cpu())
            all_features = torch.cat(all_features, dim=1).cpu()
        return all_features
        
    return embed_many



class AverageMeter(object):
    # Computes and stores the average and current value

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
