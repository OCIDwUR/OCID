"""
One-shot Cross-domain Instance Detection With Universal Representation

Test model UIIDCL
June 29th, 2025
"""

import torch
import numpy as np

eps = 1e-5

def test(model, gt_mat, mat_str, mat_w_h, exmp, sear, sim_mode='cos', device_=None):

    exmp_num = exmp.shape[0]

    with torch.no_grad():         
        model.eval().cpu()
        exmp_ = exmp.split(int(exmp.shape[0]/2)+1, dim=0)
        sear_ = sear.split(int(sear.shape[0]/len(exmp_)), dim=0)
        exmp, sear = torch.tensor([]), torch.tensor([])
        for i in range(len(exmp_)):
            exmp_1, sear_1 = model(exmp_fts=exmp_[i], sear_fts=sear_[i], test=True)
            exmp = torch.cat((exmp, exmp_1), dim=0)
            sear = torch.cat((sear, sear_1), dim=0)
            print('{:03d}'.format(i))
            exmp_1, sear_1 = [], []
        exmp_, sear_ = [], []
        model.to(device_)

    print('SHAPE:', exmp.shape, sear.shape)
 
    exmp = torch.nn.functional.normalize(exmp, dim=-1)
    sear = torch.nn.functional.normalize(sear, dim=-1)
    exmp = torch.split(exmp, 1, dim=0)
    sear = torch.split(sear, 400, dim=0)
    
    pd_mat = np.zeros((exmp_num, 5))
    for exmp_i in range(exmp_num):
        current_exmp = exmp[exmp_i]
        current_sear = sear[exmp_i]
        score = metric(current_exmp, current_sear, mode=sim_mode)
        boxes = from_index_to_boxes(score, exmp_i, mat_str, mat_w_h)

        boxes = NMS(boxes)
        score = boxes[:,4]

        order = score.argsort()[::-1]
        topbox = boxes[order[0], :]
        pd_mat[exmp_i, :] = inverse_projection(topbox, mat_w_h, exmp_i)

    mAP = get_mAP(pd_mat, gt_mat)
    return mAP


def metric(exmp, sear, mode='cos'):
    # exmp: vector, sear: matrix

    assert mode in ['cos', 'ncc', 'sad', 'ssd']
    eps = 1e-5

    if mode == 'cos':
        score = torch.mm(sear, exmp.reshape(-1,1))
    
    if mode == 'ncc':
        exmp -= exmp.mean()
        sear -= sear.mean(dim=1).reshape(-1,1).repeat(1, sear.shape[1])
        denominator = ((sear**2).sum(dim=1) * (exmp**2).sum()).sqrt()
        score = torch.mm(sear, exmp.reshape(-1,1)) / denominator.reshape(-1,1)
    
    if mode == 'sad':
        sear -= exmp.reshape(1,-1).repeat(sear.shape[0], 1)
        score = 1 / (sear.abs().sum(dim=1) + eps)
    
    if mode == 'ssd':
        sear -= exmp.reshape(1,-1).repeat(sear.shape[0], 1)
        score = 1 / ((sear**2).sum(dim=1) + eps)

    return score.squeeze().numpy()


def from_index_to_boxes(score, exmp_i, mat_str, mat_w_h):
    numi, numj = 20, 20
    t_w, t_h = mat_w_h[exmp_i, 4:]

    index = np.arange(score.shape[0])
    boxes = np.zeros((score.shape[0], 5))

    stri, strj = mat_str[exmp_i]
    boxes[:, 0] = strj * (index % numj) 
    boxes[:, 1] = stri * np.floor(index/numj)
    boxes[:, 2] = boxes[:, 0] + t_w - 1
    boxes[:, 3] = boxes[:, 1] + t_h - 1
    boxes[:, 4] = score

    return boxes


def NMS(dets, thresh=0.5):
    # x1、y1、x2、y2、score
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # area for candidate windows
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    boxes = dets[keep]
    return boxes


def inverse_projection(box, mat_w_h, exmp_i):
    t_w, t_h, inv_w, inv_h = mat_w_h[exmp_i, 2:]

    if t_w == inv_w and t_h == inv_h:
        return box

    a, b = inv_h/t_h, inv_w/t_w
    center = (box[0:2] + box[2:4]) / 2
    box[0] = np.floor(center[0]/b - t_w/2)
    box[1] = np.floor(center[1]/a - t_h/2)
    box[2] = box[0] + t_w - 1
    box[3] = box[1] + t_h - 1

    return box


def get_mAP(a, b, qualitative=False):
    # validate
    check_w = (a[:,2]-a[:,0])==(b[:,2]-b[:,0])
    check_h = (a[:,3]-a[:,1])==(b[:,3]-b[:,1])
    assert np.sum(check_w) == np.sum(check_h) == a.shape[0], 'wrong in get_mAP!'

    # calculate IoUmax()
    x1 = np.concatenate((a[:,0].reshape(-1,1), b[:,0].reshape(-1,1)), axis=1).max(axis=1)
    y1 = np.concatenate((a[:,1].reshape(-1,1), b[:,1].reshape(-1,1)), axis=1).max(axis=1)
    x2 = np.concatenate((a[:,2].reshape(-1,1), b[:,2].reshape(-1,1)), axis=1).min(axis=1)
    y2 = np.concatenate((a[:,3].reshape(-1,1), b[:,3].reshape(-1,1)), axis=1).min(axis=1)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    interS = w * h
    area_a = (a[:,2] - a[:,0] + 1) * (a[:,3] - a[:,1] + 1)
    area_b = (b[:,2] - b[:,0] + 1) * (b[:,3] - b[:,1] + 1)

    IoU = interS / (area_a + area_b - interS)
    IoU[w<=0] = 0
    IoU[h<=0] = 0
    
    if qualitative is True:
        return IoU.reshape(1,-1)

    # calcuate mAP
    thre = np.linspace(0, 1, 201)
    success_rate = []
    eps = 1e-6
    for thre_i in thre:
        success_rate.append(len(IoU[IoU>=thre_i]) / (len(IoU)+eps))

    success_rate = np.array(success_rate)
    mAP = ((success_rate[1:]+success_rate[:-1]) * (thre[1:]-thre[:-1]) / 2).sum()

    return mAP
    
