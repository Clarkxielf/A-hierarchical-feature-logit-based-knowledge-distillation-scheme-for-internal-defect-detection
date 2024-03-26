import torch
import os
import  torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

'''教师网络和学生网络特征转换一致'''
class Embedding(nn.Module):
    def __init__(self, dim_in=512, dim_out=512):
        super(Embedding,self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)
        self.norm = nn.BatchNorm1d(dim_out)

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)

        return x

'''学生网络语义特征与教师网络进行对齐的损失'''
class FM_LOSS(nn.Module):
    def __init__(self, heads = 8):
        super(FM_LOSS, self).__init__()
        self.embed = Embedding()
        self.heads = heads

    def forward(self,f_s, f_t):
        f_t = self.embed(f_t)
        S, Q, V = f_s, f_s, f_s
        T, K = f_t, f_t
        b, c, n = V.shape

        Q = rearrange(Q, 'b (h d) n -> b h n d', h = self.heads)
        K = rearrange(K, 'b (h d) n -> b h n d', h = self.heads)
        V = rearrange(V, 'b (h d) n -> b h n d', h=self.heads)

        sim = einsum('b h i d, b h j d -> b h i j', Q, K)
        sim = sim.softmax(dim=-1)
        fs = einsum('b h i j, b h j d -> b h i d', sim, V)
        fs = rearrange(fs, 'b h n d -> b (h d) n')

        loss = nn.MSELoss()(fs, f_t)

        return loss


'''目标类与非目标类损失分别计算为tckd与nckd'''
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction= 'batchmean')
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction= 'batchmean')
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# fs = torch.randn(24, 256, 9)
# ft = torch.randn(24, 512, 9)
# loss = FM_LOSS().cuda()
# loss = loss(fs.to(device), ft.to(device))
# print(loss)