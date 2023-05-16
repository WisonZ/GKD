import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def gkd_loss(logits_student, logits_teacher, target, alpha, beta, lamda, temperature, rate,frequency=100):

    class_select = logits_teacher.scatter(1, target.unsqueeze(1), 999999)
    ##
    prob_teacher = torch.mean(F.softmax(logits_teacher, dim=1),dim=0)
    sorted_prob_teacher = prob_teacher.sort(descending=True)[0]
    sum=0
    for idx, logit in enumerate(sorted_prob_teacher):
        sum+=logit
        if sum>=rate:
            hcm_n = idx+target.shape[1]//frequency
            break
    ##

    class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :hcm_n]
    hard_mask = torch.zeros_like(logits_teacher).scatter(1, class_select_include_target, 1)
    nhard_mask = 1 - hard_mask

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    pred_student = cat_mask(pred_student, hard_mask, nhard_mask)
    pred_teacher = cat_mask(pred_teacher, hard_mask, nhard_mask)
    log_pred_student = torch.log(pred_student)
    binary_ckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )

    pred_teacher_part_hard = F.softmax(
        logits_teacher / temperature - 1000.0 * nhard_mask, dim=1
    )

    log_pred_student_hard = F.log_softmax(
        logits_student / temperature - 1000.0 * nhard_mask, dim=1
    )
    hard_kd_loss = (
        F.kl_div(log_pred_student_hard, pred_teacher_part_hard, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )

    pred_teacher_part_nhard = F.softmax(
        logits_teacher / temperature - 1000.0 * hard_mask, dim=1
    )

    log_pred_student_nhard = F.log_softmax(
        logits_student / temperature - 1000.0 * hard_mask, dim=1
    )
    nhard_kd_loss = (
            F.kl_div(log_pred_student_nhard, pred_teacher_part_nhard, reduction='sum')
            * (temperature ** 2)
            / target.shape[0]
    )

    return alpha * binary_ckd_loss + beta * hard_kd_loss + lamda * nhard_kd_loss

def gkd_loss_student_base(logits_student, logits_teacher, target, alpha, beta, lamda, temperature, rate):

    class_select = logits_student.scatter(1, target.unsqueeze(1), 999999)
    ##
    prob_student = torch.sum(F.softmax(logits_student, dim=1), dim=0)
    sorted_prob_student = prob_student.sort(descending=True)[0]
    logit_length = sorted_prob_student.shape[0]
    batch_size = logits_student.shape[0]
    for idx in range(1,logit_length,100):
        if torch.sum(sorted_prob_student[:idx],dim=0) >= rate*batch_size:
            hcm_n = idx
            break
    ##
    class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :hcm_n]

    hard_mask = torch.zeros_like(logits_student).scatter(1, class_select_include_target, 1)
    nhard_mask = 1 - hard_mask

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    pred_student = cat_mask(pred_student, hard_mask, nhard_mask)
    pred_teacher = cat_mask(pred_teacher, hard_mask, nhard_mask)
    log_pred_student = torch.log(pred_student)
    binary_ckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )

    pred_teacher_part_hard = F.softmax(
        logits_teacher / temperature - 1000.0 * nhard_mask, dim=1
    )

    log_pred_student_hard = F.log_softmax(
        logits_student / temperature - 1000.0 * nhard_mask, dim=1
    )
    hard_kd_loss = (
        F.kl_div(log_pred_student_hard, pred_teacher_part_hard, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )

    pred_teacher_part_nhard = F.softmax(
        logits_teacher / temperature - 1000.0 * hard_mask, dim=1
    )

    log_pred_student_nhard = F.log_softmax(
        logits_student / temperature - 1000.0 * hard_mask, dim=1
    )
    nhard_kd_loss = (
            F.kl_div(log_pred_student_nhard, pred_teacher_part_nhard, reduction='sum')
            * (temperature ** 2)
            / target.shape[0]
    )

    return alpha * binary_ckd_loss + beta * hard_kd_loss + lamda * nhard_kd_loss




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


class GKD(nn.Module):
    """Grouped Knowledge Distillation"""

    def __init__(self):
        super().__init__()
        self.ce_loss_weight = 1.0
        self.alpha = 1.0
        self.beta = 4.0
        self.lamda = 0.0
        self.temperature = 4.0
        self.warmup = 10
        self.rate = 0.93
        #self.fn_weight = 4.0
        #self.crit = nn.MSELoss()

    def forward(self, logits_student, logits_teacher, target, epoch_idx):#feat_student,feat_teacher,
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(epoch_idx / self.warmup, 1.0) * gkd_loss_student_base( #
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.lamda,
            self.temperature,
            self.rate
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
