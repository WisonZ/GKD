import torch.nn as nn
import torch.nn.functional as F

class HintLoss(nn.Module):
    """ Fitnets: hints for thin deep nets, ICLR 2015
    """
    def __init__(self, ):
        super(HintLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.T = 2
        self.ce_loss_weight = 1.0

    def forward(self, feat_s, feat_t,logits_s,target):
        loss_kd = self.mse(F.normalize(feat_s), F.normalize(feat_t)) / (self.T * feat_s.shape[0])
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_s, target)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return losses_dict