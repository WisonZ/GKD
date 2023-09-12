import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """ Distilling the Knowledge in a Neural Network, NIPSW 2015
    """
    def __init__(self, T=4):
        super().__init__()
        self.T = T

    def forward(self, y_s, y_t,target):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        loss_ce = 1.0 * F.cross_entropy(y_s, target)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss,
            # "loss_fn":loss_fn,
        }
        return y_s, losses_dict