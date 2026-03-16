import torch


class MaskedMSE(torch.nn.Module):
    def __init__(self, norm_pix_loss=False, masked=True):
        """
        norm_pix_loss: normalize each patch by their pixel mean and variance
        masked: compute loss over the masked patches only
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked

    def forward(self, pred, mask, target):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        if self.masked:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss
