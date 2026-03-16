import torch.nn as nn
import torch.nn.functional as F

from longstream.utils.vendor.dust3r.heads.postprocess import postprocess


class LinearPts3d(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, has_conf=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf

        self.proj = nn.Linear(net.dec_embed_dim, (3 + has_conf) * self.patch_size ** 2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        feat = self.proj(tokens)
        feat = feat.transpose(-1, -2).view(
            B, -1, H // self.patch_size, W // self.patch_size
        )
        feat = F.pixel_shuffle(feat, self.patch_size)

        return postprocess(feat, self.depth_mode, self.conf_mode)
