"""ShallowConvNet (Schirrmeister et al. 2017), layer-for-layer equivalent to
braindecode 2.0's ShallowFBCSPNet with default hyperparameters and
final_conv_length="auto", on (B, 1, C, T) input, logits out.

Kept in-repo so the frozen pipeline does not depend on braindecode (which pulls
torchaudio/skorch).  tests/test_shallow.py checks numerical equality against
braindecode when it is installed.
"""
import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    def __init__(self, ch, t, nc, n_filters_time=40, filter_time_length=25, n_filters_spat=40,
                 pool_time_length=75, pool_time_stride=15, drop_prob=0.5, batch_norm_alpha=0.1):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=True)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat, (ch, 1), bias=False)
        self.bnorm = nn.BatchNorm2d(n_filters_spat, momentum=batch_norm_alpha, affine=True, eps=1e-5)
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop = nn.Dropout(drop_prob)
        n_out_time = (t - filter_time_length + 1 - pool_time_length) // pool_time_stride + 1
        self.final_layer = nn.Conv2d(n_filters_spat, nc, (1, n_out_time), bias=True)
        self.c = dict(ch=ch, t=t, nc=nc)

    def forward(self, x):                       # (B, 1, C, T)
        x = self.conv_time(x)                   # (B, 40, C, T-24)
        x = self.conv_spat(x)                   # (B, 40, 1, T-24)
        x = self.bnorm(x)
        x = x * x                               # Square
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # SafeLog
        x = self.drop(x)
        x = self.final_layer(x)                 # (B, nc, 1, 1)
        return x.flatten(1)

    @torch.no_grad()
    def load_from_braindecode(self, bd_model):
        """Copy weights from a braindecode ShallowFBCSPNet (for the equivalence test).
        braindecode's conv_time kernel is (25, 1) on (B, 1, T, C) input; ours is (1, 25) on (B, 1, C, T)."""
        sd = bd_model.state_dict()
        self.conv_time.weight.copy_(sd["conv_time_spat.conv_time.weight"].transpose(-1, -2))
        self.conv_time.bias.copy_(sd["conv_time_spat.conv_time.bias"])
        self.conv_spat.weight.copy_(sd["conv_time_spat.conv_spat.weight"].transpose(-1, -2))
        for k in ("weight", "bias", "running_mean", "running_var"):
            getattr(self.bnorm, k).copy_(sd[f"bnorm.{k}"])
        self.final_layer.weight.copy_(sd["final_layer.conv_classifier.weight"].transpose(-1, -2))
        self.final_layer.bias.copy_(sd["final_layer.conv_classifier.bias"])
