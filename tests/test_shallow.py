"""ShallowConvNet in train/shallow.py must equal braindecode's ShallowFBCSPNet numerically."""
import pytest
import torch
from train.shallow import ShallowConvNet

bd = pytest.importorskip("braindecode.models")


@pytest.mark.parametrize("ch,t,nc", [(64, 640, 2), (22, 1000, 4), (62, 1000, 2)])
def test_equals_braindecode(ch, t, nc):
    torch.manual_seed(0)
    ref = bd.ShallowFBCSPNet(n_chans=ch, n_outputs=nc, n_times=t, final_conv_length="auto")
    ours = ShallowConvNet(ch, t, nc)
    # randomise BN stats so eval mode is a real test
    with torch.no_grad():
        ref.bnorm.running_mean.normal_(); ref.bnorm.running_var.uniform_(0.5, 2)
        ref.bnorm.weight.normal_(); ref.bnorm.bias.normal_()
    ours.load_from_braindecode(ref)
    ref.eval(); ours.eval()
    x = torch.randn(5, 1, ch, t)
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in ref.parameters())
    torch.testing.assert_close(ours(x), ref(x.squeeze(1)), rtol=1e-4, atol=1e-4)
