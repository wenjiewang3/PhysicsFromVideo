import torch
import torch.nn as nn
from typing import Sequence, Literal

class FrameEncoderCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 3,
        hidden_mlp: Sequence[int] = (128,),
        act: Literal["relu","gelu","silu","leakyrelu","tanh"] = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        acts = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(0.1, inplace=True),
            "tanh": nn.Tanh(),
        }
        A = acts.get(act, nn.GELU())


        chans = [in_channels] + [base_channels * (2**i) for i in range(num_blocks)]
        blocks = []
        for i in range(num_blocks):
            c_in, c_out = chans[i], chans[i+1]
            stride = 2 if i < num_blocks-1 else 1
            blocks += [
                nn.Conv2d(c_in,  c_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                A,
                nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                A,
            ]
        self.backbone = nn.Sequential(*blocks)

        # GAP -> (B*, C, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # MLP to scalar
        mlp_layers = []
        feat_dim = chans[-1]
        for h in hidden_mlp:
            mlp_layers += [nn.Linear(feat_dim, h), A]
            if dropout > 0: mlp_layers += [nn.Dropout(dropout)]
            feat_dim = h
        mlp_layers += [nn.Linear(feat_dim, 1)]
        self.head = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in (4,5), "x must be (B,T,H,W) or (B,T,1,H,W)"
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            assert C == 1, "Only single-channel grayscale supported"
            x = x.squeeze(2)
        else:
            B, T, H, W = x.shape

        # Merge batch and time for frame-wise processing
        x = x.reshape(B*T, 1, H, W)
        feat = self.backbone(x)                  # (B*T, C, h, w)
        feat = self.gap(feat).flatten(1)         # (B*T, C)
        z = self.head(feat).reshape(B, T, 1)     # (B, T, 1)
        return z


class FrameEncoderMLP(nn.Module):
    def __init__(self, H: int, W: int,
                 hidden_dims: Sequence[int]=(1024,256),
                 act: Literal["relu","gelu","silu","leakyrelu","tanh"]="gelu",
                 dropout: float=0.0, layernorm: bool=False):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(),
        }
        A = acts.get(act, nn.GELU())
        in_dim = H*W
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if layernorm: layers.append(nn.LayerNorm(h))
            layers.append(A)
            if dropout>0: layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.H, self.W = H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in (4,5)
        if x.ndim == 5:
            B,T,C,H,W = x.shape
            assert C==1
            x = x.squeeze(2)
        else:
            B,T,H,W = x.shape
        x = x.reshape(B*T, H*W)
        z = self.net(x).reshape(B,T,1)
        return z


class FrameEncoderIntensity(nn.Module):
    def __init__(self,
                 topk_ratio: float = 0.1,
                 learn_affine: bool = True):
        super().__init__()
        assert 0.0 < topk_ratio <= 1.0
        self.topk_ratio = topk_ratio

        if learn_affine:
            self.log_a = nn.Parameter(torch.zeros(1))  # a = softplus(log_a)
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("log_a", None)
            self.register_parameter("b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in (4, 5), "x must be (B,T,H,W) or (B,T,1,H,W)"
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            assert C == 1
            x = x.squeeze(2)
        else:
            B, T, H, W = x.shape

        # (B,T,H,W) -> (B*T, H*W)
        x_flat = x.reshape(B * T, -1)

        # Average of top-k brightest pixels per frame
        k = max(1, int(self.topk_ratio * x_flat.shape[1]))
        vals, _ = torch.topk(x_flat, k, dim=1)
        m = vals.mean(dim=1, keepdim=True)  # (B*T,1)

        # Optional affine transform
        if self.log_a is not None:
            a = torch.nn.functional.softplus(self.log_a)  # >=0
            z = a * m + self.b
        else:
            z = m

        z = z.view(B, T, 1)
        return z

