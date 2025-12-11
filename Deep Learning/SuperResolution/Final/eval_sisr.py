import os;
import torch;
import torchvision;
import numpy as np;
from PIL import Image;
from torch import nn;
import torch.nn.functional as F;
from torchvision.utils import save_image, make_grid;
import torchvision.transforms.functional as FT;

class DMlp(nn.Module):

    def __init__(self, dim: int, growth_rate: float = 2.0) -> None:
        super().__init__();
        hidden_dim = int(dim * growth_rate);
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups = dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        );
        self.act = nn.GELU();
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0);

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_0(x);
        x = self.act(x);
        x = self.conv_1(x);
        return x;

class PCFN(nn.Module):

    def __init__(self, dim: int, growth_rate: float = 2.0, p_rate: float = 0.25) -> None:
        super().__init__();
        hidden_dim = int(dim * growth_rate);
        p_dim = int(hidden_dim * p_rate);
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0);
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1);
        self.act = nn.GELU();
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0);
        self.p_dim = p_dim;
        self.hidden_dim = hidden_dim;

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if(self.training):
            x = self.act(self.conv_0(x));
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim = 1);
            x1 = self.act(self.conv_1(x1));
            x = self.conv_2(torch.cat([x1, x2], dim = 1));
        else:
            x = self.act(self.conv_0(x));
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]));
            x = self.conv_2(x);
        return x;

class SMFA(nn.Module):

    def __init__(self, dim: int = 36) -> None:
        super().__init__();
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0);
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0);
        self.linear_2 = nn.Conv2d(dim * 2, dim, 1, 1, 0);
        self.lde = DMlp(dim, 2);
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size = 3, dilation = 9, padding = 9, groups = dim);
        self.gelu = nn.GELU();
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)));
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)));

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        _, _, h, w = f.shape;
        y, x = self.linear_0(f).chunk(2, dim = 1);
        x_s = self.dw_conv(F.max_pool2d(x, kernel_size = 9, padding = 4, stride = 1));
        x_v = torch.var(x, dim = (-2, -1), keepdim = True);
        x_l = x * self.gelu(self.linear_1(x_s * self.alpha + x_v * self.beta));
        y_d = self.lde(y);
        return self.linear_2(torch.cat([x_l, y_d], dim = 1));

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__();
        self.weight = nn.Parameter(torch.ones(normalized_shape));
        self.bias = nn.Parameter(torch.zeros(normalized_shape));
        self.eps = eps;

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim = True);
        s = (x - u).pow(2).mean(1, keepdim = True);
        x = (x - u) / torch.sqrt(s + self.eps);
        x = self.weight[:, None, None] * x + self.bias[:, None, None];
        return x;

class FMB(nn.Module):

    def __init__(self, dim: int, ffn_scale: float = 2.0) -> None:
        super().__init__();
        self.norm1 = LayerNorm(dim);
        self.smfa = SMFA(dim);
        self.norm2 = LayerNorm(dim);
        self.pcfn = PCFN(dim, ffn_scale);

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.smfa(self.norm1(x)) + x;
        x = self.pcfn(self.norm2(x)) + x;
        return x;

class SMFANet(nn.Module):

    def __init__(self, dim: int = 36, n_blocks: int = 8, ffn_scale: float = 2.0, upscaling_factor: int = 4) -> None:
        super().__init__();
        self.upscaling_factor = upscaling_factor;
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1);
        self.feats = nn.Sequential(*[FMB(dim, ffn_scale) for _ in range(n_blocks)]);
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        );

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(x, scale_factor = self.upscaling_factor, mode = "bicubic");
        x = self.to_feat(x);
        x = self.feats(x) + x;
        x = self.to_img(x);
        return x + residual;

path = "exp_f";
fname = "9";

img_64 = torch.cat([FT.to_tensor(Image.open(os.path.join(path, f"{fname}_wl_941_.png")).convert('L')),
                    FT.to_tensor(Image.open(os.path.join(path, f"{fname}_wl_1065_.png")).convert('L')),
                    FT.to_tensor(Image.open(os.path.join(path, f"{fname}_wl_1550_.png")).convert('L'))], dim = 0).unsqueeze(0);

net = SMFANet().eval();
net.load_state_dict(torch.load("smfagan.pth", map_location = torch.device("cpu")));

with(torch.no_grad()):
    pred = net(img_64);
    save_image(make_grid(torch.cat([F.interpolate(img_64, size = (512, 512), mode = "bicubic").view(3, 1, 512, 512),
                                    pred.view(3, 1, 512, 512)], dim = 0), nrow = 3), f"{fname}.png");
