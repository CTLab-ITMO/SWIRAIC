import os;
import torch;
import torchvision;
import numpy as np;
from PIL import Image;
from torch import nn;
import torch.nn.functional as F;
from torchvision.utils import save_image, make_grid;
from torch.nn.parameter import Parameter;
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
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0);
        self.lde = DMlp(dim, 2);
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups = dim);
        self.gelu = nn.GELU();
        self.down_scale = 8;
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)));
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)));

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        _, _, h, w = f.shape;
        y, x = self.linear_0(f).chunk(2, dim = 1);
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)));
        x_v = torch.var(x, dim = (-2, -1), keepdim = True);
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.beta)), size = (h, w), mode = 'nearest');
        y_d = self.lde(y);
        return self.linear_2(x_l + y_d);

class FMB(nn.Module):

    def __init__(self, dim: int, ffn_scale: float = 2.0) -> None:
        super().__init__();
        self.smfa = SMFA(dim);
        self.pcfn = PCFN(dim, ffn_scale);

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.smfa(F.normalize(x)) + x;
        x = self.pcfn(F.normalize(x)) + x;
        return x;
 
class SMFANet(nn.Module):

    def __init__(self, dim: int = 48, n_blocks: int = 8, ffn_scale: float = 2.0, upscaling_factor: int = 4) -> None:
        super().__init__();
        self.scale = upscaling_factor;
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1);
        self.feats = nn.Sequential(*[FMB(dim, ffn_scale) for _ in range(n_blocks)]);
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        );

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.interpolate(x, size = (x.size(2) * 4, x.size(3) * 4), mode = "bilinear");
        x = self.to_feat(x);
        x = self.feats(x) + x;
        x = self.to_img(x);
        return x + res;

idx = 2;

img_64 = torch.cat([FT.to_tensor(Image.open(os.path.join("Minerals_old", "64", "800", f"{(idx + 1):03}_800.jpg")).convert('L')),
                    FT.to_tensor(Image.open(os.path.join("Minerals_old", "64", "1050", f"{(idx + 1):03}_1050.jpg")).convert('L')),
                    FT.to_tensor(Image.open(os.path.join("Minerals_old", "64", "1550", f"{(idx + 1):03}_1550.jpg")).convert('L'))], dim = 0).unsqueeze(0);

img_256 = torch.cat([FT.to_tensor(Image.open(os.path.join("Minerals_old", "256", "800", f"{(idx + 1):03}_800.jpg")).convert('L')),
                     FT.to_tensor(Image.open(os.path.join("Minerals_old", "256", "1050", f"{(idx + 1):03}_1050.jpg")).convert('L')),
                     FT.to_tensor(Image.open(os.path.join("Minerals_old", "256", "1550", f"{(idx + 1):03}_1550.jpg")).convert('L'))], dim = 0).unsqueeze(0);

net = SMFANet().eval();
net.load_state_dict(torch.load("smfanet_4000.pth", map_location = torch.device("cpu")));

with(torch.no_grad()):
    pred = net(img_64);
    save_image(make_grid(torch.cat([F.interpolate(img_64, size = (256, 256), mode = "bicubic").view(3, 1, 256, 256),
                                    pred.view(3, 1, 256, 256),
                                    img_256.view(3, 1, 256, 256)], dim = 0), nrow = 3), "result.png");
