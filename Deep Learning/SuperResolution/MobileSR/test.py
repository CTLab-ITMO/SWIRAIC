import os;
import torch;
import torchvision;
import numpy as np;
from PIL import Image;
from torch import nn;
import torch.nn.functional as F;
from torchvision.utils import save_image, make_grid;
from einops import rearrange;
from torch.nn.parameter import Parameter;
import torchvision.transforms.functional as FT;
from torch.nn.init import trunc_normal_;

class WindowSelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8) -> None:
        super().__init__();
        self.num_heads = num_heads;
        self.window_size = window_size;
        head_dim = dim // num_heads;
        self.scale = head_dim ** -0.5;
        self.query = nn.Conv1d(dim, dim, kernel_size = 1, padding = 0, bias = False);
        self.key = nn.Conv1d(dim, dim, kernel_size = 1, padding = 0, bias = False);
        self.value = nn.Conv1d(dim, dim, kernel_size = 1, padding = 0, bias = False);
        self.beta = nn.Parameter(torch.zeros(num_heads, window_size ** 2, window_size ** 2));
        self.proj_out = nn.Conv1d(dim, dim, kernel_size = 1, padding = 0, bias = True);
        trunc_normal_(self.beta, std = 0.02);

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape;
        # Pad
        pad_r = (self.window_size - w % self.window_size) % self.window_size;
        pad_b = (self.window_size - h % self.window_size) % self.window_size;
        x = F.pad(x, (0, pad_r, 0, pad_b));
        # Window partition
        x = rearrange(x, 'b c (h s1) (w s2) -> (b h w) c (s1 s2)', s1 = self.window_size, s2 = self.window_size);
        # Project
        q = self.query(x);
        k = self.key(x);
        v = self.value(x);
        # Attention
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = self.num_heads), [q, k, v]);
        attn = torch.einsum('b h d n, b h d m -> b h n m', q, k) * self.scale + self.beta;
        attn = attn.softmax(dim = -1);
        x = torch.einsum('b h n m, b h d m -> b h d n', attn, v);
        x = rearrange(x, 'b i c j -> b (i c) j');
        x = self.proj_out(x);
        # Reverse window partition
        x = rearrange(x, 'B c (s1 s2) -> B c s1 s2', s1 = self.window_size, s2 = self.window_size);
        b = int(x.shape[0] / (h * w / self.window_size / self.window_size));
        x = rearrange(x, '(b h w) c s1 s2 -> b c (h s1) (w s2)', b = b, h = (h + pad_b) // self.window_size, w = (w + pad_r) // self.window_size);
        if((pad_r > 0) or (pad_b > 0)):
            x = x[:, :, :h, :w].contiguous();
        return x;

class PCFN(nn.Module):

    def __init__(self, dim: int, growth_rate: float = 2.0, p_rate: float = 0.25) -> None:
        super().__init__();
        hidden_dim = int(dim * growth_rate);
        p_dim = int(hidden_dim * p_rate);
        self.conv_0 = nn.Conv2d(dim, hidden_dim, kernel_size = 1, padding = 0);
        self.conv_1 = nn.Conv2d(p_dim, p_dim, kernel_size = 3, padding = 1, padding_mode = "reflect");
        self.act = nn.GELU();
        self.conv_2 = nn.Conv2d(hidden_dim, dim, kernel_size = 1, padding = 0);
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

class Transformer(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int = 4,
                 window_size: int = 8,
                 mlp_ratio: int = 4) -> None:
        super().__init__();
        self.attn = WindowSelfAttention(dim, num_heads, window_size);
        self.pcfn = PCFN(dim, mlp_ratio);

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(F.normalize(x)) + x;
        x = self.pcfn(F.normalize(x)) + x;
        return x;

class SRViT(nn.Module):

    def __init__(self, n_feats: int = 40, n_heads: int = 8, ratio: int = 2, blocks = 5, upscaling_factor: int = 4) -> None:
        super(SRViT, self).__init__();
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1);
        self.body = nn.Sequential(*[Transformer(n_feats, n_heads, 8, ratio) for i in range(blocks)]);
        self.upsampling = nn.Sequential(
            nn.Conv2d(n_feats, 3 * upscaling_factor ** 2, kernel_size = 3, padding = 1, padding_mode = "reflect"),
            nn.PixelShuffle(upscaling_factor)
        );
        
    def forward(self, x):
        res = F.interpolate(x, size = (x.size(2) * 4, x.size(3) * 4));
        x = self.head(x);
        x = self.upsampling(self.body(x) + x);
        return x + res;

path = "Minerals";
fname = "min_test20";

img_64 = torch.cat([FT.to_tensor(Image.open(os.path.join(path, "800", f"{fname}_800.jpg")).convert('L')),
                    FT.to_tensor(Image.open(os.path.join(path, "1050", f"{fname}_1050.jpg")).convert('L')),
                    FT.to_tensor(Image.open(os.path.join(path, "1550", f"{fname}_1050.jpg")).convert('L'))], dim = 0).unsqueeze(0);

net = SRViT().eval();
net.load_state_dict(torch.load("srvit_8000.pth", map_location = torch.device("cpu")));

with(torch.no_grad()):
    pred = net(img_64);
    save_image(make_grid(torch.cat([F.interpolate(img_64, size = (248, 248), mode = "bicubic").view(3, 1, 248, 248),
                                    pred.view(3, 1, 248, 248)], dim = 0), nrow = 3), f"{fname}.png");
