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

class UNet(nn.Module):

    def __init__(self) -> None:
        super().__init__();
        self.enc0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 5, padding = 2, stride = 2),
            nn.PReLU(64)
        );
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 5, padding = 2, stride = 2),
            nn.PReLU(128)
        );
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 5, padding = 2, stride = 2),
            nn.PReLU(256)
        );
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 5, padding = 2, stride = 2),
            nn.PReLU(512)
        );
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size = 5, padding = 2, output_padding = 1, stride = 2),
            nn.PReLU(256)
        );
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size = 5, padding = 2, output_padding = 1, stride = 2),
            nn.PReLU(128)
        );
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size = 5, padding = 2, output_padding = 1, stride = 2),
            nn.PReLU(64)
        );
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size = 5, padding = 2, output_padding = 1, stride = 2)
        );

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(x);
        x1 = self.enc1(x0);
        x2 = self.enc2(x1);
        x3 = self.enc3(x2);
        y = self.dec3(x3);
        y = self.dec2(torch.cat([F.interpolate(y, size = x2.shape[2:], mode = "nearest"), x2], dim = 1));
        y = self.dec1(torch.cat([F.interpolate(y, size = x1.shape[2:], mode = "nearest"), x1], dim = 1));
        y = self.dec0(torch.cat([F.interpolate(y, size = x0.shape[2:], mode = "nearest"), x0], dim = 1));
        return y;

net = UNet().eval();
net.load_state_dict(torch.load("unet_2500.pth", map_location = torch.device("cpu")));

for idx in range(334):
    img_256 = torch.cat([FT.to_tensor(Image.open(os.path.join("Minerals - копия", "256", "800", f"{(idx + 1):03}_800.jpg")).convert('L')),
                         FT.to_tensor(Image.open(os.path.join("Minerals - копия", "256", "1050", f"{(idx + 1):03}_1050.jpg")).convert('L')),
                         FT.to_tensor(Image.open(os.path.join("Minerals - копия", "256", "1550", f"{(idx + 1):03}_1550.jpg")).convert('L'))], dim = 0).unsqueeze(0);

    with(torch.no_grad()):
        pred = net(img_256);
        save_image(pred[:, 0, :, :], os.path.join("Minerals - копия", "256", "800", f"{(idx + 1):03}_800.jpg"))
        save_image(pred[:, 1, :, :], os.path.join("Minerals - копия", "256", "1050", f"{(idx + 1):03}_1050.jpg"))
        save_image(pred[:, 2, :, :], os.path.join("Minerals - копия", "256", "1550", f"{(idx + 1):03}_1550.jpg"))
