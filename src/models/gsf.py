# PyTorch GSF implementation by Swathikiran Sudhakaran
# Edited by artmerinov

import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class GSF(nn.Module):
    def __init__(self, 
                 fPlane: int,
                 num_segments: int, 
                 gsf_ch_ratio: int,
    ):
        super(GSF, self).__init__()

        self.fPlane = int(fPlane * gsf_ch_ratio / 100)
        if self.fPlane % 2 != 0:
            self.fPlane += 1
        
        self.num_segments = num_segments
        self.conv3D = nn.Conv3d(
            in_channels=self.fPlane, 
            out_channels=2, 
            kernel_size=(3, 3, 3), 
            stride=1,
            padding=(1, 1, 1), 
            groups=2
        )
        self.bn = nn.BatchNorm3d(self.fPlane)
        self.channel_conv1 = nn.Conv2d(
            in_channels=2, 
            out_channels=1, 
            kernel_size=(3, 3), 
            padding=(3//2, 3//2)
        )
        self.channel_conv2 = nn.Conv2d(
            in_channels=2, 
            out_channels=1, 
            kernel_size=(3, 3), 
            padding=(3//2, 3//2)
        )

    def lshift_zeroPad(self, x):
        out = torch.roll(x, shifts=-1, dims=2)
        out[:, :, -1, :, :] = 0
        return out

    def rshift_zeroPad(self, x):
        out = torch.roll(x, shifts=1, dims=2)
        out[:, :, 0, :, :] = 0
        return out

    def forward(self, x_full):
        # For _block_3a 
        # x_full: [n=16 * t=8, c=64, h=28, w=28]

        # Take first 0.75 fraction of channels
        x = x_full[:, :self.fPlane, :, :] # [n=16 * t=8, c=48, h=28, w=28]

        n_t, c, h, w = x.size() 
        t = self.num_segments
        n = n_t // t

        x = Rearrange("(n t) c h w -> n c t h w", n=n, t=t, c=c, h=h, w=w)(x) 
        # --> [n=16, c=48, t=8, h=28, w=28]
        
        gate = F.tanh(self.conv3D(F.relu(self.bn(x)))) # [n=16, c=2, t=8, h=28, w=28]

        x_group1 = x[:, :self.fPlane // 2 , :, :] # [n=16, c=24, t=8, h=28, w=28]
        x_group2 = x[:,  self.fPlane // 2:, :, :] # [n=16, c=24, t=8, h=28, w=28]
        
        gate_group1 = gate[:, 0, :, :, :].unsqueeze(1) # [n=16, c=1, t=8, h=28, w=28]
        gate_group2 = gate[:, 1, :, :, :].unsqueeze(1) # [n=16, c=1, t=8, h=28, w=28]

        y_group1 = gate_group1 * x_group1 # [n=16, c=24, t=8, h=28, w=28]
        y_group2 = gate_group2 * x_group2 # [n=16, c=24, t=8, h=28, w=28]

        r_group1 = x_group1 - y_group1 # [n=16, c=24, t=8, h=28, w=28]
        r_group2 = x_group2 - y_group2 # [n=16, c=24, t=8, h=28, w=28]

        y_group1 = self.lshift_zeroPad(y_group1) # [n=16, c=24, t=8, h=28, w=28]
        y_group2 = self.rshift_zeroPad(y_group2) # [n=16, c=24, t=8, h=28, w=28]

        # r_1 = torch.mean(r_group1, dim=-1, keepdim=False)
        # r_1 = torch.mean(r_1, dim=-1, keepdim=False).unsqueeze(3)
        # r_2 = torch.mean(r_group2, dim=-1, keepdim=False)
        # r_2 = torch.mean(r_2, dim=-1, keepdim=False).unsqueeze(3)
        r_1 = torch.mean(r_group1, dim=(3,4), keepdim=False).unsqueeze(3) # [n=16, c=24, t=8, 1]
        r_2 = torch.mean(r_group2, dim=(3,4), keepdim=False).unsqueeze(3) # [n=16, c=24, t=8, 1]

        # y_1 = torch.mean(y_group1, dim=-1, keepdim=False)
        # y_1 = torch.mean(y_1, dim=-1, keepdim=False).unsqueeze(3)
        # y_2 = torch.mean(y_group2, dim=-1, keepdim=False)
        # y_2 = torch.mean(y_2, dim=-1, keepdim=False).unsqueeze(3) # BxCxN
        y_1 = torch.mean(y_group1, dim=(3,4), keepdim=False).unsqueeze(3) # [n=16, c=24, t=8, 1]
        y_2 = torch.mean(y_group2, dim=(3,4), keepdim=False).unsqueeze(3) # [n=16, c=24, t=8, 1]

        y_r_1 = torch.cat([y_1, r_1], dim=3).permute(0, 3, 1, 2) # [n=16, 2, c=24, t=8]
        y_r_2 = torch.cat([y_2, r_2], dim=3).permute(0, 3, 1, 2) # [n=16, 2, c=24, t=8]

        y_1_weights = F.sigmoid(self.channel_conv1(y_r_1)).squeeze(1).unsqueeze(-1).unsqueeze(-1) 
        # [n=16, 2, c=24, t=8] -> [n=16, 1, c=24, t=8, 1, 1] -> [n=16, c=24, t=8, 1, 1]
        y_2_weights = F.sigmoid(self.channel_conv2(y_r_2)).squeeze(1).unsqueeze(-1).unsqueeze(-1) 
        # [n=16, 2, c=24, t=8] -> [n=16, 1, c=24, t=8, 1, 1] -> [n=16, c=24, t=8, 1, 1]

        y_group1 = y_group1 * y_1_weights + r_group1 * (1 - y_1_weights) # [n=16, c=24, t=8, h=28, w=28]
        y_group2 = y_group2 * y_2_weights + r_group2 * (1 - y_2_weights) # [n=16, c=24, t=8, h=28, w=28]

        y_group1 = y_group1.view(n, 2, self.fPlane//4, t, h, w).permute(0, 2, 1, 3, 4, 5) 
        # [n=16, c=48//4, 2, t=8, h=28, w=28]
        y_group2 = y_group2.view(n, 2, self.fPlane//4, t, h, w).permute(0, 2, 1, 3, 4, 5) 
        # [n=16, c=48//4, 2, t=8, h=28, w=28]
        
        y = torch.cat((
            y_group1.contiguous().view(n, self.fPlane//2, t, h, w),
            y_group2.contiguous().view(n, self.fPlane//2, t, h, w)
        ), dim=1) # [16, 48, 8, 28, 28]
        
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(n*t, c, h, w) # [128, 48, 28, 28]
        y = torch.cat([y, x_full[:, self.fPlane:, :, :]], dim=1) # [128, 64, 28, 28]

        return y