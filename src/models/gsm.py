# PyTorch GSM implementation by Swathikiran Sudhakaran
# Edited by artmerinov

import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class GSM(nn.Module):
    def __init__(self, fPlane: int, num_segments: int):
        super(GSM, self).__init__()

        self.fPlane = int(fPlane)
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

        # Take first fPlane channels
        x = x_full[:, :self.fPlane, :, :] # [n=16 * t=8, c=48, h=28, w=28]

        n_t, c, h, w = x.size() 
        t = self.num_segments
        n = n_t // t

        x = Rearrange("(n t) c h w -> n c t h w", n=n, t=t, c=c, h=h, w=w)(x) 
        # --> [n=16, c=48, t=8, h=28, w=28]

        x_group1 = x[:, :self.fPlane // 2 , :, :] # [n=16, c=24, t=8, h=28, w=28]
        x_group2 = x[:,  self.fPlane // 2:, :, :] # [n=16, c=24, t=8, h=28, w=28]

        gate = F.tanh(self.conv3D(F.relu(self.bn(x)))) # [n=16, c=2, t=8, h=28, w=28]
        gate_group1 = gate[:, 0, :, :, :].unsqueeze(1) # [n=16, c=1, t=8, h=28, w=28]
        gate_group2 = gate[:, 1, :, :, :].unsqueeze(1) # [n=16, c=1, t=8, h=28, w=28]

        y_group1 = gate_group1 * x_group1 # [n=16, c=24, t=8, h=28, w=28]
        y_group2 = gate_group2 * x_group2 # [n=16, c=24, t=8, h=28, w=28]
        # print(y_group1)

        r_group1 = x_group1 - y_group1 # [n=16, c=24, t=8, h=28, w=28]
        r_group2 = x_group2 - y_group2 # [n=16, c=24, t=8, h=28, w=28]

        # print('KEKASD')
        # print(self.lshift_zeroPad(y_group1))

        y_group1 = self.lshift_zeroPad(y_group1) + r_group1 # [n=16, c=24, t=8, h=28, w=28]
        y_group2 = self.rshift_zeroPad(y_group2) + r_group2 # [n=16, c=24, t=8, h=28, w=28]


        y_group1 = y_group1.view(n, 2, self.fPlane // 4, t, h, w).permute(0, 2, 1, 3, 4, 5) # [n=16, 12, 2, t=8, h=28, w=28]
        y_group2 = y_group2.view(n, 2, self.fPlane // 4, t, h, w).permute(0, 2, 1, 3, 4, 5) # [n=16, 12, 2, t=8, h=28, w=28]

        y = torch.cat((
            y_group1.contiguous().view(n, self.fPlane//2, t, h, w), # [n=16, c=24, t=8, h=28, w=28]
            y_group2.contiguous().view(n, self.fPlane//2, t, h, w)  # [n=16, c=24, t=8, h=28, w=28]
        ), dim=1)
        # -> [n=16, c=48, t=8, h=28, w=28]

        y = Rearrange("n c t h w -> (n t) c h w", n=n, t=t, c=c, h=h, w=w)(y) # [n=16 * t=8, c=48, h=28, w=28]
        
        return y
    

if __name__ == "__main__":
    
    x = torch.rand((16*8, 64, 28, 28)) # n*t, c, h, w
    
    # take first 48 channels out of 64
    model = GSM(fPlane=48, num_segments=8)
    y = model(x)
    print(y.size())
