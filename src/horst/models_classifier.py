import torch
from torch import nn


class ActionHead(nn.Module):
    def __init__(self, in_dim, num_a):
        super().__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.unrolling = nn.LSTM(in_dim, in_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim)
        )
        
        self.fuse_ev = nn.Sequential(
            nn.Linear(2*in_dim, 2*in_dim),
        )
    
    def forward(self, x, index, n, v, total_length):
        x = self.avgpool(x).contiguous()
        x = x[:, None, :].contiguous().repeat(1, total_length-index, 1).contiguous()
        
        h0, c0 = self.fuse_ev(torch.cat([n, v], dim=-1)).chunk(2, dim=-1)
        h0 = h0[None, ...].contiguous()
        c0 = c0[None, ...].contiguous()
        
        x = self.unrolling(x, (h0, c0))[0][:, -1, :]
        return self.classifier(x)
        

class Classifier(nn.Module):
    def __init__(self, in_dim, num_a, num_n, num_v):
        super().__init__()
        self.unroll = ActionHead(in_dim, num_a)

        self.only_a = num_n is None or num_v is None
        num_n = num_n or 100
        num_v = num_v or 50
        
        self.encoder_n = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim)
        )
        
        self.encoder_v = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim)
        )
        
        self.c_a = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_a)
        )
        
        self.c_n = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_n)
        )
        
        self.c_v = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_v)
        )
        
        self.c_a2n = nn.Sequential(
            nn.Linear(num_a, num_n),
        )
        self.c_a2v = nn.Sequential(
            nn.Linear(num_a, num_v)
        )
    
    def forward(self, x):
        out = []
        for i in range(x.size(1)):
            a, n, v = x[:, i].split(1, dim=-1)
            e_n = self.encoder_n(n.squeeze(-1))
            e_v = self.encoder_v(v.squeeze(-1))
            unrolled = self.unroll(a.squeeze(-1), i, e_n, e_v, total_length=x.size(1))
            
            c_a = self.c_a(unrolled)
            c_n = self.c_n(unrolled + e_n) + self.c_a2n(c_a)
            c_v = self.c_v(unrolled + e_v) + self.c_a2v(c_a)
            
            if self.only_a:
                out.append(c_a)
            else:
                out.append((c_a, c_n, c_v))
                
        if self.only_a:
            out = torch.stack(out, 1)
        return out