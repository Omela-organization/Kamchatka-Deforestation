from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader, dataloader
from IPython.display import clear_output
from einops import rearrange
from einops import rearrange, repeat
from time import time
import math
import numpy as np


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult, act=F.mish):
        super().__init__()
        hidden_dim = dim * ff_mult

        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        self.act = act

    def forward(self, x):
        x = self.w1(x)

        x, gate = x.chunk(2, dim=-1)

        x = self.act(x) * gate
        
        x = self.w2(x)

        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, in_chans, dim):
        super().__init__()
        self.patch_size = patch_size

        patch_dim = in_chans * patch_size **2

        self.norm = RMSNorm(patch_dim)

        self.proj = nn.Linear(patch_dim, dim)

        self.norm_out = RMSNorm(dim)

    def forward(self, x):

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
 
        x = self.norm(x)

        x = self.proj(x)

        x = self.norm_out(x)
        
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head

        self.qkv= nn.Linear(dim, dim * 3, bias=False)
   
        self.out = nn.Linear(dim, dim, bias=False)

        self.scale = 1. / math.sqrt(dim//n_head)

    def forward(self, x):

        q, k, v= self.qkv(x).chunk(3, dim=-1)


        wv, qk = self.qkv_attention(q, k, v)
        return self.out(wv)

    def qkv_attention(self, q, k, v):
        n_batch, n_ctx, n_state = q.shape
        scale = self.scale

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
    


class Block(nn.Module):

    def __init__(self, dim, n_head, ff_mult, act, skip):
        super().__init__()

        self.attn_norm = RMSNorm(dim)

        self.ff1_norm= RMSNorm(dim)

        self.attn = Attention(dim, n_head)

        self.ff1 =FeedForward(dim, ff_mult, act)

        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None


    def forward(self, x, skip=None):

        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        res = x

        x = self.attn_norm(x)

        x = res + self.attn(x)
    
        res = x

        x = self.ff1_norm(x)
        
        x = res + self.ff1(x)

        return x
    

class ResNetBlock(nn.Module):
 

    def __init__(self,dim, act):
        super( ).__init__()
        self.act = act


        self.conv_in = nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.norm1 = nn.GroupNorm(8, dim)
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False, padding_mode='reflect')

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, dilation=3, groups=dim, bias=False, padding_mode='reflect')

        self.norm2 = nn.GroupNorm(8, dim)

        self.norm3 = nn.GroupNorm(8, dim)

        self.norm4 = nn.GroupNorm(8, dim)

        self.norm5 = nn.GroupNorm(8, dim)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.norm1(x)
        x = self.act(x)

        res = x

        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)

        x = res + x
        x = self.norm3(x)
        x = self.act(x)

        res = x

        x = self.conv2(x)
        x = self.norm4(x)
        x = self.act(x)

        x = res + x
        x = self.norm5(x)
        x = self.act(x)

        

        return x
    


class OutBlock(nn.Module):

    def __init__(self, dim, hidden_dim, kernel_size, padding,num_classes, patch_size, act):
        super().__init__()
        self.patch_size = patch_size


        self.proj = nn.Linear(dim, hidden_dim)

        self.drop = nn.Dropout(0.1)

        self.conv_in = ResNetBlock(128, act)

        self.conv = nn.Conv2d(128, num_classes, kernel_size=kernel_size, padding=padding, bias=False, padding_mode='reflect')

    
    

    def forward(self, x):
        b, l, d = x.size()

        l = int(l **0.5)

        x = self.proj(x)
        x = self.drop(x)


        x =rearrange(x,'b (l w) (p1 p2 c) -> b c (l p1) (w p2)',  l = l, p1=self.patch_size, p2=self.patch_size )

        x = self.conv_in(x)

        x = self.conv(x)

        return x
    


class UVit(nn.Module):

    def __init__(self, dim, depth, act, n_head, ff_mult, img_size=256, patch_size=16, num_classes =1):
        super().__init__()

        
        self.patch_size = patch_size
        patch_dim = patch_size**2 * 3
        print(patch_dim)

        self.num_patches = (img_size//patch_size)**2

        self.patch_embed = PatchEmbed(patch_size=patch_size, 
                                      in_chans=3, 
                                      dim=dim)

        self.pos_embed = nn.Parameter(torch.zeros(1 ,self.num_patches, dim))

        self.in_blocks = nn.ModuleList([Block(dim=dim, 
                                              n_head=n_head, 
                                              ff_mult=ff_mult, 
                                              act=act, 
                                              skip=False) 
                                        for i in range(depth//2)])

        self.middle_block = nn.ModuleList([Block(dim=dim, 
                                                 n_head=n_head, 
                                                 ff_mult=ff_mult, 
                                                 act=act, 
                                                 skip=False) 
                                          for i in range(3)])

        self.out_blocks = nn.ModuleList([Block(dim=dim, 
                                               n_head=n_head, 
                                               ff_mult=ff_mult, 
                                               act=act, 
                                               skip=True)
                                        or i in range(depth//2)])

        self.out_layer = OutBlock(dim=dim, 
                                  hidden_dim=patch_dim, 
                                  kernel_size=3, 
                                  padding=1, 
                                  num_classes=num_classes, 
                                  patch_size=patch_size, 
                                  act=act)

        self.norm = RMSNorm(dim)

        
    
    def unet_forward(self, x):
        
        skips = []

        for block in self.in_blocks:
            x = block(x)
            skips.append(x)

        for block in self.middle_block:
            x = block(x)

        for block in self.out_blocks:
            x = block(x=x, skip=skips.pop())  
                  
        return x
    

    
    def forward(self, x):

        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = self.unet_forward(x)

        x = self.norm(x)

        x = self.out_layer(x)

        return x
