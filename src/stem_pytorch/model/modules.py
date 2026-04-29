import math
import warnings
from typing import Union, Optional, Callable, Tuple, List, Sequence

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn, Size
from torch.nn import Conv3d, ModuleList
from torch.nn import functional as F

Shape = Union[Size, List[int], Tuple[int, ...]]
ModuleFactory = Union[Callable[[], nn.Module], Callable[[int], nn.Module]]


class PatchEmbedding3d(nn.Module):

    def __init__(self, input_size: Shape, patch_size: Union[int, Shape], embedding: int,
        strides: Optional[Union[int, Shape]] = None,
        build_normalization: Optional[ModuleFactory] = None
    ):
        super().__init__()
        # channel, time, height, width
        c, t, h, w = input_size
        # patch_time, patch_height, patch_width
        pt, ph, pw = (patch_size, patch_size, patch_size) if type(patch_size) is int else patch_size

        # configure the strides for conv3d
        if strides is None:
            # no specified means no overlap and gap between patches
            strides = (pt, ph, pw)
        elif type(strides) is int:
            # transform the side length of strides to 3D
            strides = (strides, strides, strides)

        self.projection = Conv3d(c, embedding, kernel_size=(pt, ph, pw), stride=strides)
        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
        self.rearrange = Rearrange("b d nt nh nw -> b (nt nh nw) d")

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x = self.rearrange(x)
        if self.has_norm:
            x = self.normalization(x)
        return x


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
        build_activation: Optional[ModuleFactory] = None,
        build_normalization: Optional[ModuleFactory] = None,
        normalization_after_activation: bool = False,
        dropout_rate: float = 0.
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        self.has_act = build_activation is not None
        if self.has_act:
            self.activation = build_activation()
        else:
            self.activation = None

        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
            self.norm_after_act = normalization_after_activation
        else:
            self.normalization = None

        self.has_dropout = dropout_rate > 0
        if self.has_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.has_act and self.has_norm:
            if self.norm_after_act:
                x = self.activation(x)
                x = self.normalization(x)
            else:
                x = self.normalization(x)
                x = self.activation(x)
        elif self.has_act and not self.has_norm:
            x = self.activation(x)
        elif not self.has_act and self.has_norm:
            x = self.normalization(x)

        if self.has_dropout:
            x = self.dropout(x)
        return x


class MLP(nn.Module):

    def __init__(self, neurons: Sequence[int],
        build_activation: Optional[ModuleFactory] = None, dropout_rate: float = 0.
    ):
        super().__init__()
        n_features = neurons[1:]
        self.layers: ModuleList[Linear] = ModuleList(
            [Linear(neurons[i], neurons[i + 1], True, build_activation, None,
                False, dropout_rate
            ) for i in range(len(n_features) - 1)
            ] + [
                Linear(neurons[-2], neurons[-1], True)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Attention(nn.Module):

    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
        proj_drop=0., attn_head_dim=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Expert(nn.Module):
    def __init__(self, dim, reduction_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, reduction_dim),
            nn.GELU(),
            nn.Linear(reduction_dim, dim)
        )

    def forward(self, z: Tensor) -> Tensor:
        return z + self.net(z)


class DEEM(nn.Module):
    def __init__(self, dim, num_experts=8, top_r=2, reduction_dim=64, noise_std=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_r = top_r
        self.noise_std = noise_std

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim, reduction_dim) for _ in range(num_experts)])

        # For load balancing loss calculation
        self.register_buffer("load", torch.zeros(num_experts))

    def forward(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # z: (B, N, D)
        b, n, d = z.shape
        z_flat = z.view(-1, d)  # (B*N, D)

        # 1. Routing scores
        logits = self.gate(z_flat)

        # 2. Noise for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # 3. Top-r selection
        top_logits, top_indices = torch.topk(logits, self.top_r, dim=-1)

        # 4. Routing weights (Softmax with max-shift)
        probs = F.softmax(top_logits, dim=-1)  # (B*N, top_r)

        # 5. Expert computation
        # For simplicity and sparse selection, we calculate per token
        # For better performance, we could group tokens per expert
        out = torch.zeros_like(z_flat)

        # Calculate load for balancing loss
        if self.training:
            # count occurrences of each expert in top_indices
            load = torch.zeros(self.num_experts, device=z.device)
            unique_indices, counts = torch.unique(top_indices, return_counts=True)
            load.scatter_add_(0, unique_indices, counts.float())
            self.load = 0.9 * self.load + 0.1 * (load / (b * n * self.top_r))

        # Weighted sum of experts
        for i in range(self.top_r):
            indices = top_indices[:, i]
            weights = probs[:, i].unsqueeze(-1)

            # This is naive implementation; for real MoE, we want to batch by expert
            # But with small r and small expert count, this is acceptable for now.
            for exp_idx in range(self.num_experts):
                mask = (indices == exp_idx)
                if mask.any():
                    token_out = self.experts[exp_idx](z_flat[mask])
                    out[mask] += weights[mask] * token_out

        return out.view(b, n, d)

    def get_balancing_loss(self):
        # L_balance = alpha * CV(L)^2
        # CV = std / mean
        if self.load.sum() == 0:
            return 0.
        mean = self.load.mean()
        std = self.load.std()
        cv = std / (mean + 1e-6)
        return cv ** 2


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
        init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        attn_head_dim=None
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            neurons=[dim, mlp_hidden_dim, dim],
            build_activation=act_layer,
            dropout_rate=drop
        )

        self.deem = None  # To be set via set_deem if Stage 2

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.attn(self.norm1(x))
            if self.deem is not None:
                x = x + self.mlp(self.norm2(x)) + self.deem(self.norm2(x))
            else:
                x = x + self.mlp(self.norm2(x))
        else:
            x = x + (self.gamma_1 * self.attn(self.norm1(x)))
            if self.deem is not None:
                x = x + (self.gamma_2 * (self.mlp(self.norm2(x)) + self.deem(self.norm2(x))))
            else:
                x = x + (self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
