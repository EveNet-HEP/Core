import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)  # binarize
        return torch.div(x, keep_prob) * random_tensor


class RandomDrop(nn.Module):
    def __init__(self, drop_prob: float, num_skip: int):
        super().__init__()
        self.drop_prob = drop_prob
        self.num_skip = num_skip

    def forward(self, x):
        if not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize [ IN-PLACE OPERATION ]

        # Create a new tensor instead of modifying in-place
        output = x.clone()
        output[:, :, self.num_skip:] = x[:, :, self.num_skip:] * random_tensor.unsqueeze(2)
        return output


class TalkingHeadAttention(nn.Module):
    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate

        head_dim = self.projection_dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(projection_dim, projection_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(projection_dim, projection_dim)
        self.proj_l = nn.Linear(self.num_heads, self.num_heads)
        self.proj_w = nn.Linear(self.num_heads, self.num_heads)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x, int_matrix=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if int_matrix is not None:
            attn = attn + int_matrix


        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn = attn + ((1.0 - mask.float()) * -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LayerScale(nn.Module):
    def __init__(self, init_values, projection_dim):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(projection_dim))

    def forward(self, x, mask=None):
        if mask is not None:
            return x * self.gamma * mask
        else:
            return x * self.gamma


def get_sinusoidal_positional_encoding(T, D, device='cpu'):
    """
    Returns a (T, D) tensor of sinusoidal positional encodings.
    """
    position = torch.arange(T, dtype=torch.float, device=device).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / D))

    pe = torch.zeros(T, D, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

    return pe  # shape: (T, D)

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")