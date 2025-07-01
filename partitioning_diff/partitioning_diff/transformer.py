import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .nn import timestep_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = th.sin(position * div_term)
        pe[0, :, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[0:1, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = th.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 1, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = th.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(d_model)
        self.norm_2 = nn.InstanceNorm1d(d_model)
        self.atten = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_model*2, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, atten_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout(self.atten(x2, x2, x2, atten_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class TransformerModel(nn.Module):

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        dataset,
        use_checkpoint,
        support_boundary,
        support_conditions,
        support_partial
    ):
        
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.time_channels = model_channels
        self.use_checkpoint = use_checkpoint

        self.support_boundary = support_boundary
        self.support_conditions = support_conditions
        self.support_partial = support_partial

        self.num_layers = 4

        self.activation = nn.ReLU()

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.time_channels),
        )

        self.input_emb = nn.Linear(self.in_channels, self.model_channels)
        
        self.condition_boundary_emb = nn.Linear(80, self.model_channels)
        self.condition_door_emb = nn.Linear(8, self.model_channels)
        self.condition_number_emb = nn.Linear(8, self.model_channels)
        self.condition_node_emb = nn.Linear(4, self.model_channels)
        self.condition_adjacency_emb = nn.Linear(8, self.model_channels)
        self.condition_partial_emb = nn.Linear(self.in_channels, self.model_channels)

        self.transformer_layers = nn.ModuleList([EncoderLayer(self.model_channels, 4, 0.1, self.activation) for x in range(self.num_layers)])

        self.output_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.output_linear2 = nn.Linear(self.model_channels, self.model_channels//2)
        self.output_linear3 = nn.Linear(self.model_channels//2, self.out_channels)

        print(f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
    
    def forward(self, x, timesteps, xtalpha, epsalpha, is_syn=False, **kwargs):
        
        prefix = 'syn_' if is_syn else ''

        x = x.permute([0, 2, 1]).float()

        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        time_emb = time_emb.unsqueeze(1)

        input_emb = self.input_emb(x)

        cond_boundary = kwargs[f'{prefix}cond_boundary']
        cond_boundary_emb = self.condition_boundary_emb(cond_boundary.float())
        cond_door = kwargs[f'{prefix}cond_door']
        cond_door_emb = self.condition_door_emb(cond_door.float())
        cond_number = kwargs[f'{prefix}cond_number']
        cond_number_emb = self.condition_number_emb(cond_number.float())
        cond_node = kwargs[f'{prefix}cond_node']
        cond_node_emb = self.condition_node_emb(cond_node.float())
        cond_adjacency = kwargs[f'{prefix}cond_adjacency']
        cond_adjacency_emb = self.condition_adjacency_emb(cond_adjacency.float())
        cond_partial = kwargs[f'{prefix}cond_partial']
        cond_partial_emb = self.condition_partial_emb(cond_partial.float())

        has_condition = False
        if self.support_boundary:
            if has_condition:
                cond_emb += (cond_boundary_emb + cond_door_emb)
            else:
                cond_emb = cond_boundary_emb + cond_door_emb
                has_condition = True

        if self.support_conditions == 'ncsla':
            if has_condition:
                cond_emb += (cond_number_emb + cond_node_emb + cond_adjacency_emb)
            else:
                cond_emb = cond_number_emb + cond_node_emb + cond_adjacency_emb
                has_condition = True

        if self.support_partial:
            if has_condition:
                cond_emb += cond_partial_emb
            else:
                cond_emb = cond_partial_emb
                has_condition = True
        
        if has_condition:
            out = input_emb + cond_emb + time_emb.repeat((1, input_emb.shape[1], 1))
        else:
            out = input_emb + time_emb.repeat((1, input_emb.shape[1], 1))

        for layer in self.transformer_layers:
            out = layer(out, kwargs[f'{prefix}atten_mask'])

        out = self.output_linear1(out)
        out = self.activation(out)
        out = self.output_linear2(out)
        out = self.output_linear3(out)

        out = out.permute([0, 2, 1])

        return out