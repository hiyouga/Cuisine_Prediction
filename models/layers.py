import torch
import torch.nn as nn


class DynamicLSTM(nn.Module):
    """
    LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer.
        bidirectional: If True, becomes a bidirectional RNN. Default: False
        rnn_type: {LSTM, GRU, RNN}.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        
        Args:
            x: sequence embeddings
            x_len: squence lengths
        """
        total_length = x.size(1) if self.batch_first else x.size(0)
        ''' sort '''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx] if self.batch_first else x[:, x_sort_idx]
        ''' pack '''
        x_emb_p = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        ''' process '''
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        ''' unsort '''
        ht = ht[:, x_unsort_idx] # (num_directions * num_layers, batch_size, hidden_size)
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = nn.utils.rnn.pad_packed_sequence(out_pack,
                                                      batch_first=self.batch_first,
                                                      total_length=total_length)
            out = out[x_unsort_idx] if self.batch_first else out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)


class MultiHeadAttention(nn.Module):
    """
    Implement Multihead Attention Layer.
    
    Args:
        embed_dim: Input dimension of the layer.
        hidden_dim: Dimension of each head.
        output_dim: Output dimension.
        num_heads: Number of parallel attention heads.
        score_function: Function for computing attention scores. {dot_product, scaled_dot_product, mlp, bilinear}
        dropout: Dropout probability on outputs. Default: 0
        bias: If True, adds bias to input/output projection layers. Default: True
        add_bias_kv: If True, add bias to the key and value sequences. Default: False
        verbose: If True, output the attention score. Default: False
    """
    def __init__(self, embed_dim, hidden_dim=None, output_dim=None, num_heads=1,
                 score_function='dot_product', dropout=0, bias=True, add_bias_kv=False, verbose=False):
        super(MultiHeadAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // num_heads
        if output_dim is None:
            output_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.score_function = score_function
        self.scale = hidden_dim ** -0.5
        self.verbose = verbose

        self.w_k = nn.Linear(embed_dim, num_heads * hidden_dim, bias=add_bias_kv)
        self.w_q = nn.Linear(embed_dim, num_heads * hidden_dim, bias=bias)
        self.w_v = nn.Linear(embed_dim, num_heads * hidden_dim, bias=add_bias_kv)
        self.proj = nn.Linear(num_heads * hidden_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output_dim, eps=1e-6)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        if score_function == 'mlp':
            self.attn_weight = nn.Parameter(torch.empty(2 * hidden_dim))
            nn.init.uniform_(self.attn_weight, -self.scale, self.scale)
        elif score_function == 'bilinear':
            self.attn_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            nn.init.uniform_(self.attn_weight, -self.scale, self.scale)

    def forward(self, k, q, v=None, mask=None):
        if len(q.shape) == 2:
            q = q.unsqueeze(1) # (batch_size, q_len, embed_dim)
        if len(k.shape) == 2:
            k = k.unsqueeze(1) # (batch_size, k_len, embed_dim)
        mb_size = k.size(0)
        k_len = k.size(1)
        q_len = q.size(1)
        ''' Pass through the pre-attention projection: (batch_size, length, num_heads * hidden_dim) '''
        ''' Separate different heads: (batch_size, length, num_heads, hidden_dim) '''
        q = self.w_q(q).view(mb_size, q_len, self.num_heads, self.hidden_dim)
        k = self.w_k(k).view(mb_size, k_len, self.num_heads, self.hidden_dim)
        if v is not None:
            v_len = v.size(1)
            v = self.w_v(v).view(mb_size, v_len, self.num_heads, self.hidden_dim)
        else:
            v = k
        if mask is not None:
            mask = mask.unsqueeze(-1) # For head axis broadcasting
        if self.score_function == 'dot_product':
            attn = torch.einsum('bind,bjnd->bijn', q, k) # (batch_size, q_len, k_len, num_heads)
        elif self.score_function == 'scaled_dot_product':
            attn = torch.einsum('bind,bjnd->bijn', q, k) # (batch_size, q_len, k_len, num_heads)
            attn.mul_(self.scale)
        elif self.score_function == 'mlp':
            padded_k = k.unsqueeze(1).expand(-1, q_len, -1, -1, -1)
            padded_q = q.unsqueeze(2).expand(-1, -1, k_len, -1, -1)
            qk = torch.cat((padded_q, padded_k), dim=-1) # (batch_size, q_len, k_len, num_heads, hidden_dim*2)
            attn = self.tanh(torch.einsum('bijnd,d->bijn', qk, self.attn_weight)) # (batch_size, q_len, k_len, num_heads)
        elif self.score_function == 'bilinear':
            attn = torch.einsum('bind,dd,bjnd->bijn', q, self.attn_weight, k) # (batch_size, q_len, k_len, num_heads)
        else:
            raise ValueError
        if mask is not None:
            attn.masked_fill_(mask==0, -float('inf'))
        attn = self.softmax(attn)
        out = torch.einsum('bijn,bjnd->bind', attn, v) # (batch_size, q_len, num_heads, hidden_dim)
        ''' Combine the last two dimensions to concatenate all the heads: (batch_size, q_len, num_heads * hidden_dim) '''
        out = out.contiguous().view(mb_size, q_len, self.num_heads * self.hidden_dim)
        out = self.proj(out) # (batch_size, q_len, output_dim)
        out = self.dropout(out)
        out = self.layernorm(out)
        if self.verbose:
            return out, attn
        else:
            return out


class NoQueryAttention(MultiHeadAttention):
    """
    Implement Multihead Attention Layer without query.
    
    Args:
        q_len: Expected length of the query. Default: 1
    """
    def __init__(self, *args, q_len=1, **kwargs):
        super(NoQueryAttention, self).__init__(**kwargs)
        self.q_len = q_len
        self.q = nn.Parameter(torch.empty(q_len, self.embed_dim))
        nn.init.uniform_(self.q, -self.embed_dim ** -0.5, self.embed_dim ** -0.5)

    def forward(self, k, v=None, mask=None):
        mb_size = k.size(0)
        q = self.q.expand(mb_size, -1, -1) # (batch_size, q_len, embed_dim)
        return super().forward(k, q, v, mask)
