import torch
import torch.nn as nn
from .layers import NoQueryAttention


class TextCNN_Attn(nn.Module):

    def __init__(self, kernel_num, kernel_sizes, configs):
        super(TextCNN_Attn, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        KN = kernel_num
        KS = kernel_sizes
        C = configs['num_classes']

        self.embed = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(WD, KN, K, padding=K//2, bias=True),
                nn.ReLU(inplace=True),
            ) for K in KS
        ])
        self.attention = NoQueryAttention(embed_dim=KN, score_function='scaled_dot_product', dropout=0.3)
        self.linear = nn.Linear(len(KS) * KN, C)
        self.dropout = nn.Dropout(0.3)

    def forward(self, word):
        word_emb = self.dropout(self.embed(word)).transpose(1, 2)
        attn_out = list()
        for conv in self.conv:
            cnn_out_i = conv(word_emb).transpose(1, 2)
            attn_out_i, _ = self.attention(cnn_out_i)
            attn_out.append(attn_out_i.squeeze(1))
        attn_out = torch.cat(attn_out, dim=-1)
        output = self.linear(self.dropout(attn_out))
        return output


def textcnn_attn(configs):
    return TextCNN_Attn(256, [3,4,5], configs)
