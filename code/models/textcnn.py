import torch
import torch.nn as nn
from .layers import NoQueryAttention, mixup_process


class TextCNN(nn.Module):

    def __init__(self, kernel_num, kernel_sizes, configs):
        super(TextCNN, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        PN = configs['word_maxlen']+1
        PD = configs['position_dim']
        KN = kernel_num
        KS = kernel_sizes
        C = configs['num_classes']

        if not configs['no_pretrain']:
            self.word_embed = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        else:
            self.word_embed = nn.Embedding(WN, WD, padding_idx=0)
        self.pos_embed = nn.Embedding(PN, PD, padding_idx=0)
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(WD+PD, KN, K, padding=K//2, bias=True),
                nn.ReLU(inplace=True),
            ) for K in KS
        ])
        self.linear = nn.Linear(len(KS) * KN, C)
        self.dropout = nn.Dropout(configs['dropout'])
        if configs['score_function'] is not None:
            self.attention = NoQueryAttention(embed_dim=KN,
                                              num_heads=configs['num_heads'],
                                              score_function=configs['score_function'],
                                              dropout=configs['dropout'])
        else:
            self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, word, word_pos, lamda=None, indices=None):
        word_emb = self.dropout(self.word_embed(word))
        pos_emb = self.dropout(self.pos_embed(word_pos))
        word_feat = torch.cat((word_emb, pos_emb), dim=-1).transpose(1, 2)
        out = list()
        for conv in self.conv:
            cnn_out_i = conv(word_feat)
            if hasattr(self, 'attention'):
                out_i = self.attention(cnn_out_i.transpose(1, 2)).squeeze(1)
            else:
                out_i = self.dropout(self.maxpool(cnn_out_i).squeeze(-1))
            out.append(out_i)
        out = torch.cat(out, dim=-1)
        if lamda is not None:
            out = mixup_process(out, lamda, indices)
        out = self.linear(out)
        return out


def textcnn(configs):
    return TextCNN(256, [3,4,5], configs)
