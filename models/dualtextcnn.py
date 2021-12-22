import torch
import torch.nn as nn
from .layers import NoQueryAttention


class DualTextCNN(nn.Module):

    def __init__(self, kernel_num, kernel_sizes, configs):
        super(DualTextCNN, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        PN = configs['phrase_num']
        PD = 100
        WL = configs['word_maxlen']+1
        WLD = configs['position_dim']
        PL = configs['phrase_maxlen']+1
        PLD = configs['position_dim']
        KN = kernel_num
        KS = kernel_sizes
        C = configs['num_classes']

        self.word_embed = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        self.phrase_embed = nn.Embedding(PN, PD, padding_idx=0)
        self.word_pos_embed = nn.Embedding(WL, WLD, padding_idx=0)
        self.phrase_pos_embed = nn.Embedding(PL, PLD, padding_idx=0)
        self.word_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(WD+WLD, KN, K, padding=K//2, bias=True),
                nn.ReLU(inplace=True),
            ) for K in KS
        ])
        self.phrase_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(PD+PLD, KN, K, padding=K//2, bias=True),
                nn.ReLU(inplace=True),
            ) for K in KS
        ])
        self.linear = nn.Linear(len(KS) * KN * 2, C)
        self.dropout = nn.Dropout(configs['dropout'])
        if configs['score_function'] is not None:
            self.word_attention = NoQueryAttention(embed_dim=KN,
                                                   num_heads=configs['num_heads'],
                                                   score_function=configs['score_function'],
                                                   dropout=configs['dropout'])
            self.phrase_attention = NoQueryAttention(embed_dim=KN,
                                                     num_heads=configs['num_heads'],
                                                     score_function=configs['score_function'],
                                                     dropout=configs['dropout'])
        else:
            self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, word, phrase, word_pos, phrase_pos):
        word_emb = self.dropout(self.word_embed(word))
        phrase_emb = self.dropout(self.phrase_embed(phrase))
        word_pos = self.dropout(self.word_pos_embed(word_pos))
        phrase_pos = self.dropout(self.phrase_pos_embed(phrase_pos))
        word_feat = torch.cat((word_emb, word_pos), dim=-1).transpose(1, 2)
        phrase_feat = torch.cat((phrase_emb, phrase_pos), dim=-1).transpose(1, 2)
        out = list()
        for word_conv, phrase_conv in zip(self.word_conv, self.phrase_conv):
            word_out_i = word_conv(word_feat)
            phrase_out_i = phrase_conv(phrase_feat)
            if hasattr(self, 'word_attention'):
                word_out_i = self.word_attention(word_out_i.transpose(1, 2)).squeeze(1)
                phrase_out_i = self.phrase_attention(phrase_out_i.transpose(1, 2)).squeeze(1)
            else:
                word_out_i = self.maxpool(word_out_i).squeeze(-1)
                phrase_out_i = self.maxpool(phrase_out_i).squeeze(-1)
            out.extend([word_out_i, phrase_out_i])
        out = torch.cat(out, dim=-1)
        out = self.linear(self.dropout(out))
        return out


def dualtextcnn_128_345(configs):
    return DualTextCNN(128, [3,4,5], configs)


def dualtextcnn_256_345(configs):
    return DualTextCNN(256, [3,4,5], configs)
