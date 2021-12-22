import torch
import torch.nn as nn
from .layers import DynamicLSTM
from .layers import NoQueryAttention


class TextRNN(nn.Module):

    def __init__(self, rnn_type, num_layers, hidden_dim, configs):
        super(TextRNN, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        PL = configs['word_maxlen']+1
        PD = configs['position_dim']
        HD = hidden_dim
        C = configs['num_classes']

        self.word_embed = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        self.pos_embed = nn.Embedding(PL, PD, padding_idx=0)
        self.rnn = DynamicLSTM(WD+PD, HD, num_layers=num_layers, batch_first=True, bidirectional=True, rnn_type=rnn_type)
        self.linear = nn.Linear(2 * HD, C)
        self.dropout = nn.Dropout(configs['dropout'])
        if configs['score_function'] is not None:
            self.attention = NoQueryAttention(embed_dim=2 * HD,
                                              num_heads=configs['num_heads'],
                                              score_function=configs['score_function'],
                                              dropout=configs['dropout'])
        else:
            self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, word, word_pos):
        word_emb = self.dropout(self.word_embed(word))
        pos_emb = self.dropout(self.pos_embed(word_pos))
        word_feat = torch.cat((word_emb, pos_emb), dim=-1)
        text_len = torch.sum(word!=0, dim=-1)
        out, _ = self.rnn(word_feat, text_len.cpu())
        if hasattr(self, 'attention'):
            out = self.attention(out).squeeze(1)
        else:
            out = out.sum(dim=1).div(text_len.float().unsqueeze(-1))
        out = self.linear(self.dropout(out))
        return out


def textrnn_100(configs):
    return TextRNN('RNN', 1, 100, configs)


def textgru_100(configs):
    return TextRNN('GRU', 1, 100, configs)


def textgru_200(configs):
    return TextRNN('GRU', 1, 200, configs)


def textgru_300(configs):
    return TextRNN('GRU', 1, 300, configs)
