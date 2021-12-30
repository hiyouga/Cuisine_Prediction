import torch
import torch.nn as nn
from .layers import DynamicLSTM
from .layers import NoQueryAttention, mixup_process


class TextRNN(nn.Module):

    def __init__(self, rnn_type, num_layers, hidden_dim, configs):
        super(TextRNN, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        PN = configs['word_maxlen']+1
        PD = configs['position_dim']
        HD = hidden_dim
        C = configs['num_classes']

        if not configs['no_pretrain']:
            self.word_embed = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        else:
            self.word_embed = nn.Embedding(WN, WD, padding_idx=0)
        self.pos_embed = nn.Embedding(PN, PD, padding_idx=0)
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

    def forward(self, word, word_pos, lamda=None, indices=None):
        word_emb = self.dropout(self.word_embed(word))
        pos_emb = self.dropout(self.pos_embed(word_pos))
        word_feat = torch.cat((word_emb, pos_emb), dim=-1)
        text_len = torch.sum(word!=0, dim=-1)
        out, _ = self.rnn(word_feat, text_len.cpu())
        if hasattr(self, 'attention'):
            out = self.attention(out).squeeze(1)
        else:
            out = self.dropout(out.sum(dim=1).div(text_len.float().unsqueeze(-1)))
        if lamda is not None:
            out = mixup_process(out, lamda, indices)
        out = self.linear(out)
        return out


def textrnn(configs):
    return TextRNN('RNN', 1, 300, configs)
