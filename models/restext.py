import torch
import torch.nn as nn
from .layers import NoQueryAttention


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1x3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv1x7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        out1 = self.conv1x3(x)
        out2 = self.conv1x5(x)
        out3 = self.conv1x7(x)
        out = out1 + out2 + out3
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super(BasicBlock, self).__init__()
        self.residual = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            ConvBlock(out_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual(x) + x
        out = self.relu(out)
        return out


class ResText(nn.Module):

    def __init__(self, kernel_num, num_blocks, configs):
        super(ResText, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        PL = configs['word_maxlen']+1
        PD = configs['position_dim']
        KN = kernel_num
        C = configs['num_classes']

        self.word_embed = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        self.pos_embed = nn.Embedding(PL, PD, padding_idx=0)
        self.conv1 = nn.Sequential(
            ConvBlock(WD+PD, KN),
            nn.BatchNorm1d(KN),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList([
            BasicBlock(KN, KN, dropout=configs['dropout'])
            for _ in range(num_blocks)])
        self.linear = nn.Linear(KN, C)
        self.dropout = nn.Dropout(configs['dropout'])
        if configs['score_function'] is not None:
            self.attention = NoQueryAttention(embed_dim=KN,
                                              num_heads=configs['num_heads'],
                                              score_function=configs['score_function'],
                                              dropout=configs['dropout'])
        else:
            self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, word, word_pos):
        word_emb = self.dropout(self.word_embed(word))
        pos_emb = self.dropout(self.pos_embed(word_pos))
        word_feat = torch.cat((word_emb, pos_emb), dim=-1).transpose(1, 2)
        out = self.conv1(word_feat)
        for layer in self.layers:
            out = layer(out)
        if hasattr(self, 'attention'):
            out = self.attention(out.transpose(1, 2)).squeeze(1)
        else:
            out = self.maxpool(out).squeeze(-1)
        out = self.linear(self.dropout(out))
        return out


def restext_128_1(configs):
    return ResText(128, 1, configs)


def restext_256_1(configs):
    return ResText(256, 1, configs)


def restext_256_2(configs):
    return ResText(256, 2, configs)
