import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTextCNN(nn.Module):

    def __init__(self, kernel_num, kernel_sizes, configs):
        super(DualTextCNN, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        PN = configs['phrase_num']
        PD = 200
        WL = configs['word_maxlen']+1
        WLD = 30
        PL = configs['phrase_maxlen']+1
        PLD = 30
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, word, phrase, word_pos, phrase_pos):
        word_emb = self.dropout(self.word_embed(word))
        phrase_emb = self.dropout(self.phrase_embed(phrase))
        word_pos = self.dropout(self.word_pos_embed(word_pos))
        phrase_pos = self.dropout(self.phrase_pos_embed(phrase_pos))
        word_feat = torch.cat((word_emb, word_pos), dim=-1)
        phrase_feat = torch.cat((phrase_emb, phrase_pos), dim=-1)
        maxpool_out = list()
        for word_conv, phrase_conv in zip(self.word_conv, self.phrase_conv):
            word_out_i = word_conv(word_feat.transpose(1, 2))
            phrase_out_i = phrase_conv(phrase_feat.transpose(1, 2))
            word_maxpool_i = F.max_pool1d(word_out_i, word_out_i.size(-1)).squeeze(-1)
            phrase_maxpool_i = F.max_pool1d(phrase_out_i, phrase_out_i.size(-1)).squeeze(-1)
            maxpool_out.extend([word_maxpool_i, phrase_maxpool_i])
        maxpool_out = torch.cat(maxpool_out, dim=-1)
        output = self.linear(maxpool_out)
        return output


def dualtextcnn(configs):
    return DualTextCNN(256, [3,4,5], configs)
