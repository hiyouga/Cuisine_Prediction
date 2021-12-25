import os
import json
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class Vocab:

    def __init__(self, vocab_list, add_pad=True, add_unk=True):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad: # pad_id should be zero (for mask)
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._vocab_dict[self.pad_word] = self.pad_id
            self._length += 1
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._vocab_dict[self.unk_word] = self.unk_id
            self._length += 1
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, idx):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(idx, self.unk_word)
        return self._reverse_vocab_dict[idx]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length


class Tokenizer:

    def __init__(self, word_vocab, phrase_vocab, label_vocab, lower):
        self.vocab = {
            'word': word_vocab,
            'phrase': phrase_vocab,
            'label': label_vocab
        }
        self.maxlen = {
            'word': 128,
            'phrase': 64,
            'label': -1
        }
        self.lower = lower

    @classmethod
    def from_files(cls, fnames, lower=True):
        all_words = set()
        all_phrase = set()
        all_label = set()
        for fname in fnames:
            fdata = json.load(open(os.path.join('data', fname), 'r', encoding='utf-8'))
            for data in fdata.values():
                words = (' '.join(data['ingredients'])).split()
                phrases = data['ingredients']
                all_words.update([w.lower() if lower else w for w in words])
                all_phrase.update([p.lower() if lower else p for p in phrases])
                all_label.add(data['cuisine'])
        return cls(word_vocab=Vocab(all_words),
                   phrase_vocab=Vocab(all_phrase),
                   label_vocab=Vocab(all_label, add_pad=False, add_unk=False),
                   lower=lower)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def to_sequence(self, tokens, vocab_name, reverse=False, padding='post', truncating='post'):
        sequence = [self.vocab[vocab_name].word_to_id(t.lower() if self.lower else t) for t in tokens]
        if reverse:
            sequence.reverse()
        return self.pad_sequence(sequence,
                                 pad_id=self.vocab[vocab_name].pad_id,
                                 maxlen=self.maxlen[vocab_name],
                                 padding=padding,
                                 truncating=truncating)

    def position_sequence(self, length, vocab_name, reverse=False, padding='post', truncating='post'):
        sequence = list(range(1, length+1))
        if reverse:
            sequence.reverse()
        return self.pad_sequence(sequence,
                                 pad_id=0,
                                 maxlen=self.maxlen[vocab_name],
                                 padding=padding,
                                 truncating=truncating)


class FoodDataset(Dataset):

    def __init__(self, fname, tokenizer, split, no_data_aug):
        self.tokenizer = tokenizer
        self.no_data_aug = no_data_aug
        cache_file = os.path.join('dats', f"{split}.dat")
        if os.path.exists(cache_file):
            print(f"loading dataset: {cache_file}")
            dataset = pickle.load(open(cache_file, 'rb'))
        else:
            print('building dataset...')
            dataset = list()
            fdata = json.load(open(os.path.join('data', fname), 'r', encoding='utf-8'))
            for cid, data in fdata.items():
                dataset.append([
                    int(cid),
                    data['ingredients'],
                    tokenizer.vocab['label'].word_to_id(data['cuisine']) if split != 'test' else 0
                ])
            pickle.dump(dataset, open(cache_file, 'wb'))
        self._dataset = dataset

    def __getitem__(self, index):
        cid, phrases, label = self._dataset[index]
        if not self.no_data_aug: # do data augmentation
            phrases = phrases[:]
            random.shuffle(phrases)
            if len(phrases) > 1:
                phrases = phrases[:-1]
        words = (' '.join(phrases)).split()
        return {
            'cid': cid,
            'word': self.tokenizer.to_sequence(words, vocab_name='word'),
            'phrase': self.tokenizer.to_sequence(phrases, vocab_name='phrase'),
            'word_pos': self.tokenizer.position_sequence(len(words), vocab_name='word'),
            'phrase_pos': self.tokenizer.position_sequence(len(phrases), vocab_name='phrase'),
            'target': label
        }

    def __len__(self):
        return len(self._dataset)


def _load_wordvec(embed_file, word_dim, vocab=None):
    with open(embed_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        word_vec['<pad>'] = np.zeros(word_dim).astype('float32')
        for line in f:
            tokens = line.rstrip().split()
            if (len(tokens)-1) != word_dim:
                continue
            if tokens[0] == '<pad>' or tokens[0] == '<unk>':
                continue
            if vocab is None or vocab.has_word(tokens[0]):
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        return word_vec


def build_embedding_matrix(vocab, word_dim=300):
    cache_file = os.path.join('dats', 'embedding_matrix.dat')
    embed_file = os.path.join('..', 'glove', 'glove.840B.300d.txt')
    if os.path.exists(cache_file):
        print(f"loading embedding matrix: {cache_file}")
        embedding_matrix = pickle.load(open(cache_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), word_dim)).astype('float32')
        word_vec = _load_wordvec(embed_file, word_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(cache_file, 'wb'))
    return embedding_matrix


def build_tokenizer(fnames):
    cache_file = os.path.join('dats', 'tokenizer.dat')
    if os.path.exists(cache_file):
        print(f"loading tokenizer: {cache_file}")
        tokenizer = pickle.load(open(cache_file, 'rb'))
    else:
        print('building tokenizer...')
        tokenizer = Tokenizer.from_files(fnames=fnames)
        pickle.dump(tokenizer, open(cache_file, 'wb'))
    return tokenizer


def load_data(batch_size, dev_ratio, no_data_aug):
    tokenizer = build_tokenizer(fnames=['train.json'])
    embedding_matrix = build_embedding_matrix(tokenizer.vocab['word'])
    trainset = FoodDataset('train.json', tokenizer, split='train', no_data_aug=no_data_aug)
    dev_len = int(len(trainset) * dev_ratio) # split validation set
    assert dev_len != 0
    trainset, devset = random_split(trainset, (len(trainset)-dev_len, dev_len))
    testset = FoodDataset('test.json', tokenizer, split='test', no_data_aug=True)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    dev_dataloader = DataLoader(devset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_dataloader, dev_dataloader, test_dataloader, tokenizer, embedding_matrix
