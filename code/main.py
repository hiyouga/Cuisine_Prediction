import os
import sys
import time
import torch
import random
import models
import logging
import argparse
import warnings
import datetime
import numpy as np
from trainer import Trainer
from data_utils import load_data


class Instructor:

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
        self._print_args()
        dataloaders = load_data(batch_size=args.batch_size, dev_ratio=args.dev_ratio, no_data_aug=args.no_data_aug)
        self.train_dataloader, self.dev_dataloader, self.test_dataloader, self.tokenizer, embedding_matrix = dataloaders
        configs = {
            'num_classes': len(self.tokenizer.vocab['label']),
            'phrase_num': len(self.tokenizer.vocab['phrase']),
            'word_maxlen': self.tokenizer.maxlen['word'],
            'phrase_maxlen': self.tokenizer.maxlen['phrase'],
            'embedding_matrix': embedding_matrix,
            'position_dim': args.position_dim,
            'dropout': args.dropout,
            'score_function': args.score_function,
            'num_heads': args.num_heads,
            'no_pretrain': args.no_pretrain
        }
        self.logger.info('=> creating model')
        self.trainer = Trainer(args.model_class(configs), args)
        self.trainer.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info(f"=> cuda memory allocated: {torch.cuda.memory_allocated(self.args.device.index)}")

    def _print_args(self):
        print('TRAINING ARGUMENTS:')
        for arg in vars(self.args):
            print(f">>> {arg}: {getattr(self.args, arg)}")

    def _update_record(self, epoch, val_loss, val_acc, best_record):
        if (val_acc > best_record['val_acc']) or (val_acc == best_record['val_acc'] and val_loss < best_record['val_loss']):
            best_record['epoch'] = epoch
            best_record['val_acc'] = val_acc
            best_record['val_loss'] = val_loss
            best_record['model_state'] = self.trainer.save_state_dict()
        return best_record

    def _train(self, dataloader):
        train_loss, n_correct, n_train = 0, 0, 0
        n_batch = len(dataloader)
        self.trainer.train_mode()
        for i_batch, sample_batched in enumerate(dataloader):
            inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
            targets = sample_batched['target'].to(self.args.device)
            outputs, loss = self.trainer.train(inputs, targets)
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_train += targets.size(0)
            if not self.args.no_bar:
                ratio = int((i_batch+1)*50/n_batch) # process bar
                print(f"[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%", end='\r')
        if not self.args.no_bar:
            print()
        return train_loss / n_train, n_correct / n_train

    @torch.no_grad()
    def _validate(self, dataloader, inference=False):
        val_loss, n_correct, n_val = 0, 0, 0
        all_cid, all_pred = list(), list()
        n_batch = len(dataloader)
        self.trainer.eval_mode()
        for i_batch, sample_batched in enumerate(dataloader):
            inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
            targets = sample_batched['target'].to(self.args.device)
            outputs, loss = self.trainer.evaluate(inputs, targets)
            if inference:
                all_cid.extend(sample_batched['cid'].tolist())
                all_pred.extend([self.tokenizer.vocab['label'].id_to_word(pred.item()) for pred in torch.argmax(outputs, -1)])
            else:
                val_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_val += targets.size(0)
            if not self.args.no_bar:
                ratio = int((i_batch+1)*50/n_batch) # process bar
                print(f"[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%", end='\r')
        if not self.args.no_bar:
            print()
        if inference:
            return all_cid, all_pred
        else:
            return val_loss / n_val, n_correct / n_val

    def run(self):
        best_record = {'epoch': 0, 'val_loss': 0, 'val_acc': 0, 'model_state': None}
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(self.train_dataloader)
            val_loss, val_acc = self._validate(self.dev_dataloader)
            self.trainer.lr_scheduler_step()
            best_record = self._update_record(epoch+1, val_loss, val_acc, best_record)
            self.logger.info(f"{epoch+1}/{self.args.num_epoch} - {100*(epoch+1)/self.args.num_epoch:.2f}%")
            self.logger.info(f"[train] loss: {train_loss:.4f}, acc: {train_acc*100:.2f}")
            self.logger.info(f"[val] loss: {val_loss:.4f}, acc: {val_acc*100:.2f}")
        self.logger.info(f"best val loss: {best_record['val_loss']:.4f}, best val acc: {best_record['val_acc']*100:.2f}")
        if best_record['model_state'] is not None:
            self.trainer.load_state_dict(best_record['model_state'])
        torch.save(self.trainer.save_state_dict(), os.path.join('state_dict', f"{self.args.timestamp}.pt"))
        self.logger.info(f"model saved: {self.args.timestamp}.pt")
        all_cid, all_pred = self._validate(self.test_dataloader, inference=True)
        with open(f"{self.args.model_name}_{self.args.timestamp}_{best_record['val_acc']*100:.2f}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join([f"{cid} {pred}" for cid, pred in zip(all_cid, all_pred)]))
        self.logger.info(f"submission result saved: {self.args.model_name}_{self.args.timestamp}_{best_record['val_acc']*100:.2f}.txt")

    @torch.no_grad()
    def ensemble(self):
        n_batch = len(self.test_dataloader)
        all_cid, all_output, all_pred = list(), list(), list()
        self.trainer.eval_mode()
        for i, checkpoint in enumerate(self.args.ensemble):
            self.logger.info(f"Running {i+1}/{len(self.args.ensemble)} checkpoint...")
            self.trainer.load_state_dict(torch.load(os.path.join('state_dict', f"{checkpoint}.pt"), map_location=self.args.device))
            for i_batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
                targets = sample_batched['target'].to(self.args.device)
                outputs, loss = self.trainer.evaluate(inputs, targets)
                if i == 0:
                    all_cid.extend(sample_batched['cid'].tolist())
                    all_output.append(outputs)
                else:
                    all_output[i_batch] += outputs
                if not self.args.no_bar:
                    ratio = int((i_batch+1)*50/n_batch) # process bar
                    print(f"[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%", end='\r')
            if not self.args.no_bar:
                print()
        for i_batch in range(n_batch):
            all_pred.extend([self.tokenizer.vocab['label'].id_to_word(pred.item()) for pred in torch.argmax(all_output[i_batch], -1)])
        with open(f"ensemble_{len(self.args.ensemble)}_{self.args.timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join([f"{cid} {pred}" for cid, pred in zip(all_cid, all_pred)]))
        self.logger.info(f"submission result saved: ensemble_{len(self.args.ensemble)}_{self.args.timestamp}.txt")

    @torch.no_grad()
    def t_sne(self):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        plt.switch_backend('Agg')
        n_batch = len(self.train_dataloader)
        all_feature, all_label = list(), list()
        def hook_fn(model, input):
            all_feature.append(input[0].cpu().numpy())
        hook = self.trainer.model.linear.register_forward_pre_hook(hook_fn)
        self.trainer.eval_mode()
        self.trainer.load_state_dict(torch.load(os.path.join('state_dict', f"{self.args.checkpoint}.pt"),
                                                map_location=self.args.device))
        for i_batch, sample_batched in enumerate(self.train_dataloader):
            inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
            targets = sample_batched['target'].to(self.args.device)
            _ = self.trainer.evaluate(inputs, targets)
            all_label.append(targets.cpu().numpy())
            if not self.args.no_bar:
                ratio = int((i_batch+1)*50/n_batch) # process bar
                print(f"[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%", end='\r')
        if not self.args.no_bar:
            print()
        hook.remove()
        all_feature = np.concatenate(all_feature, axis=0)[:200, :]
        all_label = np.concatenate(all_label, axis=0)[:200]
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        data = tsne.fit_transform(all_feature)
        plt.figure()
        x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
        data = (data - x_min) / (x_max - x_min) * 0.9 + 0.05
        for i in range(data.shape[0]):
            color = plt.cm.rainbow(all_label[i] / 19)
            plt.text(data[i, 0], data[i, 1], str(all_label[i]), backgroundcolor=color, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"vis_{self.args.timestamp}.pdf", format='pdf', transparent=True, dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))
    input_colses = {
        'textcnn': ['word', 'word_pos'],
        'textrnn': ['word', 'word_pos'],
        'restext': ['word', 'word_pos'],
        'dualtextcnn': ['word', 'phrase', 'word_pos', 'phrase_pos']
    }
    parser = argparse.ArgumentParser(description='Trainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ''' dataset '''
    parser.add_argument('--dev_ratio', type=float, default=0.05, help='Ratio between 0 and 1 for spliting development set.')
    parser.add_argument('--no_data_aug', default=False, action='store_true', help='Disable data augmentation.')
    ''' model '''
    parser.add_argument('--model_name', type=str, default='textcnn', choices=model_names, help='Classifier model architecture.')
    parser.add_argument('--position_dim', type=int, default=10, help='Dimension of position embedding.')
    parser.add_argument('--score_function', type=str, default=None, help='Score function for attention layer.')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads in multihead attention layer.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')
    parser.add_argument('--no_pretrain', default=False, action='store_true', help='Do not use pretrained embeddings.')
    ''' optimization '''
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Alpha parameter for mixup.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--clip_norm', type=int, default=50, help='Maximum norm of gradients.')
    ''' ensemble '''
    parser.add_argument('--ensemble', type=str, default=None, help='Models for ensembling.')
    ''' t-SNE '''
    parser.add_argument('--checkpoint', type=str, default=None, help='Model for t-SNE visualization.')
    ''' environment '''
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Selected device.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--timestamp', type=str, default=None, help='Experiment timestamp.')
    parser.add_argument('--no_bar', default=False, action='store_true', help='Disable process bar.')
    parser.add_argument('--no_backend', default=False, action='store_true', help='Use frontend matplotlib.')
    args = parser.parse_args()
    args.model_class = models.__dict__[args.model_name]
    args.inputs_cols = input_colses[args.model_name]
    args.ensemble = [int(e.strip()) for e in args.ensemble.split(',')] if args.ensemble else None
    args.log_name = f"{args.model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:]}.log"
    args.timestamp = args.timestamp if args.timestamp else str(int(time.time())) + format(random.randint(0, 999), '03')
    args.seed = args.seed if args.seed else random.randint(0, 2**32-1)
    args.device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' set seeds '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ''' global settings '''
    for dir_name in ['dats', 'logs', 'state_dict']:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    warnings.simplefilter("ignore")
    ins = Instructor(args)
    if args.ensemble:
        ins.ensemble()
    elif args.checkpoint:
        ins.t_sne()
    else:
        ins.run()
