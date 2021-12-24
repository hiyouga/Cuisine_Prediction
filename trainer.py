import torch
import torch.nn as nn
from amp import AMP


class Trainer:

    def __init__(self, model, args):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self._clip_norm = args.clip_norm
        self.params = filter(lambda p: p.requires_grad, model.parameters())
        if args.optimizer == 'sgd':
            self.optimizer = AMP(self.params, args.lr, args.epsilon, momentum=0.9, weight_decay=args.decay)
        elif args.optimizer == 'adam':
            self.optimizer = AMP(self.params, args.lr, args.epsilon, weight_decay=args.decay, base_optimizer=torch.optim.Adam)
        else:
            raise ValueError
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.num_epoch)

    def lr_scheduler_step(self):
        self.scheduler.step()

    def to(self, device):
        self.model.to(device)

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def save_state_dict(self):
        return self.model.state_dict()

    def train(self, inputs, targets):
        def closure():
            self.optimizer.zero_grad()
            outputs = self.model(*inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, self._clip_norm)
            return outputs, loss
        outputs, loss = self.optimizer.step(closure)
        return outputs, loss

    def evaluate(self, inputs, targets):
        outputs = self.model(*inputs)
        loss = self.criterion(outputs, targets)
        return outputs, loss
