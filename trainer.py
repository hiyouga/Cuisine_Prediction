import torch
import torch.nn as nn
import numpy as np


class Trainer:

    def __init__(self, model, args):
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self._mixup_alpha = args.mixup_alpha
        self._clip_norm = args.clip_norm
        self.params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=args.lr, weight_decay=args.decay)
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
        if self._mixup_alpha < 1e-6: # no mixup
            outputs = self.model(*inputs)
            loss = self.criterion(outputs, targets).mean()
        else:
            lamda = lamda = np.random.beta(self._mixup_alpha, self._mixup_alpha, size=targets.size(0))
            lamda = torch.tensor(lamda, dtype=torch.float, device=targets.device)
            indices = torch.randperm(targets.size(0), device=targets.device)
            outputs = self.model(*inputs, lamda=lamda, indices=indices)
            loss = self._mixup_criterion(outputs, targets, lamda=lamda, indices=indices)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self._clip_norm)
        self.optimizer.step()
        return outputs, loss

    def evaluate(self, inputs, targets):
        outputs = self.model(*inputs)
        loss = self.criterion(outputs, targets).mean()
        return outputs, loss

    def _mixup_criterion(self, outputs, targets, lamda, indices):
        outputs_a, outputs_b = outputs, outputs[indices, :]
        targets_a, targets_b = targets, targets[indices]
        loss_a = self.criterion(outputs_a, targets_a)
        loss_b = self.criterion(outputs_b, targets_b)
        return torch.mean(lamda * loss_a + (1 - lamda) * loss_b)
