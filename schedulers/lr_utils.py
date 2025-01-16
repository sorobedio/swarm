from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.optim import lr_scheduler
# from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler
import math




class CustomCosineWarmRestartScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr, min_lr, first_cycle_steps, cycle_mult=1, gamma=1.0, warmup_steps=0,
                 last_epoch=-1):
        """
        Custom cosine annealing schedulers with restarts and warm-up.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_lr (float): Initial maximum learning rate.
            min_lr (float): Minimum learning rate.
            first_cycle_steps (int): Number of steps in the first cycle.
            cycle_mult (float): Cycle length multiplier after each restart.
            gamma (float): Learning rate reduction factor after each cycle.
            warmup_steps (int): Steps for linear warm-up.
            last_epoch (int): The index of the last epoch.
        """
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.cycle_steps = first_cycle_steps  # Current cycle length
        self.cycle_count = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warm-up phase
            warmup_factor = self.last_epoch / self.warmup_steps
            return [self.min_lr + (self.max_lr - self.min_lr) * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            current_epoch_in_cycle = self.last_epoch - self.warmup_steps - sum(
                self.first_cycle_steps * (self.cycle_mult ** i) for i in range(self.cycle_count)
            )
            return [
                self.min_lr
                + (self.max_lr - self.min_lr)
                * (1 + math.cos(math.pi * current_epoch_in_cycle / self.cycle_steps)) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_steps + self.cycle_steps:
            # Restart learning rate schedule
            self.cycle_count += 1
            self.cycle_steps = int(self.first_cycle_steps * (self.cycle_mult ** self.cycle_count))
            self.max_lr *= self.gamma  # Reduce max learning rate after each cycle

        super().step(epoch)



class WarmUpAndDecayLR:
    def __init__(self, optimizer, warmup_steps=200, cosine_steps=200, gamma=1.0, T_mult=1):
        """
        Combines warm-up, cosine annealing with restarts, and max learning rate decay.

        Args:
            optimizer: Optimizer to schedule.
            warmup_steps: Number of steps for linear warm-up.
            cosine_steps: Number of steps for the first cosine annealing cycle.
            gamma: Multiplicative factor for max_lr after each restart.
            T_mult: Multiplicative factor for cycle length after each restart.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        self.gamma = gamma
        self.T_mult = T_mult
        self.current_step = 0
        self.restart_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]  # Use optimizer's base LRs
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_steps, T_mult=T_mult
        )

    def step(self):
        self.current_step += 1

        # Warm-up phase
        if self.current_step <= self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # Adjust max_lr after each restart
            if self.cosine_scheduler.T_i == self.cosine_scheduler.T_cur + 1:
                # Multiply base learning rates by gamma for the next restart
                self.base_lrs = [base_lr * self.gamma for base_lr in self.base_lrs]
                self.cosine_scheduler.base_lrs = self.base_lrs

            # Step the cosine annealing scheduler
            self.cosine_scheduler.step()

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
