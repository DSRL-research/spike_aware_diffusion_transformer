import copy

import torch
import torch.nn as nn


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Config asks for device '{name}' but CUDA is not available")
    return device


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad = False
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p, q in zip(model.parameters(), self.model.parameters()):
            q.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)


def cpu_state(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def stop(self) -> bool:
        return self.bad_epochs >= self.patience
