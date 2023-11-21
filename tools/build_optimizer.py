import torch

def build_optimizer(model, config):

    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(params=model.parameters(), momentum=config.momentum, lr=config.lr, weight_decay=config.weight_decay)

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    return optimizer



