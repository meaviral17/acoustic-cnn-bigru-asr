import torch

def setup_ddp(model):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model
