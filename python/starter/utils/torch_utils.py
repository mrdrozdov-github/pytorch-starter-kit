from collections import OrderedDict

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


def recursively_set_device(inp, gpu):
    if hasattr(inp, 'keys'):
        for k in inp.keys():
            inp[k] = recursively_set_device(inp[k], gpu)
    elif isinstance(inp, list):
        return [recursively_set_device(ii, gpu) for ii in inp]
    elif isinstance(inp, tuple):
        return (recursively_set_device(ii, gpu) for ii in inp)
    elif hasattr(inp, 'cpu'):
        if gpu >= 0:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp


def torch_save(models, optimizers, meta, filename, gpu=-1):
    tosave = OrderedDict(
        models={k: m.state_dict() for k, m in models.items()},
        optimizers={k: o.state_dict() for k, o in optimizers},
        )

    # Move to CPU for convenience when loading later.
    for vv in tosave:
        recursively_set_device(vv, gpu=-1)

    tosave['meta'] = meta

    torch.save(tosave, filename)

    # Move back to original device to continue training.
    for vv in tosave:
        recursively_set_device(vv, gpu=gpu)


def torch_load(models, optimizers, filename):
    loaded = torch.load(filename)

    for k, m in models.items():
        m.load_state_dict(loaded['models'][k])

    for k, o in optimizers.items():
        o.load_state_dict(loaded['optimizers'][k])

    return loaded['meta']
