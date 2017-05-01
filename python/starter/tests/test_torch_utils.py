import unittest
import tempfile

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


from starter.utils.torch_utils import torch_save, torch_load


class TorchUtilsTestCase(unittest.TestCase):
    def test_load(self):
        in_dim = 11
        out_dim = 1

        model_to_save = nn.Linear(in_dim, out_dim, bias=False)
        model_to_load = nn.Linear(in_dim, out_dim, bias=False)

        for p in model_to_save.parameters():
            p.data.fill_(11)

        for p in model_to_load.parameters():
            p.data.fill_(0)

        assert model_to_load.weight.data[0,0] == 0.

        optimizers = {}
        meta = None
        temp = tempfile.NamedTemporaryFile()
        filename = temp.name

        # Save to and load from temporary file.
        torch_save({'m': model_to_save}, optimizers, meta, filename)
        torch_load({'m': model_to_load}, optimizers, filename)

        # Check value of scalars.
        assert model_to_load.weight.data[0,0] == 11.
        assert model_to_save.weight.data[0,0] == model_to_load.weight.data[0,0]
        
        # Cleanup temporary file.
        temp.close()


if __name__ == '__main__':
    unittest.main()
