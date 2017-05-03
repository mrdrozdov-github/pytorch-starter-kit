from __future__ import print_function

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
import sys
import time
import json
import random

from starter.utils.fs import *
from starter.utils.afs_safe_logger import Logger

import gflags

FLAGS = gflags.FLAGS


def run():
    if not os.path.exists(fs_log_dir(FLAGS)):
        os.mkdir(fs_log_dir(FLAGS))

    logger = Logger(fs_log_path(FLAGS))

    flags_json = json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True)
    logger.Log("Flag Values:\n" + flags_json)
    with open(fs_flags_json_path(FLAGS), 'w') as f:
        f.write(flags_json)

    model = None

    # Data loading code
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}
    tr_loader = torch.utils.data.DataLoader(
        datasets.MNIST(FLAGS.tr_data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=FLAGS.tr_batch_size, shuffle=True)
    ev_loader = torch.utils.data.DataLoader(
        datasets.MNIST(FLAGS.ev_data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=FLAGS.ev_batch_size, shuffle=True, drop_last=True, **kwargs)

    # Train Loop
    epoch = 0
    # model.train()
    for batch_idx, (data, target) in enumerate(tr_loader):
        if FLAGS.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        # optimizer.zero_grad()
        # output = model(data)
        # loss = F.nll_loss(output, target)
        # loss.backward()
        # optimizer.step()
        # loss_val = loss.data[0]
        loss_val = 0.0
        if batch_idx % FLAGS.log_interval == 0:
            logger.Log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(tr_loader.dataset),
                100. * batch_idx / len(tr_loader), loss_val))

        # Eval Loop
        if batch_idx % FLAGS.eval_interval == 0:
            # model.eval()
            test_loss = 0
            correct = 0
            for data, target in ev_loader:
                if FLAGS.cuda:
                    data, target = data.cuda(), target.cuda()
                # data, target = Variable(data, volatile=True), Variable(target)
                # output = model(data)
                # test_loss += F.nll_loss(output, target).data[0]
                # pred = output.data.max(1)[1] # get the index of the max log-probability
                # correct += pred.eq(target.data).cpu().sum()

            test_loss = test_loss
            test_loss /= len(ev_loader) # loss function already averages over batch size
            logger.Log('Dev set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(ev_loader.dataset),
                100. * correct / len(ev_loader.dataset)))

def load_json(load_json_path):
    loaded_flags = json.load(open(load_json_path))
    return loaded_flags


def flags():
    # Define Command Line Args
    # ========================

    # Convenience settings.
    gflags.DEFINE_bool("debug", False, "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_string("branch_name", "", "")
    gflags.DEFINE_string("sha", "", "")
    gflags.DEFINE_string("experiment_name", "", "")
    gflags.DEFINE_string("load_json_path", None, "")

    # Performance settings.
    gflags.DEFINE_bool("cuda", False, "")

    # Log settings.
    gflags.DEFINE_integer("log_interval", 100, "")
    gflags.DEFINE_string("log_path", "./logs", "A directory in which to write logs.")

    # Checkpoint settings.
    gflags.DEFINE_integer("checkpoint_interval", 1000, "")
    gflags.DEFINE_string("load_checkpoint", None, "")

    # Evaluation settings.
    gflags.DEFINE_integer("eval_interval", 100, "")
    gflags.DEFINE_boolean("early_stopping", True, "Save checkpoint when surpassing validation error.")

    # Training data settings.
    gflags.DEFINE_integer("tr_batch_size", 32, "")
    gflags.DEFINE_string("tr_data_path", "~/data/mnist", "")

    # Evaluation data settings.
    gflags.DEFINE_integer("ev_batch_size", 32, "")
    gflags.DEFINE_string("ev_data_path", "~/data/mnist", "")

    # Model settings.
    gflags.DEFINE_integer("model_dim", 100, "")

    # Optimization settings.
    gflags.DEFINE_float("lr", 0.01, "")
    gflags.DEFINE_enum("opt", "Adam", ["SGD", "Adam"], "")

    # Parse Command Line Args
    # =======================

    FLAGS(sys.argv)

    # Flag Defaults
    # =============

    FLAGS.tr_data_path = os.path.expanduser(FLAGS.tr_data_path)
    FLAGS.ev_data_path = os.path.expanduser(FLAGS.ev_data_path)

    if FLAGS.load_json_path:
        loaded_flags = load_json(load_json_path)
        for k in loaded_flags.keys():
            setattr(FLAGS, k, loaded_flags[k])

        # Optionally override loaded flags.
        FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}".format(
            timestamp,
            )

    if not FLAGS.branch_name:
        FLAGS.branch_name = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    if not torch.cuda.is_available():
        FLAGS.cuda = False


if __name__ == '__main__':
    flags()
    run()
