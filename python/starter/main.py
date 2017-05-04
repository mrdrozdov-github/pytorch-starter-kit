from __future__ import print_function
from builtins import range

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch.nn.init
nn.init = torch.nn.init

import os
import sys
import time
import json
import random

from starter.utils.afs_safe_logger import Logger
from starter.utils.torch_utils import torch_save, torch_load

import gflags

FLAGS = gflags.FLAGS


class Model(nn.Module):
    def __init__(self, inp_dim, outp_dim):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(inp_dim, 100),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(100, outp_dim),
            )

        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                print("Initializing {} with kaiming_normal.".format(name))
                nn.init.kaiming_normal(p.data, mode='fan_in')
            if 'bias' in name:
                print("Initializing {} to 0.".format(name))
                p.data.fill_(0)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten
        return self.layer(x)


def run_dev(args):
    model = args['model']
    ev_loader = args['ev_loader']
    logger = args['logger']

    model.eval()
    test_loss = 0
    correct = 0
    for data, target in ev_loader:
        if FLAGS.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        output = F.softmax(output)
        test_loss += nn.NLLLoss()(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(ev_loader) # loss function already averages over batch size
    accuracy = 100. * correct / len(ev_loader.dataset)
    logger.Log('Dev set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(ev_loader.dataset), accuracy))

    result = dict()
    result['accuracy'] = accuracy
    result['loss'] = test_loss

    return result


def run():
    log_dir = os.path.join(FLAGS.log_dir, FLAGS.experiment_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logger = Logger(FLAGS.log_path)

    flags_json = json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True)
    logger.Log("Flag Values:\n" + flags_json)
    with open(FLAGS.json_path, 'w') as f:
        f.write(flags_json)

    inp_dim = 28 * 28
    outp_dim = 10
    model = Model(inp_dim, outp_dim)
    optimizer = getattr(optim, FLAGS.opt)(model.parameters(), lr=FLAGS.lr)

    models_dict = dict(classifier=model)
    optimizers_dict = dict(classifier_opt=optimizer)

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

    ev_args = dict()
    ev_args['model'] = model
    ev_args['ev_loader'] = ev_loader
    ev_args['logger'] = logger

    if os.path.exists(FLAGS.checkpoint_path):
        loaded_results = torch_load(models_dict, optimizers_dict, FLAGS.checkpoint_path)
        logger.Log("Loaded model with Accuracy: {}".format(loaded_results.get('accuracy', None)))

    if FLAGS.eval_only:
        assert os.path.exists(FLAGS.checkpoint_path), "Requires checkpoint."
        ev_result = run_dev(ev_args)
        sys.exit()

    # Train Loop
    epoch = 0
    for epoch in range(epoch, FLAGS.max_epochs):
        for batch_idx, (data, target) in enumerate(tr_loader):
            model.train()
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)

            # Get scores
            output = F.softmax(output)

            loss = nn.NLLLoss()(output, target)
            loss.backward()
            optimizer.step()
            loss_val = loss.data[0]
            if batch_idx % FLAGS.log_interval == 0:
                logger.Log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(tr_loader.dataset),
                    100. * batch_idx / len(tr_loader), loss_val))

            # Eval Loop
            if batch_idx % FLAGS.eval_interval == 0:
                ev_result = run_dev(ev_args)

        ev_result = run_dev(ev_args)
        logger.Log('Checkpointing.')
        torch_save(models_dict, optimizers_dict, ev_result,
            FLAGS.checkpoint_path,
            gpu=0 if FLAGS.cuda else -1)


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
    gflags.DEFINE_boolean("eval_only", False, "")

    # Performance settings.
    gflags.DEFINE_bool("cuda", False, "")

    # Log settings.
    gflags.DEFINE_integer("log_interval", 100, "")
    gflags.DEFINE_string("log_dir", "./logs", "A directory in which to write logs.")
    gflags.DEFINE_string("log_path", None, "")
    gflags.DEFINE_string("json_path", None, "Save command line args to json file.")

    # Checkpoint settings.
    gflags.DEFINE_integer("checkpoint_path", None, "")
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
    gflags.DEFINE_integer("max_epochs", 1, "")
    gflags.DEFINE_float("lr", 1e-3, "")
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

    if not FLAGS.log_path:
        FLAGS.log_path = os.path.join(FLAGS.log_dir, FLAGS.experiment_name, FLAGS.experiment_name + '.log')

    if not FLAGS.json_path:
        FLAGS.json_path = os.path.join(FLAGS.log_dir, FLAGS.experiment_name, FLAGS.experiment_name + '.json')

    if not FLAGS.checkpoint_path:
        FLAGS.checkpoint_path = os.path.join(FLAGS.log_dir, FLAGS.experiment_name, FLAGS.experiment_name + '.pt')

    if not FLAGS.branch_name:
        FLAGS.branch_name = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    if not torch.cuda.is_available():
        FLAGS.cuda = False


if __name__ == '__main__':
    flags()
    run()
