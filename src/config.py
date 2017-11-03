#-*- coding: utf-8 -*-                                                                                
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--c_num', type=int, default=10)  # Number of classes

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=100)
data_arg.add_argument('--batch_size_test', type=int, default=100)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=2000)
train_arg.add_argument('--epoch_step', type=int, default=100)
train_arg.add_argument('--lr', type=float, default=1e-3)
train_arg.add_argument('--min_lr', type=float, default=1e-4)
train_arg.add_argument('--wd_ratio', type=float, default=5e-2)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--test_iter', type=int, default=1000)
misc_arg.add_argument('--save_step', type=int, default=2000)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='/home/jyh/datasets/')
misc_arg.add_argument('--random_seed', type=int, default=0)

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed    
