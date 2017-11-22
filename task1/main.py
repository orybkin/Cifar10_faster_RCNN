import numpy as np                                                                                    
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader_classification import get_loader
from utils import prepare_dirs_and_logger, save_config
from models import *

def main(config, model):
  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  train_data_loader, train_label_loader = get_loader(
    config.data_path, config.batch_size, config, 'train', True)

  if config.is_train:
    test_data_loader, test_label_loader = get_loader(
      config.data_path, config.batch_size_test, config, 'test', False)
  else:
    test_data_loader, test_label_loader = get_loader(
      config.data_path, config.batch_size_test,  config, config.split, False)

  trainer = Trainer(config, train_data_loader, train_label_loader, test_data_loader, test_label_loader, model)
  if config.is_train:
    save_config(config)
    trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()


if __name__ == "__main__":
  config, unparsed = get_config()

  # Task 1

  # # Subtask 1
  # model=ConvNet
  # main(config, model)
  #
  # # Subtask 2
  # model=MobileNet
  # main(config, model)

  # Subtask 3
  model=ResNet
  main(config, model)