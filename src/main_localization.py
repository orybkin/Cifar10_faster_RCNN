import numpy as np                                                                                    
import tensorflow as tf

from trainer_localization import Trainer
from config import get_config
from data_loader_localization import get_loader
from utils import prepare_dirs_and_logger, save_config
from models import *

def main(config, model):
  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  train_data_loader, train_label_loader, train_location_loader, train_size_loader, train_msk_loader = get_loader(
    config.data_path, config.batch_size, 'train', True)

  if config.is_train:
    test_data_loader, test_label_loader, test_location_loader, test_size_loader, test_msk_loader = get_loader(
      config.data_path, config.batch_size_test, 'test', False)
  else:
    test_data_loader, test_label_loader, test_location_loader, test_size_loader, test_msk_loader = get_loader(
      config.data_path, config.batch_size_test, config.split, False)

  trainer = Trainer(config, train_data_loader, train_label_loader, test_data_loader, test_label_loader,
                    train_location_loader, train_size_loader,  test_location_loader, test_size_loader,
                    train_msk_loader, test_msk_loader, model)
  if config.is_train:
    save_config(config)
    trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

if __name__ == "__main__":
  config, unparsed = get_config()

  # Task 2

  # Subtask 1
  model=RCNN
  main(config, model)

