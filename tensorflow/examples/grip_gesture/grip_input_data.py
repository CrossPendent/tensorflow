from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import sys
import numpy as np

DATA_PATH_PREFIX = 'log_'
DATA_PATH_TRAIN = '{}train'.format(DATA_PATH_PREFIX)
DATA_PATH_TEST = '{}test'.format(DATA_PATH_PREFIX)


def _get_data_from_file(file_path):
  data = []
  file_count = 1
  while True:
    file_name = '{}/{:04d}.txt'.format(file_path, file_count)
    if not os.path.exists(file_name):
      break
    with open(file_name, 'r') as f:
      file_records = f.read().splitlines()
      for record in file_records:
        if len(record) <= 1:
          continue
        record = record.split(',')
        data += [float(record[i * 3]) for i in range(3)]  # Just left channels are taken
    file_count += 1
  return data


class GripData(object):
  def __init__(self, data_dir, labels):
    self.base_dir = data_dir
    self.train_dir = '{}/{}'.format(data_dir, DATA_PATH_TRAIN)
    self.test_dir = '{}/{}'.format(data_dir, DATA_PATH_TEST)
    self.labels = labels
    self.train_data = {'x':[], 'y':[]}
    self.test_data = {'x':[], 'y':[]}
    self._last_batched_point = 0
    self._batch_loop_count = 0
    self._load_raw_data()

  def getNextTrainBatch(self, batch_size):

    start = self._last_batched_point

    if self._batch_loop_count is 0 and start is 0:
      shuffle_index = np.arange(0, len(self.train_data['x']))
      np.random.shuffle(shuffle_index)

    if start+batch_size > len(self.train_data['x']):
      batch_index = shuffle_index[start:]
      rest_item_size = batch_size-start
      np.random.shuffle(shuffle_index)

      batch_index += shuffle_index[:rest_item_size]
      self._last_batched_point = rest_item_size
    else:
      self._last_batched_point += batch_size
      end = self._last_batched_point
      batch_index = shuffle_index[start:end]

    print('batch_index={}'.format(batch_index))

    return [self.train_data['x'][i] for i in batch_index], [self.train_data['y'][i] for i in batch_index]

  def getTestData(self):
    return self.test_data

  def getData(self):
    return self.train_data, self.test_data

  def _load_raw_data(self):
    for label in self.labels:
      train_label_path = '{}/{}'.format(self.train_dir, label)
      _x = _get_data_from_file(train_label_path)
      self.train_data['x'] += _x
      self.train_data['y'] += [label for _ in range(len(_x))]

      if len(self.train_data['x']) is 0:
        print('Could not load training dataset from \'{}\'.'.format(os.path.abspath(train_label_path)))

      test_label_path = '{}/{}'.format(self.test_dir, label)
      _x = _get_data_from_file(test_label_path)
      self.test_data['x'] += _x
      self.test_data['y'] += [label for _ in range(len(_x))]
      if len(self.test_data['x']) is 0:
        print('Could not load test dataset from \'{}\'.'.format(os.path.abspath(test_label_path)))
