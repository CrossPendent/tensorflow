from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest as test
import grip_input_data as gid

class GripInputDataTest(test.TestCase):
  def setUp(self):
    self._dataset_dir = './grip_dataset'
    self._labels = ['other', 'slide_up', 'slide_down']
    self.grip_data = gid.GripData(self._dataset_dir, self._labels)

  def testGetNextTrainBatch(self):
    batch_x, batch_y = self.grip_data.getNextTrainBatch(100)
    train_data, test_data = self.grip_data.getData()
    count = 0

    self.assertEqual(len(batch_x), len(batch_y))
    for _ in range(len(batch_x)//100+1):
      self.assertEqual(len(batch_x), 100)
      for i in range(len(batch_x)):
        if batch_x[i] != train_data['x'][i]:
          count += 1
    self.assertNotEqual(count, 0)

  def testGetTestData(self):
    test_data = self.grip_data.getTestData()
    self.assertIn('x', test_data.keys())
    self.assertIn('y', test_data.keys())
    for label in self._labels:
      self.assertIn(label, test_data['y'])

  def testGetData(self):
    train_data, test_data = self.grip_data.getData()

    self.assertIn('x', train_data.keys())
    self.assertIn('y', train_data.keys())
    self.assertIn('x', test_data.keys())
    self.assertIn('y', test_data.keys())
    for label in self._labels:
      self.assertIn(label, train_data['y'])
      self.assertIn(label, test_data['y'])


if __name__ == "__main__":
  test.main(verbosity=2)
