from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

if __name__ == '__main__':
  x = np.array(range(1,10), dtype=np.int32)
  print(x)
  np.random.shuffle(x)
  print(x)
