from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grip_input_data as gid

def _get_data():
  return {'x':[1,2,3,4,5,6], 'y':[1,1,1,1,1,1]}

if __name__ == '__main__':
  data = {'x':[], 'y':[]}
  data += _get_data()
  print(data)
  data = gid.GripData('./', ['others'])
  data._load_raw_data('./')