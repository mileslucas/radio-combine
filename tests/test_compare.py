import unittest
from scripts import compare
import numpy as np

class TestCompare(unittest.TestCase):
	
	def test_get_data(self):
		data_path = 'data/orion.gbt.im'
		im = compare.get_data(data_path)
		expected = ['amps', 'smap', 'path', 'noise', 'name']
		self.assertEqual(set(list(im)), set(expected))

	
if __name__=='__main__':
	unittest.main()	

