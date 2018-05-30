import unittest
from scripts import compare
import numpy as np

class TestCompare(unittest.TestCase):
	
	def test_get_arr(self):
		data_path = 'data/orion.gbt.im'
		arr = compare.get_arr(data_path)
		self.assertEqual(arr.shape, (300,300))

	def test_arr_transform(self):
		fx = np.linspace(1, 1000, 101)
		fy = np.linspace(1, 1000, 101)
		amps = np.random.normal(loc=100, scale=10, size=(101, 101))
		r, a = compare.arr_transform(fx, fy, amps)
		self.assertEqual(len(a), amps.shape[0]*amps.shape[1])
		self.assertFalse(min(r) < 0)
		self.assertEqual(len(r), len(a))


if __name__=='__main__':
	unittest.main()	

