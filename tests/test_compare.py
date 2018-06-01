import unittest
from scripts import compare
import numpy as np

class TestCompare(unittest.TestCase):
	
	def test_get_arr(self):
		data_path = 'data/orion.gbt.im'
		arr = compare.get_arr(data_path)
		self.assertEqual(arr.shape, (300,300))

	def test_get_psd(self):
		s = np.linspace(-50, 50, 101)
		u = np.fft(sx)
		amps = np.random.normal(loc=100, scale=10, size=(101, 101))
		r, a = compare.get_psd(dict(n_x=101, d_x=1, n_y=101, d_y=1), amps)
		self.assertEqual(len(a), amps.shape[0]*amps.shape[1])
		self.assertFalse(min(r) < 0)
		self.assertEqual(len(r), len(a))
		self.assertEqual(max(r), np.linalg.norm((50, 50)))


if __name__=='__main__':
	unittest.main()	

