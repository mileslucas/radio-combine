import unittest
from scripts import compare
import numpy as np

class TestCompare(unittest.TestCase):
	
	def setUp(self):	
		data_path = 'data/orion.gbt.im'
		self.im = compare.get_data(data_path)

	def test_get_data(self):
		expected = ['amps', 'smap', 'path', 'noise', 'name']
		self.assertEqual(set(list(self.im)), set(expected))

	def test_get_psd_exists(self):
		im = compare.get_psd(self.im)
		self.assertTrue('psd' in list(im))

	def test_get_psd_keys(self):
		im = compare.get_psd(self.im)
		expected = ['uv', 'pow']  
		self.assertTrue(all([k in list(im['psd']) for k in expected]))

	def test_get_psd_vals(self):
		im = compare.get_psd(self.im)
		self.assertEqual(im['psd']['uv'].shape, im['psd']['pow'].shape)
if __name__=='__main__':
	unittest.main()	

