import unittest
from scripts import compare
import numpy as np

data_path = 'data/orion.gbt.im'

class TestGetData(unittest.TestCase):
	
	def setUp(self):	
		self.im = compare.get_data(data_path)

	def test_get_data(self):
		expected = ['amps', 'smap', 'path', 'noise', 'name']
		self.assertEqual(set(list(self.im)), set(expected))

	def test_get_data_smap_keys(self):
		expected = ['n_x', 'd_x', 'n_y', 'd_y']
		self.assertTrue(all([k in list(self.im['smap']) for k in expected]))

class TestGetPSD(unittest.TestCase):

	def setUp(self):
		im = compare.get_data(data_path)
		self.im = compare.get_psd(im)

	def test_get_psd_exists(self):
		self.assertTrue('psd' in list(self.im))

	def test_get_psd_keys(self):
		expected = ['uv', 'pow']  
		self.assertTrue(all([k in list(self.im['psd']) for k in expected]))

	def test_get_psd_vals(self):
		self.assertEqual(self.im['psd']['uv'].shape, self.im['psd']['pow'].shape)

class TestMaskPSD(unittest.TestCase):

	def setUp(self):
		im = compare.get_data
		im = compare.get_psd(im)
		self.im = compare.mask_psd(im)

	def test_mask_psd_exists(self):
		self.assertTrue('mask_psd' in list(self.im))

	def test_mask_psd_keys(self):
		expected = ['uv', 'pow']  
		self.assertTrue(all([k in list(self.im['psd']) for k in expected]))

	def test_mask_psd_vals(self):
		im2 = compare.mask_psd(self.im, threshold=3)
		self.assertTrue(im2['mask_psd']['thresh'] > self.im['mask_psd']['thresh'])
	

if __name__=='__main__':
	unittest.main()	

