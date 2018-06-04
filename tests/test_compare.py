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
		im = compare.get_data(data_path)
		im = compare.get_psd(im)
		self.im = compare.mask_psd(im)

	def test_mask_psd_exists(self):
		expected = ['mask_psd', 'ft_noise']
		self.assertTrue(all([k in list(self.im) for k in expected]))

	def test_mask_psd_keys(self):
		expected = ['uv', 'pow', 'thresh']  
		self.assertTrue(all([k in list(self.im['mask_psd']) for k in expected]))

	def test_mask_psd_vals(self):
		t1 = self.im['mask_psd']['thresh']
		im2 = compare.mask_psd(self.im, nsigma=3)
		self.assertTrue(im2['mask_psd']['thresh'] > t1)
	
class TestGetRatio(unittest.TestCase):

	def setUp(self):
		im1 = compare.get_data(data_path)
		im1 = compare.get_psd(im1)
		self.im1 = compare.mask_psd(im1)
		
		im2 = compare.get_data('data/orion.gbt.noisy.im')
		im2 = compare.get_psd(im2)
		self.im2 = compare.mask_psd(im2)
		self.ratio = compare.get_ratio(self.im1, self.im2)

	def test_ratio_keys(self):
		expected = ['uv', 'pow_a', 'pow_b', 'ratio', 'err']
		self.assertTrue(all([k in list(self.ratio) for k in expected]))

	def test_ratio_shapes(self):
		self.assertEqual(self.ratio['uv'].shape, self.ratio['ratio'].shape)
		self.assertEqual(self.ratio['uv'].shape, self.ratio['err'].shape)
		self.assertEqual(self.ratio['uv'].shape, self.ratio['pow_a'].shape)
		self.assertEqual(self.ratio['uv'].shape, self.ratio['pow_b'].shape)

	def test_err_vals(self):
		expect = 1 / np.mean((self.ratio['pow_a'][0], self.ratio['pow_b'][0]))
		self.assertTrue(np.isclose(self.ratio['err'][0], expect))

	def test_interp_width(self):
		r2 = compare.get_ratio(self.im1, self.im2, bin_width=50)
		self.assertTrue(len(r2['uv']) > len(self.ratio['uv']))

				
if __name__=='__main__':
	unittest.main()	

