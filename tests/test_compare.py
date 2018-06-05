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

class TestBinPSD(unittest.TestCase):

	def setUp(self):
		im = compare.get_data(data_path)
		im2 = compare.get_data(data_path)
		im = compare.get_psd(im)
		im2 = compare.get_psd(im2)
		self.uv1 = np.arange(0, 30000, 1000)
		self.uv2 = np.arange(0, 30000, 100)
		self.im1 = compare.bin_psd(im, self.uv1)
		self.im2 = compare.bin_psd(im2, self.uv2)

	def test_bin_psd_exists(self):
		expected = 'bin_psd'
		self.assertIn(expected, list(self.im1))

	def test_bin_psd_keys(self):
		expected = ['uv', 'pow']
		self.assertTrue(all([k in list(self.im1['bin_psd']) for k in expected]))
	
	def test_bin_lengths(self):
		l1 = len(self.im1['bin_psd']['uv'])
		l2 = len(self.im2['bin_psd']['uv'])
		self.assertTrue(l1 < l2)

class TestFitPSD(unittest.TestCase):
	
	def setUp(self):
		im = compare.get_data(data_path)
		self.im = compare.get_psd(im)
		self.fit_im = compare.fit_psd(im)

	def test_fit_psd_exists(self):
		expected = ['best_fit', 'fit_params']
		self.assertTrue(all([k in list(self.fit_im) for k in expected]))

	def test_fit_psd_params(self):
		self.assertEqual(len(self.fit_im['fit_params']), 4)

	def test_fit_psd_fit(self):
		uv = np.linspace(0, 30000)
		y = self.fit_im['best_fit'](uv)
		
		self.assertEqual(uv.shape, y.shape)

	def test_fit_psd_kernels(self):
		fit = compare.fit_psd(self.im, kernel='norm')
		fit2 = compare.fit_psd(self.im, kernel='exp')
		uv = np.linspace(0, 30000)
		self.assertEqual(np.mean(fit['best_fit'](uv) - fit2['best_fit'](uv)), 0)

	def test_fit_psd_kernel_error(self):
		with self.assertRaises(ValueError):
			fit3 = compare.fit_psd(self.im, kernel='asd;fajsl;fa')




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

