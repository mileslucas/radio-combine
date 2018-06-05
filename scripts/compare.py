# Compare.py
# Miles Lucas - mdlucas@nrao.edu

import numpy as np
import argparse
from scipy.optimize import least_squares

# CASA imports
from casac import casac
ia = casac.image()
tb = casac.table()
log = casac.logsink()

def compare(path_a, path_b, regrid=False, plot=True):
	'''
	Compare the given images

	Params
	------
	image_a: str
		The path to the first image
	image_b: str
		The path to the second image
	regrid: bool (optional)
		If true, regrids the second image to the first image's coordinate system. Default=False
	plot: bool (optional)
		If true, creates plots of the power spectra. Default=True

	Returns
	-------
	ratio: dict
		The dictionary of the ratio of each image. See `get_ratio` for more information. 
	'''
	if regrid:
		path_b = regrid_im(path_a, path_b)

	image_a = get_data(path_a)
	image_b = get_data(path_b)

	image_a = get_psd(image_a)
	image_b = get_psd(image_b)

	image_a = mask_psd(image_a)
	image_b = mask_psd(image_b)

	image_a = fit_psd(image_a)
	image_b = fit_psd(image_b)

	ratio = get_ratio(image_a, image_b)

	if plot:
		comparison_plot(image_a, image_b, ratio)
	
	return ratio
	

def regrid_im(image_a, image_b):
	'''
	Regrids image_b from image_a

	Params
	------
	image_a: str
		The path to the first (reference) image
	image_b: str
		The path to the second image)

	Returns
	-------
	outname: str
		The path of the regridded image. This should be the same as <image_b>.regrid
	'''

	ia.open(image_a)
	cs = ia.coordsys()
	ia.close()
	ia.open(image_b)
	tokens = image_b.split('.')
	outname = '.'.join(tokens[:-1]) + '.regrid.' + tokens[-1]
	ib = ia.regrid(outfile=outname, csys=cs.torecord(), shape=ia.shape(), overwrite=True)
	ib.done()
	cs.done()
	ia.done()
	ia.close()

	return outname


def get_data(image_path):
	'''
	Gets the numpy array for a given image_path
	
	Params
	------
	image_path: str
		The path to the image
	
	Returns
	------
	image: dict
		A dictionary containing the following information
		path: str
			The image path
		name: str
			The image name
		amps: ndarray
			Two dimensional array of the image map
		noise: float
			The sigma value of the image for use in evaluating noise threshold later
		smap: dict
			A dictionary holding the length and increments of the skymap axes
	'''

	image = {'path': image_path}
	# Get the Powers
	tb.open(image_path)
	image['amps'] = np.squeeze(tb.getcol('map'))
	tb.close()

	# Get the sky map axes
	ia.open(image_path)
	summ = ia.summary(list=False)
	stats = ia.statistics()
	image['name'] = ia.name().split('/')[-1]
	ia.close()
	
	image['smap'] = {
		'n_x': summ['shape'][0],
		'd_x': summ['incr'][0],
		'n_y': summ['shape'][1],
		'd_y': summ['incr'][1],
	}
	image['noise'] = stats['sigma']
	return image


def get_psd(image):
	'''
	Creates a power spectrum density (PSD) from given skymap axis information and 2D amplitudes

	Params
	-------
	image: dict
		The relevant image information. See output of `get_data` for expected items

	Returns
	-------
	image: dict
		A dictionary containing all the original information plus the following:
		psd: dict
			A dictionary containing the uvdist and power
	'''
	# Get the frequencies
	us = np.fft.fftfreq(image['smap']['n_x'], image['smap']['d_x'])
	vs = np.fft.fftfreq(image['smap']['n_y'], image['smap']['d_y'])
	fft = np.fft.fft2(image['amps'])
	uvdist = []
	power = []
	for i, u in enumerate(us):
		for j, v in enumerate(vs):
			uvdist.append(np.linalg.norm((u, v)))
			power.append(np.abs(fft[i, j]))

	# This will sort the arrays by uvdist but maintain link between distance and power
	uvdist, power = zip(*sorted(zip(uvdist, power)))	
	
	image['psd'] = {
		'uv': np.array(uvdist),
		'pow': np.array(power)
	}

	return image

def mask_psd(image, nsigma=2, num_samps=1000):
	'''
	Masks the psd based on the noise floor of the original image

	Params
	------
	image: dict
		The dictionary of image information. The passed dictionary requires 
		a 'psd' dictionary with 'uv' and 'pow' keys as well as 
	nsigma: int or float
		The number of sigma above which to accept. 

	num_samps: int
		The number of random samples for sampling the noise of the image

	Returns
	-------
	image: dict
		The same input dictionary with an additional `mask_psd` dict that
		mimics the original psd with masking and the threshold
	'''
	
	samps = np.random.normal(loc=0, scale=image['noise'], size=num_samps)
	ft_samps = np.fft.fft(samps)
	image['ft_noise']= np.mean(np.abs(ft_samps))
	thresh = nsigma * image['ft_noise']
	mask = image['psd']['pow'] > thresh
	image['mask_psd'] = {
		'pow': image['psd']['pow'][mask],
		'uv': image['psd']['uv'][mask],
		'thresh': thresh
	}
	return image

def fit_psd(image):
	'''
	'''

	def model(p, x):
		W, alpha, A, s = p
		sinc = np.array([np.sin(np.pi * xi / alpha) / (np.pi * xi / alpha) if xi != 0 else 1 for xi in x])
		gaus = A / s * np.exp(-0.5 * (x / s)**2)
		
		return W + sinc**2 * gaus
	def errfunc(p, x, y):
		return model(p, x) - y
	p0 = [0, 1e4, max(image['psd']['pow']), 3e4]
	bounds = (0, np.inf)
	res = least_squares(errfunc, p0, args=(image['psd']['uv'], image['psd']['pow']), bounds=bounds, loss='cauchy')

	log.post(repr(res.x))
	
	image['fit_params'] = res.x
	image['best_fit'] = lambda x: model(res.x, x)
			
	return image



def get_ratio(image_a, image_b, bin_width=100):
	'''
	Interpolates the psd of both images and gets the ratio of those interpolations.

	Params
	-----
	image_a: dict
		The image dictionary for the first image
	image_b: dict
		The image dictionary for the second image
	bin_width: int (optional)
		The width (in wavelengths) of each bin for interpolation. Default=100

	Returns
	-------
	ratio: dict
		A dictionary with the following keys and values
		uv: array-like
			The uv distance of the interpolated values
		pow_a: array-like
			The interpolated power from image a
		pow_b: array-like
			The interpolated power from image b
		ratio: array-like
			The PSD power ratio (b/a) of the interpolated values
		err: array-like
			The pointwise error of the power ratio
		
	'''
	
	uv = np.arange(0, min((max(image_a['psd']['uv']), max(image_a['psd']['uv']))), bin_width)
	pow_a = image_a['best_fit'](uv)
	pow_b = image_b['best_fit'](uv)

	ratio = pow_b / pow_a
	err = 1 / (np.mean((pow_b, pow_a), axis=0))
	
	return {'uv':uv, 'pow_a': pow_a, 'pow_b': pow_b, 'ratio':ratio, 'err':err}


def comparison_plot(image_a, image_b, ratio, save=None):
	'''
	Plots comparison of the two power spectrum. This method will throw an exception if 
	there is no display. 
	
	Params
	------
	image_a: dict
		The dictionary with values for the first image. Must have masked psd.

	image_b: dict
		The dictionary with values for the second image. Must have masked psd.

	ratio: dict
		A dictionary for the ratio between the two images. See `get_ratio` for more information.

	save: str (optional)
		The filename to save the comparison plot. If none, the plot will not be saved. Default=None.
	'''
	import matplotlib.pyplot as plt
	import matplotlib as mpl

	# Style
	#mpl.style.use('seaborn')
	style = {
		'figure.figsize': (9, 6),
		'axes.labelsize': 14,
		'axes.titlesize': 16,
		'legend.fontsize': 14,
	}
	mpl.rcParams.update(style)

	line_props = {
		'ls': '',
		'marker': '.',
		'ms': 4,
		'mew': 0,
		'alpha': 0.4,
		'lw': 1,
	}

	# Plots
	grid = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])
	fig = plt.figure(figsize=(18,9))
	ax1 = plt.subplot(grid[0,0])
	plt.semilogy(image_a['psd']['uv']/1000, image_a['psd']['pow'], c='.5', **line_props)
	plt.semilogy(ratio['uv']/1000, ratio['pow_a'], c='b')
	plt.ylabel('Power')
	plt.title(image_a['name'])

	ax2 = plt.subplot(grid[0,1], sharex=ax1, sharey=ax1)
	plt.semilogy(image_b['psd']['uv']/1000, image_b['psd']['pow'], c='.5', **line_props)
	plt.semilogy(ratio['uv']/1000, ratio['pow_b'], c='g')
	plt.title(image_b['name'])
	plt.xlabel(r'UV Distance ($k\lambda$)')
	ax2.get_yaxis().set_visible(False)

	ax3 = plt.subplot(grid[0, 2], sharex=ax1)
	scale = 0.1 / min(ratio['err'])
	plt.errorbar(ratio['uv']/1000, ratio['ratio'], yerr=scale * ratio['err'], fmt='ro', ecolor='0.3', barsabove=True)
	plt.title('Comparison of PSD')
	ax3.yaxis.tick_right()
	ax3.yaxis.set_label_position('right')
	plt.xlim(-0.25, None)
	plt.axhline(1, ls='--', c='k')
	plt.ylabel('Power Ratio')

	plt.subplots_adjust(wspace=0.0, hspace=0.0)	
	plt.show()
	
	if save is not None:
		plt.savefig(save)

	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('image_a', help='path to the first image')
	parser.add_argument('image_b', help='path to the second image')
	parser.add_argument('-r', '--regrid', action='store_true', help="regrids image_b to image_a's coordinates")
	parser.add_argument('-n', '--no-plot', dest='plot', action='store_false')
	args = parser.parse_args()
	compare(args.image_a, args.image_b, args.regrid, args.plot)
