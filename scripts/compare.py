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

def compare(path_a, path_b, regrid=False, binwidth=500,  plot=True, save=None):
	'''
	Compare the given images

	Params
	------
	image_a: str
		The path to the first image
	image_b: str
		The path to the second image
	regrid: bool (optional)
		If true, regrids the second image to the first image's coordinate system. 
		Default=False
	binwidth: int (optional)
		The width of the bins for averaging the psds when creating the ratio. Default=500
	plot: bool (optional)
		If true, creates plots of the power spectra. Default=True
	save: str (optional)
		If not None, will save the produced plot at this filename. Default=None

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

	ratio = get_ratio(image_a, image_b, binwidth)

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

def bin_psd(image, x):
	'''
	This method will average-bin the psd. It will maintain the 0 uv point and
	average the next (x[i], x[i+1]] range.

	Params
	------
	image: dict
		The image to average. It must have a 'psd' dictionary inside it
	x: array-like
		The uv array that marks the edges of the bins.

	Returns
	-------
	image: dict
		The same image as input but with a new 'bin_psd' dictionary containing
		'uv' and 'pow' arrays
	'''
	vals = []
	vals.append(image['psd']['pow'][0])
	for i in range(1, len(x)):
		low = x[i-1]
		high = x[i]
		mask = (image['psd']['uv'] > low) & (image['psd']['uv'] <= high)
		mean_pow = np.mean(image['psd']['pow'][mask])
		vals.append(mean_pow)

	image['bin_psd'] = {
		'uv':np.array(x),
		'pow':np.array(vals),
	}
	return image

def fit_psd(image, kernel='gaussian'):
	'''
	This will attempt to least squares fit the psd with one of the following models.

	$$ f(\lambda) = W + R(\lambda) * K(\lambda) $$
	where $R$ is the response function
	$$ R(\lambda) = \frac{\sin{\pi \lambda / \alpha}}{\pi \lambda / \alpha} $$
	and $K$ is the kernel. If the kernel is exponential 
	$$ K = A * e ^ {-\lambda / s} $$
	If the kernel is Gaussian
	$$ K = A * e ^ {-0.5 (\lambda / s)^2 } $$

	So there are four free parameters: $W$, $\alpha$, $A$, $s$. The fitting is accomplished
	by least-squares fitting with a Cauchy loss function. Each of the parameters is bounded
	from $(0, \infty)$. 

	Params
	------
	image: dict
		The image dictionary. Must have a 'psd' dicitonary with 'uv' and 'pow' arrays
	kernel: str {'gaussian', 'exponential'}
		The kernel to fit with the model.
	
	Returns
	-------
	image: dict
		The image inputted with two new values:
		'fit_params': array
			The array of best fit parameters
		'best_fit': func
			The model function with the best fit parameters. To get a fit, just
			input the lambda values.
	'''

	if kernel.lower() in ['gaussian', 'gauss', 'normal', 'norm']:
		kern = lambda A, s, l: A * np.exp( -0.5 * (l / s)**2)
	elif kernel.lower() in ['exponential', 'expon', 'exp']:
		kern = lambda A, s, l: A * np.exp( -l / s)
	else:
		raise ValueError('Not a recognized kernel')

	def model(p, l):
		W, alpha, A, s = p
		sinc = np.array([np.sin(np.pi * li / alpha) / (np.pi * li / alpha) 
			if li != 0 else 1 for li in l])
		
		return W + sinc**2 * kern(A, s, l)

	def errfunc(p, l, y):
		return model(p, l) - y

	p0 = [0, 1e4, max(image['psd']['pow']), 3e4]
	bounds = (0, np.inf)
	res = least_squares(errfunc, p0, args=(image['psd']['uv'], 
		image['psd']['pow']), bounds=bounds, loss='cauchy')

	log.post('Fit params for {}'.format(image['name']))
	log.post('\t W={}, alpha={}, A={}, s={}\n'.format(*res.x))
	
	image['fit_params'] = res.x
	image['best_fit'] = lambda l: model(res.x, l)
			
	return image

def get_ratio(image_a, image_b, bin_width=500):
	'''
	Interpolates the psd of both images and gets the ratio of those interpolations.

	Params
	-----
	image_a: dict
		The image dictionary for the first image
	image_b: dict
		The image dictionary for the second image
	bin_width: int (optional)
		The width (in wavelengths) of each bin. Default=500

	Returns
	-------
	ratio: dict
		A dictionary with the following keys and values
		uv: array-like
			The uv distance of the binned values
		pow_a: array-like
			The binned power from image a
		pow_b: array-like
			The binned power from image b
		ratio: array-like
			The PSD power ratio (b/a) of the binned values
		err: array-like
			The pointwise error of the power ratio
	'''
	
	uv = np.arange(0, min((max(image_a['psd']['uv']), max(image_a['psd']['uv']))), bin_width)
	image_a = bin_psd(image_a, uv)
	image_b = bin_psd(image_b, uv)
	pow_a = image_a['bin_psd']['pow']
	pow_b = image_b['bin_psd']['pow']

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
		A dictionary for the ratio between the two images. See `get_ratio` for 
		more information.

	save: str (optional)
		The filename to save the comparison plot. If none, the plot will not 
		be saved. Default=None.
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
	fig = plt.figure(figsize=(21,7))
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
	plt.errorbar(ratio['uv']/1000, ratio['ratio'], yerr=scale * ratio['err'], 
			fmt='ro', ecolor='0.3',)
	plt.yscale('log')
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
	parser = argparse.ArgumentParser(description='Compare the two images by getting \
			the PSD, binning them, and getting the ratio of the two. Ideally \
			this raito should be 1.0 near the 0 uv point')
	parser.add_argument('image_a', help='path to the first image')
	parser.add_argument('image_b', help='path to the second image')
	parser.add_argument('-r', '--regrid', action='store_true', 
		help="regrids image_b to image_a's coordinates")
	parser.add_argument('-w', '--width', type=int, metavar='binwidth', default=500,
		help='The binwidth for binning the PSDs')
	parser.add_argument('-s', '--save', metavar='filename', default=None,
		help='Save the final image at the given filename.')
	parser.add_argument('--no-plot', dest='plot', action='store_false', 
		help='Does not plot the final output. Useful if no X11 display')
	args = parser.parse_args()

	compare(args.image_a, args.image_b, args.regrid, args.width, args.plot, args.save)
