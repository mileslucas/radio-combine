# Compare.py
# Miles Lucas - mdlucas@nrao.edu

import numpy as np
import argparse

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
	psds: ndarray
		This is a 2x2 array with the two power spectrum densities for each image. The format is
		[psd_a, psd_b] where each psd consists of [freq, pow]. 
	'''
	if regrid:
		path_b = regrid_im(image_a, image_b)
	image_a = get_data(path_a)
	image_b = get_data(path_b)

	r_a, pow_a, ft_noise_a = get_psd(smap_a, amps_a, noise_a)
	r_b, pow_b, ft_noise_b = get_psd(smap_b, amps_b, noise_b)

	if plot:
		comparison_plot(r_a, pow_a, ft_noise_a,  image_a, r_b, pow_b, ft_noise_b, image_b)
	

	# Verify Fourier transform property
	idx = r_a == 0
	log.post('Image: {}\nTotal power: {}\nMatches image: {}\n'.format(image_a, np.real(pow_a[idx][0]), np.sum(amps_a) == pow_a[idx]))
	idx = r_b == 0
	log.post('Image: {}\nTotal power: {}\nMatches image: {}\n'.format(image_b, np.real(pow_b[idx][0]), np.sum(amps_b) == pow_b[idx]))
	return [[r_a, pow_a], [r_b, pow_b]]

	

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
	image['name'] = ia.name()
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
		'uv': uvdist,
		'pow': power
	}

	return image

def mask_psd(image, num_samps=1000, threshold=2):
	'''
	'''
	
	samps = np.random.normal(loc=0, scale=image['noise'], size=num_samps)
	ft_samps = np.fft.fft(samps)
	image['ft_noise']= np.mean(np.abs(ft_samps))

	mask = image['psd']['pow'] > threshold * image['ft_noise']
	image['mask_psd'] = {
		'pow': image['psd']['pow'][mask],
		'uv': image['psd']['uv'][mask]
	}
	return image


def comparison_plot(r_a, pow_a, ft_noise_a,  name_a, r_b, pow_b, ft_noise_b, name_b):
	'''
	Plots comparison of the two power spectrum. This method will throw an exception if 
	there is no display. 
	
	Params
	------
	r_a: array-like
		The distances of the psd for the first image
	pow_a: array-like
		The powers of the psd for the first image
	name_a: str
		The name of the first image or dataset
	r_b: array-like
		The distances of the psd for the second image
	pow_b: array-like
		The powers of the psd for the second image
	name_b: str
		The name of the second image or dataset
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
	# Mask the Data
	thresh_a = 2 * ft_noise_a
	mask_a = pow_a >  thresh_a
	r_a = r_a[mask_a]
	pow_a = pow_a[mask_a]

	thresh_b = 2 * ft_noise_b
	mask_b = pow_b > thresh_b
	r_b  = r_b[mask_b]
	pow_b = pow_b[mask_b]

	# Get bin the data
	bin_width = 100
	uv = np.arange(0, min((max(r_a), max(r_b))), bin_width)
	int_pow_a = np.interp(uv, r_a, pow_a)
	int_pow_b = np.interp(uv, r_b, pow_b)

	ratio = int_pow_b / int_pow_a
	err = 1 / (np.mean((int_pow_b, int_pow_a), axis=0))
	scale = .1 / min(err)

	# Plots
	grid = plt.GridSpec(2, 3, width_ratios=[1, 1, 2])
	fig = plt.figure(figsize=(18,9))
	ax1 = plt.subplot(grid[0,0])
	plt.semilogy(r_a/1000, pow_a, c='b', **line_props)
	plt.ylabel(name_a)
	plt.title('PSD')
	plt.gca().get_xaxis().set_visible(False)

	plt.subplot(grid[1,0], sharex=ax1, sharey=ax1)
	plt.semilogy(r_b/1000, pow_b, c='g', **line_props)
	plt.ylabel(name_b)

	ax3 = plt.subplot(grid[0,1], sharey=ax1)
	plt.semilogy(uv/1000, int_pow_a, 'b.', mew=0)
	plt.title('Interpolated PSD')
	plt.gca().get_xaxis().set_visible(False)
	plt.gca().get_yaxis().set_visible(False)
	plt.xlim(-0.25, None)
	
	plt.subplot(grid[1,1], sharex=ax3, sharey=ax1)
	plt.semilogy(uv/1000, int_pow_b, 'g.', mew=0)
	plt.gca().get_yaxis().set_visible(False)
	plt.xlim(-0.25, None)

	ax5 = plt.subplot(grid[:, 2], sharex=ax3)
	plt.errorbar(uv/1000, ratio, yerr=scale*err, fmt='o')
	plt.title('Comparison of PSD')
	ax5.yaxis.tick_right()
	plt.xlim(-0.25, None)
	plt.axhline(1, ls='--', c='k')

	
	fig.text(0.04, 0.5, 'Power', fontsize=14, va='center', rotation = 'vertical')
	fig.text(0.5, 0.04, r'UV Distance ($k\lambda$)', ha='center', fontsize=14)
	fig.text(0.96, 0.5, r'Power ratio', va='center', fontsize=14, rotation='vertical')

	plt.subplots_adjust(wspace=0.0, hspace=0.0)	
	plt.show()


	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('image_a', help='path to the first image')
	parser.add_argument('image_b', help='path to the second image')
	parser.add_argument('-r', '--regrid', action='store_true', help="regrids image_b to image_a's coordinates")
	parser.add_argument('-n', '--no-plot', dest='plot', action='store_false')
	args = parser.parse_args()
	compare(args.image_a, args.image_b, args.regrid, args.plot)
